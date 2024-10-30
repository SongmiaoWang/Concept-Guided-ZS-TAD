import torch
import pickle
import numpy as np
from PIL import Image
import torch.nn as nn
from ops.roi_pool import RoIPool
import torchvision.models as mdls
from scipy import ndimage
from utils import build_clip_model, fill_holes
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor


class Adapter(nn.Module):
    def __init__(self, feat_dim, hidd_dim, num_heads, num_layers, dropout_rate=0.1):
        super(Adapter, self).__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidd_dim = hidd_dim

        # Encoder layer consists of a multi-head self-attention mechanism followed by a feed-forward neural network
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidd_dim,
            dropout=dropout_rate
        )

        # Transformer encoder consists of multiple layers of the encoder_layer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def _get_actn_preds(self, transformed, class_feats):
        class_feats = class_feats
        class_feats = class_feats.t()  # (512, 200)
        class_feats = class_feats.unsqueeze(0)  # (1, 512, 200)
        class_feats = class_feats.repeat(transformed.size(0), 1, 1)  # (16, 512, 132)
        class_scores = torch.bmm(transformed, class_feats)
        
        return class_scores

    def forward(self, src, class_feats):
        # src shape: (batch_size, num_items, input_dim)
        # Pass the input through the transformer encoder
        transformed = self.transformer_encoder(src.permute(1, 0, 2))

        # Permute back to the shape (batch_size, num_items, feat_dim/out_dim)
        transformed = transformed.permute(1, 0, 2)
        
        actn_preds = self._get_actn_preds(transformed, class_feats)
        orig_actn_preds = self._get_actn_preds(src, class_feats)

        return transformed, actn_preds, orig_actn_preds

class ResNetTransformer(nn.Module):
    def __init__(self, output_features=512, num_layers=6, nhead=8, dim_feedforward=2048):
        super(ResNetTransformer, self).__init__()
        self.resnet = mdls.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.fc = nn.Linear(512, output_features)

        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=output_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,  # Important for batch processing
        )
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        # x expected to be of shape (batch, sequence_length, channels, height, width)
        batch_size, sequence_length, C, H, W = x.size()
        
        x = x.view(batch_size * sequence_length, C, H, W)
        
        x = self.resnet(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        x = x.view(batch_size, sequence_length, -1)
        
        x = self.transformer_encoder(x)
        
        x = x.permute(0, 2, 1)
        
        return x

class TwoStageOVDet(nn.Module):
    def __init__(self, cfg):
        super(TwoStageOVDet, self).__init__()
        self.cfg = cfg
        self.adapter = Adapter(**cfg.adapter_cfgs)
        self.backbone = ResNetTransformer()
        self.linear = nn.Sequential(
            nn.Linear(cfg.roi_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, cfg.prop_feat_dim)
        )
        self.roi_extractor = RoIPool(cfg.roi_size*2, cfg.roi_scale)
    
    def load_adapter(self):
        weights = torch.load(self.cfg.pret_adapter_pth)
        self.adapter.load_state_dict(weights)
        for param in self.adapter.parameters():
            param.requires_grad = False
        print('Load adpter from: {}'.format(self.cfg.pret_adapter_pth))
        
    def add_batch_ind(self, rois):
        # rois: tensor of shape (batch_size, video_size, 2). video_size is the number of proposals per video
        rois_np = rois.cpu().numpy()
        batch_size, video_size = rois.shape[:2]
        batch_ind = np.arange(batch_size).reshape([batch_size, 1, 1]).repeat(video_size, axis=1)
        rois_with_batch_ind = np.concatenate((batch_ind, rois_np), axis=-1).reshape([batch_size*video_size, -1])
        return torch.from_numpy(rois_with_batch_ind.astype('float32')).cuda()
        
    def forward(self, raw_vids, prop_sted_frms, prop_vifi_feats, actn_feats):
        base_ft = self.backbone(raw_vids)  # 10 512 32
        batch_size, num_prop = prop_sted_frms.shape[0], prop_sted_frms.shape[1]
        rois_with_batch_ind = self.add_batch_ind(prop_sted_frms)  # 10 15 2
        prop_feats = self.roi_extractor(base_ft.contiguous(), rois_with_batch_ind.contiguous())  # features of all proposals
        prop_feats = prop_feats.reshape(batch_size, num_prop, -1)
        prop_feats = self.linear(prop_feats)
        adpted_feats, adpt_actn_preds, orig_actn_preds = self.adapter(prop_feats, actn_feats)
        vifi_adpted_feats, vifi_adpt_actn_preds, vifi_orig_actn_preds = self.adapter(prop_vifi_feats, actn_feats)
        
        return prop_feats, prop_vifi_feats, adpted_feats, vifi_adpted_feats, orig_actn_preds, vifi_orig_actn_preds, adpt_actn_preds, vifi_adpt_actn_preds

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts):
        # x = prompts + self.positional_embedding.type(self.dtype)
        x = prompts.to(torch.float32)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), prompts.argmax(dim=-1)] @ self.text_projection

        return x

class TemporalTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, 128, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, vid_feats):
        vid_feats = vid_feats + self.positional_encoding
        vid_feats = vid_feats.transpose(0, 1)
        output = self.transformer_encoder(vid_feats)
        output = output.transpose(0, 1)
        return output

class AttentionProjectionModule(nn.Module):
    def __init__(self, featdim=512, num_words=2000, num_heads=8):
        super(AttentionProjectionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = featdim // num_heads
        self.num_words = num_words
        assert self.head_dim * num_heads == featdim, "embedding dimension must be divisible by num_heads"
        
        self.q_proj = nn.Linear(featdim, featdim)
        self.k_proj = nn.Linear(featdim, featdim)
        self.v_proj = nn.Linear(featdim, featdim)
        
        self.out_proj = nn.Linear(featdim, featdim)

    def forward(self, video_feats, word_feats):
        # video_feats: (B, T, featdim)
        # word_feats: (num_words, featdim)
        
        B, T, featdim = video_feats.shape
        
        word_feats = word_feats.unsqueeze(0).expand(B, -1, -1)  # (B, num_words, featdim)
        
        Q = self.q_proj(word_feats)  # (B, num_words, featdim)
        K = self.k_proj(video_feats)  # (B, T, featdim)
        V = self.v_proj(video_feats)  # (B, T, featdim)
        
        Q = Q.view(B, self.num_words, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, num_words, head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, num_words, T)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, num_heads, num_words, T)
        
        updated_word_feats = torch.matmul(attention_weights, V)  # (B, num_heads, num_words, head_dim)
        updated_word_feats = updated_word_feats.transpose(1, 2).contiguous()  # (B, num_words, num_heads, head_dim)
        updated_word_feats = updated_word_feats.view(B, self.num_words, self.num_heads * self.head_dim)  # (B, num_words, featdim)
        
        projection = torch.matmul(video_feats.view(B, T, featdim), updated_word_feats.transpose(1, 2))  # (B, T, num_words)

        return projection

class OneStageOVDet(nn.Module):
    def __init__(self, cfg):
        super(OneStageOVDet, self).__init__()
        self.cfg = cfg
        self.temp_conv = nn.Sequential(
            nn.Conv1d(cfg.num_cncpt_vec, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=3, padding=1),
        )
        # self.clip_model = build_clip_model(cfg.clp_statedict_pth).to('cuda').to(torch.float32)
        # print("Turning off gradients in both the image and the text encoder")
        # for name, param in self.clip_model.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)
        # self.txt_encoder = TextEncoder(self.clip_model).to(torch.float32)
        self.vid_temp_model = TemporalTransformer(cfg.temp_transfm_dim, cfg.temp_transfm_head, cfg.temp_transfm_layers)
        
        # self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clp_pth)
        # self.processor = CLIPProcessor.from_pretrained(cfg.clp_pth)
        # self.learnable_vecs = nn.Parameter(torch.zeros(cfg.num_cncpt_vec, 512))
        with open(cfg.base_cncpt_pth, 'rb') as file:
            self.actn_concepts = pickle.load(file).to('cuda')
        self.proj_model = AttentionProjectionModule(cfg.proj_dim, cfg.num_cncpt_vec, cfg.proj_head)
    
    def get_prompt(self,cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append("a video of action"+" "+c)
        return temp_prompt
    
    def forward(self, vid_feats, segs_feats, actn_feats, seg_steds, is_inference):
        # vid_feats = self.vid_temp_model(vid_feats)
        
        # learnable_cncpts = self.txt_encoder(self.learnable_vecs.unsqueeze(0)).squeeze(0)
        # learnable_cncpts = learnable_cncpts / learnable_cncpts.norm(dim=-1, keepdim=True)
        # actn_concepts = learnable_cncpts + self.actn_concepts
        
        # feats_proj = torch.matmul(vid_feats, self.actn_concepts.T)
        vid_feats_ts = self.vid_temp_model(vid_feats)
        vid_feats_proj = vid_feats + vid_feats_ts  # residual
        feats_proj = self.proj_model(vid_feats_proj, self.actn_concepts.detach())
        # feats_proj = vid_feats
        
        frames_cls = self.temp_conv(feats_proj.permute(0, 2, 1))
        frames_cls = torch.sigmoid(frames_cls)  # batchsize, 1, vidlen
        
        # weight sum
        weights = frames_cls.squeeze(1)
        
        if not is_inference:
            weighted_seg_feats = []
            for b in range(len(seg_steds)):
                batch_seg_steds = seg_steds[b]
                vid_segs = []
                for seg in batch_seg_steds:
                    stf, edf = seg[0], seg[1]
                    seg_weights = weights[b][stf:edf+1]
                    seg_weights = torch.softmax(seg_weights, dim=0).unsqueeze(-1)
                    weighted_seg_feat = vid_feats[b][stf:edf+1] * seg_weights
                    weighted_seg_feat = weighted_seg_feat.sum(dim=0)
                    vid_segs.append(weighted_seg_feat)
                vid_segs = torch.stack(vid_segs)
                weighted_seg_feats.append(vid_segs)
            
            actn_logits = []
            for feat in weighted_seg_feats:
                actn_logits.append(torch.matmul(feat, actn_feats.T.detach()))
        
            return frames_cls, actn_logits, feats_proj
        
        else:
            actn_logits_frames = torch.matmul(vid_feats, actn_feats.T)
            
            return vid_feats, frames_cls, actn_logits_frames