import sys
import torch
import random
import warnings
import numpy as np
from utils import *
from dataset import *
import itertools,operator
import torch.optim as optim
from torch.backends import cudnn

from models import *
from config import *

def run(cfg):
    # Set random seeds
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    torch.cuda.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    
    # cudnn settings
    cudnn.benchmark = False          
    cudnn.deterministic = True
    warnings.filterwarnings("ignore")
    
    # show config parameters
    cfg.init_config()
    show_config(cfg)
    
    # datasets & loaders
    params = {
        'batch_size': cfg.train_batch_size,
        'shuffle': cfg.shuffle,
        'num_workers': cfg.num_workers,
    }
    if cfg.mode == 1:  # train & test
        params['collate_fn'] = vid_collate_fn_trte
        train_set, test_set = return_dataset(cfg)
        train_loader = data.DataLoader(train_set, **params)
        params['batch_size'] = cfg.test_batch_size
        validation_loader = data.DataLoader(test_set, **params)
    else:  # inference
        params['collate_fn'] = vid_collate_fn_infn
        eval_set = return_dataset(cfg)
        params['batch_size'] = cfg.eval_batch_size
        eval_loader = data.DataLoader(eval_set, **params)
        
    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    if cfg.mode == 1:  # train & test
        model = OneStageOVDet(cfg)
    else:  # eval
        model = OneStageOVDet(cfg)
        model.load_state_dict(torch.load(cfg.pret_detector_pth), strict=False)
        print('model loaded from :{}'.format(cfg.pret_detector_pth))
    model = model.to(device=device)
    model.train()
    model.apply(set_bn_eval)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,
                           weight_decay=cfg.weight_decay)
    
    if cfg.mode == 1:  # train & test
        train, test = train_net, test_net
    
    if cfg.mode == 2:
        infn_info = infn_net(eval_loader, eval_set, model, device, cfg)
        show_epoch_info('Test', cfg.log_path, infn_info, cfg)
        sys.exit()
    
    if cfg.test_before_train:
        test_info = test(validation_loader, test_set, model, device, 0, cfg)
        show_epoch_info('Test', cfg.log_path, test_info, cfg)
    
    # iterations
    best_result_acc = {'epoch': 0, 'acc': 0}
    best_result_maskiou = {'epoch': 0, 'maskiou@5': 0}
    best_overall = 0.
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):
        # if epoch in cfg.lr_plan:
        #     adjust_lr(optimizer, cfg.lr_plan[epoch])
        # One epoch of forward and backward
        train_info = train(train_loader, train_set, model, device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info, cfg)
        
        if cfg.mode != 0 and epoch % cfg.test_interval_epoch == 0 and not cfg.only_train:
            test_info = test(validation_loader, test_set, model, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info, cfg)
            
            # save model
            if cfg.save_log_file:
                overall = test_info['maskiou@5']
                if overall >= best_overall or epoch == cfg.max_epoch:
                    best_overall = overall
                    state_dict = model.state_dict()
                    keys_to_remove = [key for key in state_dict if "clip_model" in key]
                    for key in keys_to_remove:
                        del state_dict[key]
                    dict_pth = cfg.result_path + '/' + 'model_epoch{}.pth'.format(epoch)
                    torch.save(state_dict, dict_pth)
                    print('model saved at: {}'.format(dict_pth))
                

            if test_info['acc'] > best_result_acc['acc']:
                best_result_acc = test_info
            print_log(cfg.log_path, cfg.save_log_file, 
                    'Best acc: %.5f%% at epoch #%d.' % (
                        best_result_acc['acc'], best_result_acc['epoch']))

            if test_info['maskiou@5'] > best_result_maskiou['maskiou@5']:
                best_result_maskiou = test_info
            print_log(cfg.log_path, cfg.save_log_file, 
                    'Best maskiou@5: %.5f%% at epoch #%d.' % (
                        best_result_maskiou['maskiou@5'], best_result_maskiou['epoch']))


def train_net(data_loader, dataset, model, device, optimizer, epoch, cfg):
    epoch_timer = Timer()
    iou_meter_3 = AverageMeter()
    iou_meter_4 = AverageMeter()
    iou_meter_5 = AverageMeter()
    iou_meter_6 = AverageMeter()
    iou_meter_7 = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for batch_data in tq(data_loader):
        model.train()
        model.apply(set_bn_eval)

        # prepare batch data
        clip_vid_feats = batch_data['clip_vid_feats'].to(device)
        fg_mask = batch_data['fg_masks'].to(device)
        out_segs_feats = [feat.to(device) for feat in batch_data['out_segs_feats']]
        segs_actn_gts = [tensors_to_device(act, device) for act in batch_data['all_actns']]
        seg_steds = batch_data['seg_steds']
        # out_segs_feats, segs_actn_gts = check_seg_feats(out_segs_feats, segs_actn_gts)
        actn_feats = dataset.actn_feats.to(device)
        
        # forward
        frames_logits, segs_actn_logits, feats_proj = model(clip_vid_feats, out_segs_feats, actn_feats, seg_steds, False)
        S_matrix_pred, S_matrix_gt = sample_segments(seg_steds, segs_actn_gts, feats_proj, actn_feats)
        segment_cls_loss, acc = get_segment_cls_loss(segs_actn_logits, segs_actn_gts)
        fg_maskloss = F.binary_cross_entropy(frames_logits, fg_mask.unsqueeze(1))
        proj_matrix_loss = masked_mse_loss(S_matrix_pred, S_matrix_gt)
        total_loss = fg_maskloss + .2 * segment_cls_loss + proj_matrix_loss
        
        # mask eval
        iou_3 = my_iou(frames_logits, fg_mask, .3)
        iou_4 = my_iou(frames_logits, fg_mask, .4)
        iou_5 = my_iou(frames_logits, fg_mask, .5)
        iou_6 = my_iou(frames_logits, fg_mask, .6)
        iou_7 = my_iou(frames_logits, fg_mask, .7)
        
        # temp_mask = torch.zeros_like(frames_logits, device='cpu')
        # temp_mask[frames_logits>.5] = 1
        # iou = mask_eval(temp_mask, fg_mask.cpu())
        
        iou_meter_3.update(iou_3, cfg.train_batch_size)
        iou_meter_4.update(iou_4, cfg.train_batch_size)
        iou_meter_5.update(iou_5, cfg.train_batch_size)
        iou_meter_6.update(iou_6, cfg.train_batch_size)
        iou_meter_7.update(iou_7, cfg.train_batch_size)
        acc_meter.update(acc, cfg.test_batch_size)
        
        # Optim
        optimizer.zero_grad()
        loss_meter.update(total_loss.item(), cfg.train_batch_size)
        total_loss.backward()
        optimizer.step()
        
    train_info = {
    'time': epoch_timer.timeit(),
    'maskiou@3': iou_meter_3.avg,
    'maskiou@4': iou_meter_4.avg,
    'maskiou@5': iou_meter_5.avg,
    'maskiou@6': iou_meter_6.avg,
    'maskiou@7': iou_meter_7.avg,
    'epoch': epoch,
    'acc': acc_meter.avg,
    'loss': loss_meter.avg
    }
    
    return train_info
        

def test_net(data_loader, dataset, model, device, epoch, cfg):
    epoch_timer = Timer()
    loss_meter = AverageMeter()
    iou_meter_3 = AverageMeter()
    iou_meter_4 = AverageMeter()
    iou_meter_5 = AverageMeter()
    iou_meter_6 = AverageMeter()
    iou_meter_7 = AverageMeter()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for batch_data in tq(data_loader):

            # prepare batch data
            clip_vid_feats = batch_data['clip_vid_feats'].to(device)
            fg_mask = batch_data['fg_masks'].to(device)
            out_segs_feats = [feat.to(device) for feat in batch_data['out_segs_feats']]
            segs_actn_gts = [tensors_to_device(act, device) for act in batch_data['all_actns']]
            seg_steds = batch_data['seg_steds']
            # out_segs_feats, segs_actn_gts = check_seg_feats(out_segs_feats, segs_actn_gts)
            actn_feats = dataset.actn_feats.to(device)
        
            # forward
            frames_logits, segs_actn_logits, feats_proj = model(clip_vid_feats, out_segs_feats, actn_feats, seg_steds, False)
            S_matrix_pred, S_matrix_gt = sample_segments(seg_steds, segs_actn_gts, feats_proj, actn_feats)
            segment_cls_loss, acc = get_segment_cls_loss(segs_actn_logits, segs_actn_gts)
            fg_maskloss = F.binary_cross_entropy(frames_logits, fg_mask.unsqueeze(1))
            proj_matrix_loss = masked_mse_loss(S_matrix_pred, S_matrix_gt)
            total_loss = fg_maskloss + .2 * segment_cls_loss + proj_matrix_loss
            
            # mask eval
            iou_3 = my_iou(frames_logits, fg_mask, .3)
            iou_4 = my_iou(frames_logits, fg_mask, .4)
            iou_5 = my_iou(frames_logits, fg_mask, .5)
            iou_6 = my_iou(frames_logits, fg_mask, .6)
            iou_7 = my_iou(frames_logits, fg_mask, .7)
            
            # temp_mask = torch.zeros_like(frames_logits, device='cpu')
            # temp_mask[frames_logits>.5] = 1
            # iou = mask_eval(temp_mask, fg_mask.cpu())
            
            iou_meter_3.update(iou_3, cfg.test_batch_size)
            iou_meter_4.update(iou_4, cfg.test_batch_size)
            iou_meter_5.update(iou_5, cfg.test_batch_size)
            iou_meter_6.update(iou_6, cfg.test_batch_size)
            iou_meter_7.update(iou_7, cfg.test_batch_size)
            acc_meter.update(acc, cfg.test_batch_size)
            loss_meter.update(total_loss.item(), cfg.test_batch_size)
            
        test_info = {
        'time': epoch_timer.timeit(),
        'maskiou@3': iou_meter_3.avg,
        'maskiou@4': iou_meter_4.avg,
        'maskiou@5': iou_meter_5.avg,
        'maskiou@6': iou_meter_6.avg,
        'maskiou@7': iou_meter_7.avg,
        'epoch': epoch,
        'acc': acc_meter.avg,
        'loss': loss_meter.avg
        }
        
        return test_info


def infn_net(data_loader, dataset, model, device, cfg):
    num_class = dataset.actn_feats.shape[0]
    key_list = dataset.actns_names
    # val_list = [i for i in range(len(key_list))]
    new_props = []
    iou_meter_3 = AverageMeter()
    iou_meter_4 = AverageMeter()
    iou_meter_5 = AverageMeter()
    iou_meter_6 = AverageMeter()
    iou_meter_7 = AverageMeter()
    # inference
    with torch.no_grad():
        for batch_data in tq(data_loader):
            model.train()
            model.apply(set_bn_eval)

            # prepare batch data           
            clip_vid_feats = batch_data['clip_vid_feats'].to(device)
            vid_name = batch_data['vid_name'][0]
            fg_mask = batch_data['fg_masks'].to(device)
            segs_feats = [feat.to(device) for feat in batch_data['segs_feats']]
            segs_actn_gts = [tensors_to_device(act, device) for act in batch_data['segs_lbs']]
            
            # vid_name, clip_vid_feats = batch_data
            # vid_name = vid_name[0]
            # clip_vid_feats = clip_vid_feats.to(device)
            actn_feats = dataset.actn_feats.to(device)
            
            # temp for exp
            gt_seg_actn = torch.matmul(segs_feats[0].mean(0), actn_feats.permute(1, 0))  # 50
            gt_seg_actn = key_list[torch.argmax(gt_seg_actn)]
            gt_actn = key_list[segs_actn_gts[0][0]]
            
            # forward
            vid_feats, frames_cls, actn_logits_frames = model(clip_vid_feats, None, actn_feats, None, True)  # single vid
            
            # mask eval
            iou_3 = my_iou(frames_cls, fg_mask, .3)
            iou_4 = my_iou(frames_cls, fg_mask, .4)
            iou_5 = my_iou(frames_cls, fg_mask, .5)
            iou_6 = my_iou(frames_cls, fg_mask, .6)
            iou_7 = my_iou(frames_cls, fg_mask, .7)
            
            iou_meter_3.update(iou_3, cfg.eval_batch_size)
            iou_meter_4.update(iou_4, cfg.eval_batch_size)
            iou_meter_5.update(iou_5, cfg.eval_batch_size)
            iou_meter_6.update(iou_6, cfg.eval_batch_size)
            iou_meter_7.update(iou_7, cfg.eval_batch_size)
            
            # frames_cls = fg_mask.unsqueeze(0)
            
            top_br_pred = actn_logits_frames.permute(0, 2, 1)
            
            soft_cas = torch.softmax(top_br_pred[0], dim=0)
            label_pred = torch.softmax(torch.mean(top_br_pred[0][:num_class,:],dim=1),axis=0).detach().cpu().numpy()
            vid_label_id = np.argmax(label_pred)
            soft_cas_np = soft_cas[:num_class].detach().cpu().numpy()
            # thres = cfg.class_snip_thresh
            max_score, score_idx  = torch.max(soft_cas[:num_class],0)
            soft_cas_np = soft_cas[:num_class].detach().cpu().numpy()
            score_map = {}
            # top_np = top_br_pred[0][:num_class].detach().cpu().numpy()  
            # top_np_max = np.mean(top_np,axis=1)
            max_score_np = max_score.detach().cpu().numpy()
            score_idx = score_idx.detach().cpu().numpy()

            for ids in range(len(score_idx)):
                score_map[max_score_np[ids]]= score_idx[ids]

            k = cfg.top_k_snip ## more fast inference
            max_idx = np.argpartition(max_score_np, -k)[-k:]

            ### indexes of top K scores ###
            top_k_idx = max_idx[np.argsort(max_score_np[max_idx])][::-1].tolist()
            top_k_idx = [top_k_idx[0]]
            seq = frames_cls.squeeze(0).squeeze(0).cpu().numpy()
            thres = cfg.mask_snip_thresh
            
            top_k_idx = max_idx[np.argsort(max_score_np[max_idx])][::-1].tolist()
            top_k_idx = [top_k_idx[0]]
            
            for locs in top_k_idx:
                for j in thres:
                    filtered_seq = seq > j
                    integer_map = map(int,filtered_seq)
                    filtered_seq_int = list(integer_map)
                    filtered_seq_int2 = ndimage.binary_fill_holes(filtered_seq_int).astype(int).tolist()
                    # vid_label = get_seg_cls(vid_feats, actn_feats, key_list, filtered_seq_int2)
                    
                    if 1 in filtered_seq_int:
                        #### getting start and end point of mask from mask branch ####
                        start_pt1 = filtered_seq_int2.index(1)
                        end_pt1 = len(filtered_seq_int2) - 1 - filtered_seq_int2[::-1].index(1) 
                        r = max((list(y) for (x,y) in itertools.groupby((enumerate(filtered_seq_int)),operator.itemgetter(1)) if x == 1), key=len)
                        start_pt = r[0][0]
                        end_pt = r[-1][0]
                        
                        if (end_pt - start_pt)/cfg.tscale > 0.02 :
                        #### get (start,end,cls_score,reg_score,label) for each top-k snip ####
                            score_ = max_score_np[locs]
                            cls_score = score_
                            # lbl_id = score_map[score_]
                            reg_score = np.amax(seq[start_pt+1:end_pt-1])
                            # label = key_list[val_list.index(lbl_id)]
                            # vid_label = key_list[val_list.index(vid_label_id)]  # ----
                            score_shift = np.amax(soft_cas_np[vid_label_id,start_pt:end_pt])
                            prop_start = start_pt1/cfg.tscale
                            prop_end = end_pt1/cfg.tscale
                            new_props.append([vid_name, prop_start, prop_end, score_shift*reg_score, score_shift*cls_score, gt_seg_actn])
                                    
        ### filter duplicate proposals --> Less Time for Post-Processing #####
        new_props = np.stack(new_props)
        b_set = set(map(tuple,new_props))  
        result = map(list,b_set) 

        ### save the proposals in a csv file ###
        col_name = ["video_name","xmin", "xmax", "clr_score", "reg_score","label"]
        new_df = pd.DataFrame(result, columns=col_name)
        new_df.to_csv(cfg.result_path + "/output.csv", index=False)
        
        infn_info = {
        'time': 0.,
        'maskiou@3': iou_meter_3.avg,
        'maskiou@4': iou_meter_4.avg,
        'maskiou@5': iou_meter_5.avg,
        'maskiou@6': iou_meter_6.avg,
        'maskiou@7': iou_meter_7.avg,
        'epoch': 0.,
        'acc': 0.,
        'loss': 0.
        }

        print("Inference finished")
        
        return infn_info
            
            

# def infn_net(data_loader, dataset, model, device, cfg):
#     num_cls = dataset.actn_feats.shape[0]
#     dataset_detections = [dict() for _ in range(num_cls)]
#     gt_list = []
#     # inference
#     with torch.no_grad():
#         for batch_data in tq(data_loader):
#             model.train()
#             model.apply(set_bn_eval)

#             # prepare batch data
#             out_gts, clip_vid_feats = batch_data
#             clip_vid_feats = clip_vid_feats.to(device)
#             actn_feats = dataset.actn_feats.to(device)
#             # forward
#             stn, edn, actn_logits = model(clip_vid_feats, None, actn_feats, True)  # single vid
            
#             _, topk_indices = torch.topk(actn_logits, 1, dim=1)
#             adpt_actn_preds = topk_indices.tolist()
            
#             cur_vid = out_gts[0][0][0]
#             for i, prop_cls_pred in enumerate(adpt_actn_preds):
#                 prop_cls_pred = prop_cls_pred[0]
#                 # prop_cls_pred = int(out_gts[0][1])
#                 st, ed = stn, edn
#                 score = 1.
#                 if cur_vid not in dataset_detections[prop_cls_pred]:
#                     dataset_detections[prop_cls_pred][cur_vid] = [[st, ed, score, 0, 0]]
#                 else:
#                     dataset_detections[prop_cls_pred][cur_vid].append([st, ed, score, 0, 0])
            
#             for gt in out_gts:
#                 gt_list.append([gt[0][0], gt[1].item(), gt[2].item(), gt[3].item()])
                
#         for c, clss in enumerate(dataset_detections):
#             for vid_name, prop_infos in clss.items():
#                 dataset_detections[c][vid_name] = np.vstack(prop_infos)
        
#         # # NMS
#         # print("Performing nms with thr {} ...".format(cfg.nms_threshold))
#         # for clss in range(num_cls):  # performing nms on each video
#         #     dataset_detections[clss] = {
#         #         k: temporal_nms(v, cfg.nms_threshold) for k,v in dataset_detections[clss].items()
#         #     }
#         # print("NMS Done.")

#         # all inference procedure completed, start evaluation
#         print("\nAll inference procedure completed, start evaluation...\n")

#         # gen .pc
#         if cfg.save_log_file:
#             pickle.dump(dataset_detections, open(cfg.result_path + '/dataset_detections.pc', 'wb'), pickle.HIGHEST_PROTOCOL)
#             pickle.dump(gt_list, open(cfg.result_path + '/gt_list.pc', 'wb'), pickle.HIGHEST_PROTOCOL)
