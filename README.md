# Concept-Guided-ZS-TAD

Pre-trained Vision-Language (ViL) models have shown strong zero-shot capabilities in various video understanding tasks. However, when applied to Zero-Shot Temporal Action Detection (ZS-TAD), existing ZS-TAD methods often face challenges in generalizing to unseen action categories due to their reliance on visual features, leading to misalignment between visual and semantic spaces. In this paper, we propose a novel Concept-guided Semantic Projection framework to enhance the generalization ability of ZS-TAD models. By projecting video features into a unified action concept space, our approach focuses on the semantic structure of actions, rather than solely relying on visual details. To further improve feature consistency across action categories, we introduce a Mutual Contrastive Loss, ensuring semantic coherence and better feature discrimination. Extensive experiments on ActivityNet and THUMOS14 benchmarks demonstrate that our method outperforms state-of-the-art ZS-TAD models.
