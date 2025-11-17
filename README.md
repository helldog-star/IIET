# IIET: Efficient Numerical Transformer via Implicit Iterative Euler Method

Official PyTorch implementation of our EMNLP 2025 paper.

## Abstract
High-order numerical methods enhance Transformer performance in tasks like NLP and CV, but introduce a performance-efficiency trade-off due to increased computational overhead. Our analysis reveals that conventional efficiency techniques, such as distillation, can be detrimental to the performance of these models, exemplified by PCformer. To explore more optimizable ODE-based Transformer architectures, we propose the Iterative Implicit Euler Transformer (IIET), which simplifies highorder methods using an iterative implicit Euler approach. This simplification not only leads to superior performance but also facilitates model compression compared to PCformer. To enhance inference efficiency, we introduce Iteration Influence-Aware Distillation (IIAD). Through a flexible threshold, IIAD allows users to effectively balance the performanceefficiency trade-off. On lm-evaluation-harness, IIET boosts average accuracy by 2.65% over vanilla Transformers and 0.8% over PCformer. Its efficient variant, E-IIET, significantly cuts inference overhead by 55% while retaining 99.4% of the original task accuracy. Moreover, the most efficient IIET variant achieves an average performance gain exceeding 1.6% over vanilla Transformer with comparable speed.

## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{liu2025iiet,
  title={IIET: Efficient Numerical Transformer via Implicit Iterative Euler Method},
  author={Liu, Xinyu and Li, Bei and Liu, Jiahao and Ruan, Junhao and Jiao, Kechen and Tang, Hongyin and Wang, Jingang and Xiao, Tong and Zhu, Jingbo},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={8955--8969},
  year={2025}
}
