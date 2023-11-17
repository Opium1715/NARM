<a name="Z8Hwj"></a>
# Notification
This is an implementation of this paper(**Neural Attentive Session-based Recommendation**) based on Tensorflow 2.X, which contains the extra functions as below:

- Log(output figure for loss and MRR@20, P@20)
- preformance enhanced evaluation（maybe unstable）
<a name="i1yiS"></a>
## Requirements
TensorFlow 2.X (version>=2.10 is prefer)<br />Python 3.9<br />CUDA11.6 and above is prefer<br />cudnn8.4.0 and above is prefer<br />**Caution:** For who wants to run in native-Windows, TensorFlow **2.10** was the **last** TensorFlow release that supported GPU on native-Windows.

<a name="aRMxw"></a>
## [Citation](https://github.com/CRIPAC-DIG/SR-GNN/tree/e21cfa431f74c25ae6e4ae9261deefe11d1cb488#citation)
```
Citation
@inproceedings{10.1145/3132847.3132926,
author = {Li, Jing and Ren, Pengjie and Chen, Zhumin and Ren, Zhaochun and Lian, Tao and Ma, Jun},
title = {Neural Attentive Session-Based Recommendation},
year = {2017},
isbn = {9781450349185},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3132847.3132926},
doi = {10.1145/3132847.3132926},
abstract = {Given e-commerce scenarios that user profiles are invisible, session-based recommendation is proposed to generate recommendation results from short sessions. Previous work only considers the user's sequential behavior in the current session, whereas the user's main purpose in the current session is not emphasized. In this paper, we propose a novel neural networks framework, i.e., Neural Attentive Recommendation Machine (NARM), to tackle this problem. Specifically, we explore a hybrid encoder with an attention mechanism to model the user's sequential behavior and capture the user's main purpose in the current session, which are combined as a unified session representation later. We then compute the recommendation scores for each candidate item with a bi-linear matching scheme based on this unified session representation. We train NARM by jointly learning the item and session representations as well as their matchings. We carried out extensive experiments on two benchmark datasets. Our experimental results show that NARM outperforms state-of-the-art baselines on both datasets. Furthermore, we also find that NARM achieves a significant improvement on long sessions, which demonstrates its advantages in modeling the user's sequential behavior and main purpose simultaneously.},
booktitle = {Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
pages = {1419–1428},
numpages = {10},
keywords = {session-based recommendation, attention mechanism, recurrent neural networks, sequential behavior},
location = {Singapore, Singapore},
series = {CIKM '17}
}
```
