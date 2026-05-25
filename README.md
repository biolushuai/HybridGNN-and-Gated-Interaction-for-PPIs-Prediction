# GHADGI: Grpah hybrid Attention and Dynamic Gate Interaction for PPIs Prediction

This repository is the implementation of 

# Abstract
Accurate prediction of protein-protein interaction (PPI) sites is crucial for advancing molecular biology and facilitating drug discovery.  While traditional experimental methods are reliable, they are often labor-intensive and time-consuming. Although deep learning-based approaches have been developed for this task, there is still a need for enhanced predictive accuracy.  To address this challenge, we propose GHADGI, a novel model that employs a Graph Hybrid Attention module for multi-view structural encoding and a Dynamic Gated Interaction mechanism for adaptive feature fusion. Our model is built on a parallel architecture that combines a Graph Hybrid Attention module for multi-view structural feature extraction and a Transformer module for capturing long-range dependencies. A dynamic gated interaction mechanism is introduced to enable bidirectional communication between these two branches, effectively fusing 3D structural and 1D sequential information. Additionally, we utilize a denoising autoencoder to refine high-dimensional embeddings from protein language models, thereby improving the quality of input features. Experimental evaluations on benchmark datasets demonstrate that GHADGI achieves competitive performance compared to existing state-of-the-art methods.

## 1. Datasets and trainded models
The datasets used for training PMSFF and the trained models mentioned in our manuscrpit can be downloaded from https://pan.baidu.com/s/1R1d3ixNpBgTuCY0WvRMftQ （Password: PBRS）

## 2. Requirement
* Python = 3.9.10  
* Pytorch = 1.10.2  
* Scikit-learn = 1.0.2

## 3. Usage
develop_mspbrsp.py provides the code to reproduce the PMSFF (hyperparameters can be reset in configs.py).

get_preds_single.py shows an example how to generate binding residue predictions.

ProtT5 embeddings can be generated using bio_embeddings (https://github.com/sacdallago/bio_embeddings).

We provide an example in ./test_data and the ProtT5 embedding of testing protein is saved in a csv file.

## 4. Citation
If you are using PMSFF and find it helpful for PBRs prediction, we would appreciate if you could cite the following publication:

[1] Shuai Lu, Yuguang Li, Xiaofei Nan*, Shoutao Zhang*. Attention-based Convolutional Neural Networks for Protein-Protein Interaction Site Prediction[C]. The 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM2021), 2021, 141-144. DOI:10.1109/BIBM52615.2021.9669435.

[2] Yuguang Li, Shuai Lu*, Qiang Ma, Xiaofei Nan, Shoutao Zhang. Protein-Protein Interaction Site Prediction Based on Attention Mechanism and Convolutional Neural Networks[J]. Doi: 10.1109/TCBB.2023.3323493.

[3] Yuguang Li, Xiaofei Nan, Shoutao Zhang, Qinglei Zhou, Shuai Lu*, Zhen Tian*. PMSFF: Improved Protein Binding Residues Prediction Through Multi-scale Sequence-based Feature Fusion Strategy[J]. Biomolecules, 2024, 14(10): 1220.


## 5. References
[1] Min Zeng, Fuhao Zhang, Fang-Xiang Wu, Yaohang Li, Jianxin Wang, Min Li*. Protein-protein interaction site prediction through combining local and global features with deep neural networks[J]. Bioinformatics, 36(4), 2020, 1114–1120. DOI:10.1093/bioinformaticsz699.  

[2] Bas Stringer*, Hans de Ferrante, Sanne Abeln, Jaap Heringa, K. Anton Feenstra and Reza Haydarlou* (2022). PIPENN: Protein Interface Prediction from sequence with an Ensemble of Neural Nets[J]. Bioinformatics, 38(8), 2022, 2111–2118. DOI:10.1093/bioinformatics/btac071.

[3] Dallago C, Schütze K, Heinzinger M, Olenyi T, Littmann M, Lu AX, Yang KK, Min S, Yoon S, Morton JT, & Rost B (2021). Learned embeddings from deep learning to visualize and predict protein sets[J]. Current Protocols, 1, e113. DOI: 10.1002/cpz1.113

## 6. Contact
For questions and comments, feel free to contact: ieslu@zzu.edu.cn.
