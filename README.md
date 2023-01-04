Sal2RN: A Spatial-Spectral Salient ReinforcementNetwork for Hyperspectral and LiDAR Data Fusion Classification, TGRS, 2022.
==
[Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Yuzhe Liu](https://github.com/lyz123-xidian), [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), Kailiang Han and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).
***
Code for the paper: [Sal2RN: A Spatial-Spectral Salient ReinforcementNetwork for Hyperspectral and LiDAR Data Fusion Classification](https://ieeexplore.ieee.org/document/9998520).


<div align=center><img src="/Image/framework.png" width="80%" height="80%"></div>
Fig. 1: Framework of the proposed Sal2RN. It consists of two parts: multi-source feature extraction network and multi-source fusion classification network. Feature extraction is divided into three branches, in which CIM and CSCM are the cross-layer interaction module and center spectrum correction module respectively. The Dense Block is a four-layer dense connection structure, which is applied for initial spatial feature extraction.

Training and Test Process
--
1) Please prepare the training and test data as operated in the paper. The datasets are Houston2013, Trento, MUUFL Gulfport. The data is placed under the 'data' folder. The file format is tif.
2) Run "demo.py" to to reproduce the Sal2RN results on Trento data set.

We have successfully tested it on Ubuntu 18.04 with PyTorch 1.12.0.

References
--
If you find this code helpful, please kindly cite:

[1]J. Li, Y. Liu, R. Song, Y. Li, K. Han and Q. Du, "Sal2RN: A Spatial-Spectral Salient Reinforcement Network for Hyperspectral and LiDAR Data Fusion Classification," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2022.3231930.

Citation Details
--
BibTeX entry:
```
@ARTICLE{9998520,
  author={Li, Jiaojiao and Liu, Yuzhe and Song, Rui and Li, Yunsong and Han, Kailiang and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Sal2RN: A Spatial-Spectral Salient Reinforcement Network for Hyperspectral and LiDAR Data Fusion Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2022.3231930}}
```
 
Licensing
--
Copyright (C) 2022 Yuzhe Liu

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
