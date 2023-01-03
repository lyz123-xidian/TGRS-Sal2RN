Sal2RN: A Spatial-Spectral Salient ReinforcementNetwork for Hyperspectral and LiDAR Data Fusion Classification, TGRS, 2022.
==
[Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), Yuzhe Liu, [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN),[Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html),Kailiang Han and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).
***
Code for the paper: [Sal2RN: A Spatial-Spectral Salient ReinforcementNetwork for Hyperspectral and LiDAR Data Fusion Classification](https://ieeexplore.ieee.org/document/9998520).

<div align=center><img src="/Image/frameworks.jpg" width="80%" height="80%"></div>
Fig. 1: Framework of the proposed Sal2RN. It consists of two parts: multi-source feature extraction network and multi-source fusion classification network. Feature extraction is divided into three branches, in which CIM and CSCM are the cross-layer interaction module and center spectrum correction module respectively. The Dense Block is a four-layer dense connection structure, which is applied for initial spatial feature extraction.

Training and Test Process
--
1) Please prepare the training and test data as operated in the paper. And the websites to access the datasets are also provided. The used OCBS band selection method is referred to [https://github.com/tanmlh](https://github.com/tanmlh)
2) Run "demo.py" to generate the meta-training data 
3) Run the 'CMFSL_UP_main.py' to reproduce the CMFSL results on [Pavia University](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene) data set.

We have successfully tested it on Ubuntu 16.04 with PyTorch 1.1.0. Below is the classification map with five shots of training samples from each class. 

<div align=center><p float="center">
<img src="/Image/false_color.jpg" height="300" width="200"/>
<img src="/Image/gt.jpg" height="300"width="280"/>
<img src="/Image/classification_map.jpg" height="300"width="200"/>
</p></div>
<div align=center>Fig. 2: The composite false-color image, groundtruth, and classification map of Pavia University dataset.</div>  

References
--
If you find this code helpful, please kindly cite:

[1] B. Xi, J. Li, Y. Li, R. Song, D. Hong and J. Chanussot, "Few-Shot Learning With Class-Covariance Metric for Hyperspectral Image Classification," in IEEE Transactions on Image Processing, vol. 31, pp. 5079-5092, 2022, doi: 10.1109/TIP.2022.3192712.

Citation Details
--
BibTeX entry:
```
@ARTICLE{Xi_2022TIP_CMFSL,
  author={Xi, Bobo and Li, Jiaojiao and Li, Yunsong and Song, Rui and Hong, Danfeng and Chanussot, Jocelyn},
  journal={IEEE Transactions on Image Processing}, 
  title={Few-Shot Learning With Class-Covariance Metric for Hyperspectral Image Classification}, 
  year={2022},
  volume={31},
  number={},
  pages={5079-5092},
  doi={10.1109/TIP.2022.3192712}}
```
 
Licensing
--
Copyright (C) 2022 Bobo Xi

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
