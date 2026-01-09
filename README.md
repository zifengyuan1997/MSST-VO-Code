Source code of the deep learning-based Visual Odometry model MSST-VO

Zifeng Yuan, Yuyao Shen, Jun Wang, and Yongqing Wang. MSST-VO: Monocular Visual Odometry for Ground Vehicles Based on Multiscale Spatial-Temporal Feature Aggregation Network. Submitted to IEEE Transactions on Vehicular Technology. 

The project is exploited based on [DeepVO](https://github.com/ChiWeiHsiao/DeepVO-pytorch). The code can be fully executed by copying and replacing the files in our repository with those in [DeepVO](https://github.com/ChiWeiHsiao/DeepVO-pytorch).
The STDFF submodule is designed based on [DeformableConv2d](https://blog.csdn.net/linghu8812/article/details/120748357) in the [Torchvision](https://docs.pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html).  
The GA-ConvLSTM submodule is designed based on [ConvLSTM](https://github.com/ndrplz/ConvLSTM_pytorch). 
The CASTIF submodule is designed based on [Cross-attention](https://blog.csdn.net/qq_39506862/article/details/133868090?ops_request_misc=elastic_search_misc&request_id=08acae94c9d7efdf81c981d966c90984&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-133868090-null-null.142^v102^pc_search_result_base9&utm_term=Multihead%20Cross%20Attention&spm=1018.2226.3001.4187). 

The code is successfully tested in the PyTorch environment with the following packages: 
- PyTorch 2.6.0
- Torchvision 0.21.0
- Python 3.12.9
- Numpy 2.2.2
