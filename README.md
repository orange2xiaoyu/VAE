## VAE

### 各文件功能

test_network/vae.py: 针对不同尺度的图像，逆卷积网络的设计不同；

train_image.py: 以kitti数据集为例，训练了00,01,02三个序列的数据；

vector_image.py: 输入单张图像进行测试，输出encoder和全连接层测结果。

