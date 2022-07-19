# 搭建vae
"""
该文件包括当输入为(40,40)时,编码网络和解码网络的设置;网络输入为(376,1241)时，对应的编码器网络和解码器网络;网络输入为(370,1226)时，对应的网络设置以及网络输入为(94,310)时，网络的设置。
"""

import torch
import torch.nn as nn


#例1：原始网络
seq1 = nn.Sequential(
          nn.Conv2d(3,32,3,2,1),
          nn.Conv2d(32,64,3,2,1),
          nn.Conv2d(64,128,3,2,1),
          nn.Conv2d(128,256,4,2,1)
        )
# seq2 = nn.Sequential(
#           nn.Linear(1024,128),
#           nn.Linear(128,32*2),
#           nn.Linear(32*2,128),
#           nn.Linear(128,1024)
#         )
# seq3 = nn.Sequential(
#           nn.ConvTranspose2d(256,128,4,2),
#           nn.ConvTranspose2d(128,64,3,2),
#           nn.ConvTranspose2d(64,32,3,1),
#           nn.ConvTranspose2d(32,32,3,1),
#           nn.ConvTranspose2d(32,16,3,1),
#           nn.ConvTranspose2d(16,3,4,2)
#         )

# #对上述seq进行输入
# input = torch.randn(512,3,40,40)
# x = seq1(input)
# x = x.view(x.shape[0], -1)
# x = seq2(x)
# x = x.reshape(512, 256, 2, 2)
# x = seq3(x)
# print(x.shape)


# =================================
seq4 = nn.Sequential(
          nn.Conv2d(3,16,3,2,1),
          nn.Conv2d(16,32,3,2,1),
          nn.Conv2d(32,64,3,2,1),
          nn.Conv2d(64,128,4,2,1)
        )
seq5 = nn.Sequential(
          nn.Linear(256*6*19,256),
          nn.Linear(256,32*2),
          nn.Linear(32*2,256),
          nn.Linear(256,256*6*19)
        )
seq6 = nn.Sequential(
          nn.ConvTranspose2d(256,128,3,2,(1,1)),
          nn.ConvTranspose2d(128,64,3,2,(1,0)),
          nn.ConvTranspose2d(64,32,3,2),
          nn.ConvTranspose2d(32,32,3,1),
          nn.ConvTranspose2d(32,16,3,1),
          nn.ConvTranspose2d(16,3,4,2,(1,1))
        )
# 原始图像太大，先缩小为1/4
input = torch.randn(32,3,94,310)  # 笔记本电脑batchsize=512会因为内存不足崩溃，先设置为32进行调试
x = seq1(input)  # torch.Size([32, 128, 6, 19])
import pdb;pdb.set_trace()
x = x.view(x.shape[0], -1)  # torch.Size([32, 14592])
x = seq5(x)  # torch.Size([32, 14592])
x = x.reshape(32, 256, 6, 19)  # torch.Size([32, 128, 6, 19])
x = seq6(x)  # torch.Size([32, 3, 72, 176])
print(x.shape)

# =================================
# seq7 = nn.Sequential(
#           nn.Conv2d(3,32,3,2,1),
#           nn.Conv2d(32,64,3,2,1),
#           nn.Conv2d(64,128,3,2,1),
#           nn.Conv2d(128,256,4,2,1)
#         )

# seq8 = nn.Sequential(
#           nn.Linear(459264,256),
#           nn.Linear(256,32*2),
#           nn.Linear(32*2,256),
#           nn.Linear(256,459264)
#         )

# seq9 = nn.Sequential(
#           nn.ConvTranspose2d(256,128,3,2,(0,1)),
#           nn.ConvTranspose2d(128,64,3,2,(1,1)),
#           nn.ConvTranspose2d(64,32,3,2),
#           nn.ConvTranspose2d(32,32,3,1,(1,1)),
#           nn.ConvTranspose2d(32,16,3,1,(1,0)),
#           nn.ConvTranspose2d(16,3,3,2,(0,1))
#         )


# #原始图像大小，减小batch_size便于计算
# input = torch.randn(32,3,376,1241)  
# x = seq7(input)  # torch.Size([32, 256, 23, 78])
# x = x.view(x.shape[0], -1)  # torch.Size([32, 459264])
# x = seq8(x)  # torch.Size([32, 459264])
# x = x.reshape(32, 256, 23, 78)  # torch.Size([32, 256, 23, 78])
# x = seq9(x)
# m = nn.ZeroPad2d((0, 0, 1, 0))
# x = m(x)
# print(x.shape)

# =================================
# seq10 = nn.Sequential(
#           nn.Conv2d(3,32,3,2,1),
#           nn.Conv2d(32,64,3,2,1),
#           nn.Conv2d(64,128,3,2,1),
#           nn.Conv2d(128,256,4,2,1)
#         )

# seq11 = nn.Sequential(
#           nn.Linear(256*23*77,256),
#           nn.Linear(256,32*2),
#           nn.Linear(32*2,256),
#           nn.Linear(256,256*23*77)
#         )

# seq12 = nn.Sequential(
#           nn.ConvTranspose2d(256,128,3,2,(0,1)),
#           nn.ConvTranspose2d(128,64,3,2,(1,1)),
#           nn.ConvTranspose2d(64,32,3,2,(1,1)),
#           nn.ConvTranspose2d(32,32,3,1,(1,0)),
#           nn.ConvTranspose2d(32,16,3,1,(1,0)),
#           nn.ConvTranspose2d(16,3,4,2,(1,1))
#         )


# #原始图像大小，减小batch_size便于计算
# input = torch.randn(32,3,370,1226)  
# x = seq10(input)  # torch.Size([32, 256, 23, 77])

# x = x.view(x.shape[0], -1)  # torch.Size([32, 256*23*77])
# x = seq11(x)  # torch.Size([32, 256*23*77])
# # import pdb;pdb.set_trace()
# x = x.reshape(32, 256, 23, 77)  # torch.Size([32, 256, 23, 77])
# # x = seq12(x)

# # print(x.shape)
