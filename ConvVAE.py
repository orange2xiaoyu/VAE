from torch import nn
import torch

latent_dim = 64
inter_dim = 256
# latent_dim = 32
# inter_dim = 128
# mid_dim = (256, 2, 2)                    #原始数据64*64卷积之后的后三维为256,2,2
# mid_dim = (256, 23, 77)                    #kitti数据集1241*376卷积之后的后三维为256,23,78;1226*370卷积之后的后三维为256*23*77
mid_dim = (256, 6, 19)                    #kitti数据集310*94卷积之后的后三维为256,6,19;
mid_num = 1
for i in mid_dim:
    mid_num *= i        #mid_num=1024
    
class ConvVAE(nn.Module):
    def __init__(self, latent=latent_dim):
        super(ConvVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.2),
        )

        self.fc1 = nn.Linear(mid_num, 1024)
        self.fc2 = nn.Linear(1024, inter_dim)
        self.fc3 = nn.Linear(inter_dim, latent * 2)

        self.fcr3 =  nn.Linear(latent, inter_dim)
        self.fcr2 = nn.Linear(inter_dim, 1024)
        self.fcr1 = nn.Linear(1024, mid_num)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,(1,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(128,64,3,2,(1,0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(64,32,3,2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(32,32,3,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(32,16,3,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(.2),

            nn.ConvTranspose2d(16,3,4,2,(1,1)),
            nn.Sigmoid()
        )

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        batch = x.size(0)
        x = self.encoder(x)
        x = self.fc1(x.view(batch, -1))                          #batch=512,原始x=(512,256,2,2),展开后得到512*1024
        h = self.fc2(x)
        h = self.fc3(h)

        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterise(mu, logvar)

        decode = self.fcr3(z)
        decode = self.fcr2(decode)
        decode = self.fcr1(decode)
        recon_x = self.decoder(decode.view(batch, *mid_dim))
        return recon_x, mu, logvar
