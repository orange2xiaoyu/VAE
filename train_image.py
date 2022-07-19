from modulefinder import IMPORT_NAME
from pkgutil import ImpImporter

import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ConvVAE import ConvVAE
from Pokemon import Pokemon,Kitti
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv


kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
recon_loss = lambda recon_x, x: F.mse_loss(recon_x, x, reduction='sum')

epochs =35
batch_size = 32

best_loss = 1e9
best_epoch = 0

valid_losses = []
train_losses = []
train_kl_losses = []
train_recon_losses = []
valid_kl_losses = []
valid_recon_losses = []

transform = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.Resize([94,310]),
    transforms.ToTensor(),
])

data_path = "/home/jinyu/lijinyu/datasets/kitti/data_odometry/stereo/"
pokemon_vali = Kitti(data_path, train=True, transform=transform)
pokemon_train = Kitti(data_path, train=True, transform=transform)

# pokemon_train = Pokemon("/home/jinyu/lijinyu/code/autoencoder/figure/", train=True, transform=transform)
# pokemon_vali = Pokemon("/home/jinyu/lijinyu/code/autoencoder/figure/", train=True, transform=transform)


train_loader = DataLoader(pokemon_train, batch_size=batch_size, shuffle=True)
vali_loader = DataLoader(pokemon_vali, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvVAE()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in tqdm(range(epochs),desc="train epochs"):
    print(f"Epoch {epoch}")
    model.train()
    train_loss = 0.
    train_recon = 0.
    train_kl = 0.
    train_num = len(train_loader.dataset)

    for idx, x in enumerate(train_loader):
        
        batch = x.size(0)
        x = x.to(device)
        recon_x, mu, logvar = model(x)
        recon = recon_loss(recon_x, x)
        kl = kl_loss(mu, logvar)
        recon = recon.sqrt()
        recon = recon/5.
        kl *= 10

        loss = recon + kl
        loss = loss / batch
        train_loss += loss.item()
        train_recon += recon.item()
        train_kl += kl.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(f"Training loss is{loss: .3f} \t and Recon loss is {recon / batch: .3f} \t ,and KL is {kl / batch: .3f} in Step {idx}")

    train_recon_losses.append(train_recon / train_num)
    train_kl_losses.append(train_kl / train_num)
    train_losses.append(train_loss / train_num)
    

    valid_loss = 0.
    valid_recon = 0.
    valid_kl = 0.
    valid_num = len(vali_loader.dataset)
    model.eval()
    with torch.no_grad():
        for idx, x in enumerate(vali_loader):
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            recon = recon_loss(recon_x, x)
            kl = kl_loss(mu, logvar)
            recon = recon.sqrt()
            recon = recon/5.
            kl *= 10

            loss = recon + kl
            valid_loss += loss.item()
            valid_kl += kl.item()
            valid_recon += recon.item()

        valid_recon_losses.append(valid_recon/ valid_num)
        valid_kl_losses.append(valid_kl/ valid_num)
        valid_losses.append(valid_loss / valid_num)

        print(
            f"Valid loss is {valid_loss / valid_num: .3f} \t ,and Recon loss is {valid_recon / valid_num: .3f} \t ,and KL is {valid_kl / valid_num: .3f} in epoch {epoch}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

            torch.save(model.state_dict(), 'best_model_pokemon')
            print("Model saved")

filename = './model/train_loss_'+str(epoch)+'.csv'
with open(filename, 'w', encoding='utf-8') as f:
     csv_writer = csv.writer(f)
     title = ['train_recon', 'train_kl','train_loss']
     csv_writer.writerow(title)
     for i in range(len(train_losses)):
         actors_data = [train_recon_losses[i],train_kl_losses[i],train_losses[i]]
         csv_writer.writerow(actors_data)

filename = './model/vali_loss_'+str(epoch)+'.csv'
with open(filename, 'w', encoding='utf-8') as f:
     csv_writer = csv.writer(f)
     title = ['vali_recon', 'vali_kl','vali_loss']
     csv_writer.writerow(title)
     for i in range(len(valid_losses)):
         actors_data = [valid_recon_losses[i],valid_kl_losses[i],valid_losses[i]]
         csv_writer.writerow(actors_data)


# 训练结束查看整个过程中loss的变化
plt.plot(train_losses, label='Train')
plt.plot(valid_losses, label='Valid')
plt.legend()
plt.title('Learning Curve')

plt.savefig("./model/loss_fig_"+str(epoch)+".png")
# plt.show()
print("trian is ending,and loss fig has been saved.")
