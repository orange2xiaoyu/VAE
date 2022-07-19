import random
import os
from torch.utils.data import DataLoader, Dataset

class Kitti(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        # kitti数据集
        image_list = []
        # kitti数据集有三个尺寸的图像数据，我们在这只使用1241*376(1226,370)
        # train_list = ["00","01","02","13","14","15","16","17","18","19","20","21"]
        train_list = ["00","01","02"]
        for i in train_list:
            train_path = self.root + i + "/image_0/"
            for x in os.listdir(train_path ):
                self.image_path = os.path.join(train_path, x)
                image_list.append(self.image_path)
        random.shuffle(image_list)

        if transform is not None:
            self.transform = transform

        if train:
            self.images = image_list[: int(.8 * len(image_list))]

        else:
            self.images = image_list[int(.8 * len(image_list)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item])

class Pokemon(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(Pokemon, self).__init__()
        self.root = root
        self.image_path=[os.path.join(root, x) for x in os.listdir(root)]
        
        random.shuffle(self.image_path)

        if transform is not None:
            self.transform = transform

        if train:
            self.images = self.image_path[: int(.8 * len(self.image_path))]

        else:
            self.images = self.image_path[int(.8 * len(self.image_path)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item])