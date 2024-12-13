import glob
import os
import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision
from PIL import Image
import random
import traceback

class Pokmans_Data(Dataset):
    def __init__(self, root_path, mode=None):
        super(Pokmans_Data, self)
        self.pokman_names = sorted(os.listdir(root_path))
        self.labels = {}
        for i, name in enumerate(self.pokman_names):
            self.labels[name] = i
        self.all_imgs = []
        for name in self.pokman_names:
            self.all_imgs.extend(glob.glob(os.path.join(root_path, name, '*')))
        random.shuffle(self.all_imgs)
        if mode == "train":
            self.imgs = self.all_imgs[:int(len(self.all_imgs) * 0.8)]
        if mode == "test":
            self.imgs = self.all_imgs[int(len(self.all_imgs) * 0.8):]
        # print(self.labels)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        # "E:\pokeman\bulbasaur\00000000.png"
        name = self.imgs[item].split("\\")[-2] # bulbasaur
        img = cv2.imread(self.imgs[item])
        tf_img = self.tansfromData(img)
        label = self.labels[name] # 0
        return label, tf_img

    def tansfromData(self, img):
        img = Image.fromarray(img)
        tf_img = torchvision.transforms.Compose([
            # Resize一定一定要写元祖形式，不然只有一个位置变成224，很TM坑
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = tf_img(img)
        return img




if __name__ == '__main__':
    # 每个精灵对应的标签
    # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
    root_path = r"E:\pokeman"
    mode = "train"
    pkd = Pokmans_Data(root_path, mode)
    pkd_datas = DataLoader(pkd, batch_size=32, shuffle=True)
    for label, data in pkd_datas:
        print(label, "\n", data.shape)

    # tensor([1, 1, 4, 0, 2, 1, 4, 4, 3, 0, 0, 0, 3, 3, 4, 1, 0, 1, 0, 3, 2, 3, 0, 3,3, 4, 3, 2, 1, 1, 1, 4])
    # torch.Size([32, 3, 224, 224])

    # for label, tf_img in pkd:
    #     print(label, tf_img.shape)
    # 0 torch.Size([3, 224, 224])
    # 3 torch.Size([3, 224, 224])
