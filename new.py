import glob
import os
import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import random

class PokemonDataset(Dataset):
    """
    自定义数据集类，用于加载口袋妖怪数据集。

    参数:
    - root_path: 数据集的根路径。
    - mode: 数据集的模式，可以是 'train' 或 'test'，用于区分训练集和测试集。
    """
    def __init__(self, root_path, mode=None):
        super(PokemonDataset, self).__init__()
        self.root_path = root_path
        self.mode = mode
        self.pokemon_names = sorted(os.listdir(root_path))
        self.labels = {name: idx for idx, name in enumerate(self.pokemon_names)}
        self.all_images = [img_path for name in self.pokemon_names for img_path in glob.glob(os.path.join(root_path, name, '*'))]
        random.shuffle(self.all_images)

        if mode == "train":
            self.images = self.all_images[:int(len(self.all_images) * 0.8)]
        elif mode == "test":
            self.images = self.all_images[int(len(self.all_images) * 0.8):]
        else:
            self.images = self.all_images

    def __len__(self):
        """
        返回数据集的大小。

        返回:
        - 数据集的样本数。
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        根据索引获取样本。

        参数:
        - index: 样本索引。

        返回:
        - 标签和对应的图片数据。
        """
        image_path = self.images[index]
        pokemon_name = os.path.basename(os.path.dirname(image_path))
        image = cv2.imread(image_path)
        transformed_image = self.transform(image)
        label = self.labels[pokemon_name]
        return label, transformed_image

    def transform(self, image):
        """
        对图片进行预处理。

        参数:
        - image: 原始图片数据。

        返回:
        - 预处理后的图片数据。
        """
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

if __name__ == '__main__':
    # 每个精灵对应的标签
    # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
    root_path = r"E:\pokeman"
    mode = "train"
    dataset = PokemonDataset(root_path, mode)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for labels, data in dataloader:
        print(labels, "\n", data.shape)
