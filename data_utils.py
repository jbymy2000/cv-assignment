import os
import numpy as np
import torch


class MyDataset():
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.data_dir = os.path.join(self.root_dir, 'raw_data')
        self.label_dir = os.path.join(self.root_dir, 'groundtruth')
        self.images = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_name = img_name.split('.')[0] + '.png'
        label_path = os.path.join(self.label_dir, label_name)
        label = Image.open(label_path)

        # 将标签图像转换为 PyTorch 张量
        label = transforms.ToTensor()(label)

        print(f"{img_name}, Image size:", image.size())
        print(f"{img_name}, Label size:", label.size())

        return image, label
