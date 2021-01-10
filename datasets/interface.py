import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.transforms import Resize


class HAM10000(Dataset):
    def __init__(self, input_path="archive/", resolution=(225, 225), read_tensor=True):

        if not read_tensor:
            resolution_change = Resize(resolution)

            # Training data +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            path1 = input_path + "HAM10000_images_part_1"
            path2 = input_path + "HAM10000_images_part_2"
            img_names1 = os.listdir(path1)
            img_names2 = os.listdir(path2)

            self.data = torch.empty(
                len(img_names1) + len(img_names2), 3, resolution[0], resolution[1])

            for idx, name in enumerate(img_names1):
                img_name = path1 + "/" + name
                image = torch.from_numpy(plt.imread(img_name)).permute(2, 0, 1)
                image = resolution_change(image)
                self.data[idx] = image

            for idx, name in enumerate(img_names2):
                img_name = path2 + "/" + name
                image = torch.from_numpy(plt.imread(img_name)).permute(2, 0, 1)
                image = resolution_change(image)
                self.data[idx + len(img_names1)] = image

            torch.save(self.data, input_path + 'ham10000.pt')

            # Label data ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            pathl = input_path + "HAM10000_segmentations_lesion_tschandl"
            label_names = os.listdir(pathl)
            img_names = [name[:-4] for name in img_names1 + img_names2]

            self.label = torch.empty(
                len(img_names1) + len(img_names2), 1, resolution[0], resolution[1])

            for idx, name in enumerate(label_names):
                img_name = pathl + "/" + name
                image = torch.from_numpy(
                    plt.imread(img_name)).unsqueeze(0)
                if len(image.shape) == 4:
                    image = image[:, :, :, 0]
                image = resolution_change(image)

                idx = img_names.index(name[:-17])
                self.label[idx] = torch.round(image)

            torch.save(self.label.long(), input_path + 'ham10000_labels.pt')

        else:
            self.data = torch.load(input_path + 'ham10000.pt')
            print('Loaded HAM10000 images from ' + input_path +
                  'ham10000.pt with shape', self.data.shape)

            self.label = torch.load(input_path + 'ham10000_labels.pt').long()
            print('Loaded HAM10000 masks from ' + input_path +
                  'ham10000.pt with shape', self.label.shape)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    HAM10000()
