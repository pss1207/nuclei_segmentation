

import os
import sys
import warnings

import numpy as np

from tqdm import tqdm
from skimage.io import imread
import torch.utils.data as data

import torch
from PIL import Image
from torchvision import transforms
import PIL

input_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
]
)
target_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224),interpolation=PIL.Image.NEAREST),
    transforms.ToTensor(),]
)

def data_gen(file_path):
    # Set some parameters
    IMG_CHANNELS = 3

    dataset = []

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    # Get train and test IDs
    train_ids = next(os.walk(file_path))[1]
    # Get and resize train images and masks
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        item = {}
        path = file_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]


        if os.path.isdir(path + '/masks/') == True:
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(mask_, axis=-1)
                mask = np.maximum(mask, mask_)
        else:
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

        item['input'] = torch.from_numpy(img)
        item['target'] = torch.from_numpy(mask)

        dataset.append(item)

    print('Done!')

    return dataset

class Dataset(data.Dataset):
    def __init__(self, file_path):
        self.dataset = data_gen(file_path)

    def __getitem__(self, index):
        data = self.dataset[index]

        input = data['input'].numpy()
        target = data['target'].byte().numpy()


        input = input_transform(input)
        target = target_transform(target)

        return input, target

    def __len__(self):
        return len(self.dataset)

class CustomDatasetDataLoader():
    def __init__(self, file_path):
        self.dataset = CreateDataset(file_path)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

def CreateDataset(file_path):
    dataset = Dataset(file_path)

    print("dataset is created")
    return dataset

def CreateDataLoader(file_path):
    data_loader = CustomDatasetDataLoader(file_path)
    return data_loader





