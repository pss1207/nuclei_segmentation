from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from data_loader import CreateDataLoader
from model import UNet16
from loss import LossBinary


#from base_dataset import Preproc, Rescale, RandomCrop, ToTensor, Normalization, Resize
from tqdm import tqdm
plt.ion()   # interactive mode


file_path = '/media/hdd/data/nuclei/stage1_train/'
BATCH_SIZE = 8

data_loader = CreateDataLoader(file_path)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None, gray=False):
    """Imshow for Tensor."""
    if gray == False:
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
    else:
        inp = inp.numpy().transpose((1, 2, 0)).squeeze()
        plt.imshow(inp, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 99999.9

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        scheduler.step()
        model.train()  # Set model to training mode

        running_loss = 0.0

        # Iterate over data.
        step = 0
        for i, data in enumerate(dataset):
            step += 1
            inputs = data[0]
            targets = data[1]

            inputs = inputs.to(device)
            targets = targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item()

            #if step%32==0:
            #    print('Epoch: {} {}/{} Running Loss: {:.4f}'.format(epoch, step*inputs.size(0), dataset_size, running_loss/(step*inputs.size(0))))

        epoch_loss = running_loss / dataset_size

        print('Train Epoch Loss: {:.4f} '.format(epoch_loss))

        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_model.pth')
            print ('Best Epoch: {}'.format(epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, data in enumerate(dataset):
            inputs = data[0]
            targets = data[1]
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            for j in range(inputs.size()[0]):

                ax = plt.subplot(num_images, 3, images_so_far+1)
                ax.axis('off')
                imshow(inputs.cpu().data[j], gray=False)
                ax = plt.subplot(num_images, 3, images_so_far+2)
                ax.axis('off')
                imshow(outputs.cpu().data[j], gray=True)
                ax = plt.subplot(num_images, 3, images_so_far+3)
                ax.axis('off')
                imshow(targets.cpu().data[j], gray=True)

                images_so_far += 3

                if images_so_far == num_images*3:
                    model.train(mode=was_training)
                    return


        model.train(mode=was_training)

model_ft = UNet16()
model_ft = model_ft.to(device)

for params in model_ft.parameters():
    params.requires_grad = True


criterion = LossBinary(jaccard_weight=1)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)


visualize_model(model_ft)
plt.ioff()
plt.show()
