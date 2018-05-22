from __future__ import print_function, division

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from data_loader import CreateDataLoader
from model import TernausNet16
from loss import LossBinary


plt.ion()   # interactive mode


file_path = '/media/hdd/data/nuclei/stage1_test/'
BATCH_SIZE = 8

data_loader = CreateDataLoader(file_path)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#test images = %d' % dataset_size)

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
        inp_th = (inp>0.5).astype(np.uint8)
        plt.imshow(inp_th, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def test(model, criterion):
    best_model_wts = torch.load('best_model.pth')
    model.load_state_dict(best_model_wts)

    model.eval()  # Set model to training mode
    running_loss = 0.0

    # Iterate over data.
    step = 0
    for i, data in enumerate(dataset):
        step += 1
        inputs = data[0]
        targets = data[1]

        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # statistics
        running_loss += loss.item()

    epoch_loss = running_loss / dataset_size

    print('Test Loss: {:.4f} '.format(epoch_loss))

    print()

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

            outputs = model(inputs)

            for j in range(inputs.size()[0]):

                ax = plt.subplot(num_images, 2, images_so_far+1)
                ax.axis('off')
                imshow(inputs.cpu().data[j], gray=False)
                ax = plt.subplot(num_images, 2, images_so_far+2)
                ax.axis('off')
                imshow(outputs.cpu().data[j], gray=True)

                images_so_far += 2

                if images_so_far == num_images*2:
                    model.train(mode=was_training)
                    return


        model.train(mode=was_training)

model_ft = TernausNet16()
model_ft = model_ft.to(device)


criterion = LossBinary(jaccard_weight=1)

model_ft = test(model_ft, criterion)


visualize_model(model_ft)
plt.ioff()
plt.show()
