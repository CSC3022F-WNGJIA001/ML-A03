# CSC3022F 2021 ML Assignment 3
# Part 2: Image Classification
# Author: WNGJIA001

# import packages
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def main():
    # preprocessing transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),])
    # loading data
    mnist_trainset = datasets.MNIST(root='./data/', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

    mnist_testset = datasets.MNIST(root='./data/', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

    dataiter = iter(train_loader) # creating a iterator
    images, labels = dataiter.next() # creating images for image and lables for image number (0 to 9)

    #print(images.shape)
    #print(labels.shape)

    #plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');
    #plt.show()
    



if __name__ == '__main__':
    main()
