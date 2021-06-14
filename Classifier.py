# CSC3022F 2021 ML Assignment 3
# Part 2: Image Classification
# Author: WNGJIA001

# import packages
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image


# preprocessing transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
# loading data
mnist_trainset = datasets.MNIST(root='./data/', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testset = datasets.MNIST(root='./data/', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)
dataiter = iter(train_loader) # creating a iterator

# building neural network
input_size = 28*28 # number of pixels in a single image
hidden_sizes = [128, 64] # number of nodes in hidden layers
output_size = 10 # number of output nodes (0...9)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                    nn.ReLU(),
                                    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                    nn.ReLU(),
                                    nn.Linear(hidden_sizes[1], output_size),
                                    nn.LogSoftmax(dim=1))

    def forward(self, images):
        images = images.view(images.shape[0], -1) # flatten images with size [64,784]
        x = self.linear(images)
        return x

model = NeuralNetwork()
# loss function
criterion = nn.NLLLoss() # negative log-likelihood loss
# defining the optimiser with stochastic gradient descent and default parameters
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# training the nn
print("Pytorch Output...")
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # compute predicted output and loss
        pred = model(X) # output
        loss = loss_fn(pred, y)
        # backpropagation
        optimizer.zero_grad() # reset the gradients
        loss.backward() # backward pass
        optimizer.step() # optimizes and updates the weights
        # output message
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # output message
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 1 # number of iteration for training
for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train_loop(train_loader, model, criterion, optimizer)
    test_loop(test_loader, model, criterion)
print("Done!")

# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
# function to predict image
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

input_str = input("Please enter a filepath:\n")
while input_str != "exit":
    with Image.open(input_str) as img:
        print(predict_image(img))
    input_str = input("Please enter a filepath:\n")
