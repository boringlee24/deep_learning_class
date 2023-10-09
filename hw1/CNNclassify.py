# a function to train CNN classifier on MNIST and CIFAR10
# and save the trained model to a file
# input: dataset name (MNIST or CIFAR10)
# output: trained model saved to a file
# usage example:
# from CNNclassify import trainCNN
# trainCNN('MNIST')
# trainCNN('CIFAR10')
# --------------------------------------------------------

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from CNN_helper import MNIST_Classifier, CIFAR10_Classifier
import argparse


def trainCNN(args):
    # load the dataset
    if args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
        ])

        trainset = torchvision.datasets.MNIST(root='/work/li.baol/data', train=True,
                                              download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='/work/li.baol/data', train=False,
                                             download=True, transform=transform)
        model = MNIST_Classifier()

    elif args.cifar:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize each channel to range [-1, 1]
        ])

        trainset = torchvision.datasets.CIFAR10(root='/work/li.baol/data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='/work/li.baol/data', train=False,
                                               download=True, transform=transform)
        model = CIFAR10_Classifier()
    else:
        print('Error: unrecognized dataset name')
        return


    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    # train the network
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print(inputs.shape)
            outputs = net(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print(running_loss)
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i, running_loss))
                running_loss = 0.0

    print('Finished Training')

    # save the trained model
    PATH = './' + dataset_name + '_net.pth'
    torch.save(net.state_dict(), PATH)

    # test the network on the test data
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print images
    # imshow(torchvision.utils.make_grid(images))
    # plt.show()
    print('GroundTruth: ', ' '.join('%5s' % labels[j].numpy() for j in range(4)))

    net = Classifier()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % predicted[j].numpy()
                                  for j in range(4)))
    
    # test the network on the whole test data  
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # print(images.shape)
            outputs = net(images)
            # print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted.shape)
            total += labels.size(0)
            # print(total)
            correct += (predicted == labels).sum().item()
            # print(correct)
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    

def main():
    parser = argparse.ArgumentParser(description='CNN classification')

    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run.")

    train_parser = subparsers.add_parser("train", help="Train the model.")
    train_parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training.")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training.")
    train_parser.add_argument("--mnist", action="store_true", help="Train on MNIST dataset.")
    train_parser.add_argument("--cifar", action="store_true", help="Train on CIFAR dataset.")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")

    args = parser.parse_args()

    if args.command == "train":
        trainCNN(args)

if __name__ == '__main__':
    main()