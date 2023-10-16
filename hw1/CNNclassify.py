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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

def trainCNN(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training using {device}")

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

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f"./runs/{type(model).__name__}/{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    model.to(device)
    # define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train the network
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)

    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc%':<10} | {'Test Loss':<10} | {'Test Acc%':<10}")
    print('-' * 60)  # print a separator line

    best_test_accuracy = 0.0
    Path.mkdir(Path('./model'), exist_ok=True)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        model.train()
        train_loss = 0.0
        correct_train_predictions = 0
        total_train_predictions = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # evaluate 
            train_loss += loss.item()

            _, predicted = outputs.max(1)
            total_train_predictions += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(trainloader)
        train_accuracy = (correct_train_predictions / total_train_predictions) * 100
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        model.eval()
        test_loss = 0.0
        correct_test_predictions = 0
        total_test_predictions = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()

                _, predicted = outputs.max(1)
                total_test_predictions += labels.size(0)
                correct_test_predictions += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(testloader)
        test_accuracy = (correct_test_predictions / total_test_predictions) * 100
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), f'./model/{type(model).__name__}.pth')

        if epoch % args.print_freq == 0:
            # print statistics
            print(f"{epoch+1:<6} | {avg_train_loss:<10.4f} | {train_accuracy:<10.2f} | {avg_test_loss:<10.4f} | {test_accuracy:<10.2f}")

    writer.close()
    print('Finished Training')

def testCNN(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing using {device}")

    # run inference on args.filename picture
    image = plt.imread(args.filename)
    mnist_image = False
    # get dimension of the image and decide if it is cifar or mnist
    if len(image.shape) == 3 and image.shape[0] == 32:
        print("Using CIFAR10 trained model")
        # load the trained model from model/CIFAR10_Classifier.pth
        model = CIFAR10_Classifier()
        model.load_state_dict(torch.load('./model/CIFAR10_Classifier.pth'))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))  # Normalize each channel to range [-1, 1]
        ])
    elif len(image.shape) == 2 and image.shape[0] == 28:
        print("Using MNSIT trained model")
        # load the trained model from model/MNIST_Classifier.pth
        model = MNIST_Classifier()
        model.load_state_dict(torch.load('./model/MNIST_Classifier.pth'))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
        ])
        mnist_image = True
    else:
        raise RuntimeError("Error: unrecognized image format")
    
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # convert the prediction from number to class name
    if mnist_image:
        classes = ('0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9')
    else:
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"prediction result: {classes[predicted.item()]}")
    print(f"predicted probability: {round(torch.nn.functional.softmax(output, dim=1)[0][predicted].item(),4)}")
        
    feature_maps = model.conv1(image)
    # Number of feature maps to visualize (in this case, 32)
    num_feature_maps = feature_maps.size(1)

    # Create a grid to display the feature maps
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(15, 8))

    for i in range(num_feature_maps):
        row = i // 8
        col = i % 8
        ax = axes[row, col]

        # Get a specific feature map
        feature_map = feature_maps[0, i].detach().cpu().numpy()

        ax.imshow(feature_map, cmap='gray') 
        ax.axis('off')

    plt.tight_layout()
    if mnist_image:
        save_path = f'pngs/CONV_rslt_mnist.png'
    else:
        save_path = f'pngs/CONV_rslt_cifar.png'
    plt.savefig(save_path)

def main():
    parser = argparse.ArgumentParser(description='CNN classification')

    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run.")

    train_parser = subparsers.add_parser("train", help="Train the model.")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training.")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training.")
    train_parser.add_argument("--mnist", action="store_true", help="Train on MNIST dataset.")
    train_parser.add_argument("--cifar", action="store_true", help="Train on CIFAR dataset.")
    train_parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    train_parser.add_argument("--print_freq", type=int, default=2, help="Print frequency")

    test_parser = subparsers.add_parser("test", help="Test the trained model.")
    test_parser.add_argument("filename", type=str, help="Filename for inference.")

    args = parser.parse_args()

    if args.command == "train":
        trainCNN(args)
    if args.command == "test":
        testCNN(args)

if __name__ == '__main__':
    main()