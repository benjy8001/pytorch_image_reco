from CIFAR10 import CIFAR10
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from Net import Net
import torch


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__':
    # Chargement des données CIFAR10
    data = CIFAR10()
    trainSet = data.loadTrainSet(size=50)
    testSet = data.loadTestSet(size=25)

    # Création du réseau de neurones
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Apprentissage
    for epoch in range(2):
        running_loss = 0.0
        for i, Data in enumerate(trainSet, 0):
            inputs, labels = Data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Fin de l\'Apprentissage')

    # Récupération des données
    dataiter = iter(testSet)
    images, labels = dataiter.next()

    # Affichage de la mosaïque
    plt.axis('off')
    imshow(torchvision.utils.make_grid(images, nrow=5))
    for row in range(5):
        for col in range(5):
            plt.text(2 + 34 * col, 8 + 34 * row, CIFAR10.classes[labels[row * 5 + col]], fontsize=15, color='yellow')

    # Calcul des prédictions et affichage des erreurs
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    for row in range(5):
        for col in range(5):
            if CIFAR10.classes[labels[row * 5 + col]] != CIFAR10.classes[predicted[row * 5 + col][0]]:
                plt.text(2 + 34 * col, 24 + 34 * row, CIFAR10.classes[predicted[row * 5 + col][0]], fontsize=15, color='red')

    plt.show()
