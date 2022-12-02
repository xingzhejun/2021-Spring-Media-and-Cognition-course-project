"""
用于进行t-sne可视化
"""

from dataset import tiny_caltech35
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import base_model


import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt




cuda0 = torch.device('cuda:0')


def main(config):
    transform_train = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = tiny_caltech35(transform=transform_train, used_data=['train'])
    # if you want to add the addition set and validation set to train
    # train_dataset = tiny_caltech35(transform=transform_train, used_data=['train', 'val', 'addition'])

    train_dataset = tiny_caltech35(transform=transform_train, used_data=['train', 'val', 'addition'])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=True)
    # 转移至cuda
    model = base_model(class_num=config.class_num).to(cuda0)
    model.load_state_dict(torch.load('./model.pth'))

    # you can use validation dataset to adjust hyper-parameters
    # val_accuracy = test(val_loader, model)
    test_accuracy = test(train_loader, model)
    print('===========================')
    # print("val accuracy:{}%".format(val_accuracy * 100))
    print("test accuracy:{}%".format(test_accuracy * 100))




def test(data_loader, model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(cuda0)
            label = label.to(cuda0)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum()
    accuracy = correct * 1.0 / len(data_loader.dataset)
    
    print(output.cpu().numpy().shape)
    output = output.cpu().numpy().reshape(output.shape[0], -1)
    label = label.cpu().numpy()


    tsne = TSNE()
    X_embedded = tsne.fit_transform(output[label<5])
    palette = sns.color_palette("bright", 5)

    # X_embedded = X_embedded
    label = label[label<5]
    print(X_embedded.shape)
    print(label[label<5])
    # print(palette)
    
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=label, legend='full', palette=palette)
    plt.show()
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[112, 112])
    parser.add_argument('--batch_size', type=int, default=2100)
    parser.add_argument('--class_num', type=int, default=35)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 50])

    config = parser.parse_args()
    main(config)
