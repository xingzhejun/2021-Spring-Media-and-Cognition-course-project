"""
测试不同损失函数的效果
"""


from dataset import tiny_caltech35
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import base_model

from prefetcher import data_prefetcher


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

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=True)
    # 转移至cuda
    model = base_model(class_num=config.class_num).to(cuda0)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
    # creiteron = torch.nn.CrossEntropyLoss().to(cuda0)
    # creiteron = torch.nn.MSELoss().to(cuda0)
    creiteron = torch.nn.L1Loss().to(cuda0)

    # you may need train_numbers and train_losses to visualize something
    train_numbers, train_losses = train(config, train_loader, model, optimizer, scheduler, creiteron)

    # you can use validation dataset to adjust hyper-parameters
    val_accuracy = test(val_loader, model)
    test_accuracy = test(test_loader, model)
    print('===========================')
    print("val accuracy:{}%".format(val_accuracy * 100))
    print("test accuracy:{}%".format(test_accuracy * 100))


def train(config, data_loader, model, optimizer, scheduler, creiteron):
    model.train()
    train_losses = []
    train_numbers = []
    counter = 0

    
    for epoch in range(config.epochs):
        prefetcher = data_prefetcher(data_loader)
        data, label = prefetcher.next()
        batch_idx = 0
        # for batch_idx, (data, label) in enumerate(data_loader):
        while data is not None:
            data = data.to(cuda0)
            label = label.to(cuda0)
            
            batch_idx += 1
            output = model(data)

            one_hot = torch.nn.functional.one_hot(label, num_classes=35).to(torch.float32)
            # print(output.shape)
            # print(one_hot.shape)

            loss = creiteron(output, one_hot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += data.shape[0]
            accuracy = (label == output.argmax(dim=1)).sum() * 1.0 / output.shape[0]
            data, label = prefetcher.next()
            if batch_idx % 20 == 0:
                print('Train Epoch: {} / {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch, config.epochs, batch_idx * len(data), len(data_loader.dataset),
                                          100. * batch_idx / len(data_loader), loss.item(), accuracy.item()))
                train_losses.append(loss.item())
                train_numbers.append(counter)
        scheduler.step()
        torch.save(model.state_dict(), './model.pth')

        
    return train_numbers, train_losses


def test(data_loader, model):
    model.eval()
    correct = 0
    prefetcher = data_prefetcher(data_loader)
    with torch.no_grad():
        data, label = prefetcher.next()
        # for data, label in data_loader:
        while data is not None:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum()
            data, label = prefetcher.next()
    accuracy = correct * 1.0 / len(data_loader.dataset)
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[112, 112])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--class_num', type=int, default=35)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--milestones', type=int, nargs='+', default=[40, 50])

    config = parser.parse_args()
    main(config)
