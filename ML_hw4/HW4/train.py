import torch
import torch.nn as nn
from datasets.dataloader import make_train_dataloader
from models.model import ExampleCNN

import os
import copy
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch.nn.functional as F
import argparse

# training parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# data path and weight path
base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "data", "train")
weight_path = os.path.join(base_path, "weights", "weight.pth")
img_path = os.path.join(base_path, "result")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set epochs and learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--img_path', type=str, default=img_path)
    parser.add_argument('--weight_path', type=str, default=weight_path)
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='Loss function for training')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    class FocalLoss(nn.Module):
        '''
        Multi-class Focal loss implementation
        '''
        def __init__(self, gamma=2, weight=None):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.weight = weight

        def forward(self, input, target):
            """
            input: [N, C]
            target: [N, ]
            """
            logpt = F.log_softmax(input, dim=1)
            pt = torch.exp(logpt)
            logpt = (1-pt)**self.gamma * logpt
            loss = F.nll_loss(logpt, target, self.weight)
            return loss


    
    # 根據命令列參數設置損失函數
    if args.loss_function == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_function == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    elif args.loss_function == 'FocalLoss':
        criterion = FocalLoss()
    elif args.loss_function == 'L1Loss':
        criterion = nn.L1Loss()
    elif args.loss_function == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")

    epochs = args.epochs
    learning_rate = args.learning_rate
    img_path = args.img_path
    weight_path = args.weight_path
    batch_size = args.batch_size
    loss_function = args.loss_function



print(epochs)
print(learning_rate)
print(img_path)
print(weight_path)


# make dataloader for train data
train_loader, valid_loader = make_train_dataloader(train_data_path, batch_size)

# set cnn model
model = ExampleCNN()
model = model.to(device)

# set optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#criterion = nn.CrossEntropyLoss()

# train
train_loss_list = list()
valid_loss_list = list()
train_accuracy_list = list()
valid_accuracy_list = list()
best = 100
best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(epochs):
    print(f'\nEpoch: {epoch+1}/{epochs}')
    print('-' * len(f'Epoch: {epoch+1}/{epochs}'))
    train_loss, valid_loss = 0.0, 0.0
    train_correct, valid_correct = 0, 0
    train_accuracy, valid_accuracy = 0.0, 0.0

    model.train()
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)

        # 做 one-hot 編碼 ，以符合lossfunction的張量維度
        one_hot = torch.zeros((target.shape[0],2)).to(device)
        one_hot[target==0]=torch.Tensor([1,0]).to(device)
        one_hot[target==1]=torch.Tensor([0,1]).to(device)

        # forward + backward + optimize
        output  = model(data)
        _, preds = torch.max(output.data, 1)
        if(loss_function == "FocalLoss")
            loss = criterion(output, target)
        else:
            loss = criterion(output, one_hot)
        optimizer.zero_grad()   # zero the parameter gradients
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        train_correct += torch.sum(preds == target.data)
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_accuracy = float(train_correct) / len(train_loader.dataset)
    train_accuracy_list.append((train_accuracy))

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(valid_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)

            one_hot = torch.zeros((target.shape[0],2)).to(device)
            one_hot[target==0]=torch.Tensor([1,0]).to(device)
            one_hot[target==1]=torch.Tensor([0,1]).to(device)

            output = model(data)
            if(loss_function == "FocalLoss")
                loss = criterion(output, target)
            else:
                loss = criterion(output, one_hot)
            _, preds = torch.max(output.data, 1)

            valid_loss += loss.item() * data.size(0)
            valid_correct += torch.sum(preds == target.data)
        valid_loss /= len(valid_loader.dataset)
        valid_loss_list.append(valid_loss)
        valid_accuracy = float(valid_correct) / len(valid_loader.dataset)
        valid_accuracy_list.append((valid_accuracy))
    
    # print loss and accuracy in one epoch
    print(f'Training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}')
    print(f'Training accuracy: {train_accuracy:.4f}, validation accuracy: {valid_accuracy:.4f}')

    # record best weight so far
    if valid_loss < best :
        best = valid_loss
        best_model_wts = copy.deepcopy(model.state_dict())
# save the best weight
torch.save(best_model_wts, weight_path)

# plot the loss curve for training and validation
print("\nFinished Training")
pd.DataFrame({
    "train-loss": train_loss_list,
    "valid-loss": valid_loss_list
}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1,epoch+1)
plt.xlabel("Epoch"),plt.ylabel("Loss")
plt.savefig(os.path.join(base_path, img_path, "Loss_curve"))

# plot the accuracy curve for training and validation
pd.DataFrame({
    "train-accuracy": train_accuracy_list,
    "valid-accuracy": valid_accuracy_list
}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1,epoch+1)
plt.xlabel("Epoch"),plt.ylabel("Accuracy")
plt.savefig(os.path.join(base_path, img_path, "Training_accuracy"))
