import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 6, 3)

        self.fc1 = nn.Linear(6*7*7, 180)
        self.fc2 = nn.Linear(180, 140)
        self.fc3 = nn.Linear(140, 40)
        self.fc4 = nn.Linear(40, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

def get_train_dataloader():
    np_xtrain = pickle.load(open('./training_data/xtrain.p', 'rb'))
    np_ytrain = pickle.load(open('./training_data/ytrain.p', 'rb'))

    tensor_xtrain = torch.stack([torch.Tensor(i) for i in np_xtrain])
    tensor_ytrain = torch.stack([torch.Tensor(i) for i in np_ytrain])
    #tensor_ytrain = torch.stack(np_ytrain)
    train_dataset = utils.TensorDataset(tensor_xtrain, tensor_ytrain)
    return utils.DataLoader(train_dataset, batch_size=8, shuffle=True)


def main():

    net = Net()
    loss_func = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=1e-3, momentum=0.98)

    # get training data
    train_dataloader = get_train_dataloader()
    for q in range(10):
        for i, data in enumerate(train_dataloader):
            # split data into input and label
            x, y = data

            y = y.squeeze_()

            # run through network
            out = net(x)

            # calculate loss and backprop
            labels = torch.Tensor(y)
            loss = loss_func(out, labels.long())
            print(loss.item())
            loss.backward()
            opt.step()




if __name__ == '__main__':
    main()
