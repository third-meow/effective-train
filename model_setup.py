import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as utils

BATCH_SIZE = 8

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)

        self.fc1 = nn.Linear(128*7*7, 280)
        self.fc2 = nn.Linear(280, 240)
        self.fc3 = nn.Linear(240, 140)
        self.fc4 = nn.Linear(140, 4)

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

def get_test_dataloader():
    np_xtest = pickle.load(open('./training_data/xtest.p', 'rb'))
    np_ytest = pickle.load(open('./training_data/ytest.p', 'rb'))

    tensor_xtest = torch.stack([torch.Tensor(i) for i in np_xtest])
    tensor_ytest = torch.stack([torch.Tensor(i) for i in np_ytest])

    test_dataset = utils.TensorDataset(tensor_xtest, tensor_ytest)
    return utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

def get_train_dataloader():
    np_xtrain = pickle.load(open('./training_data/xtrain.p', 'rb'))
    np_ytrain = pickle.load(open('./training_data/ytrain.p', 'rb'))

    tensor_xtrain = torch.stack([torch.Tensor(i) for i in np_xtrain])
    tensor_ytrain = torch.stack([torch.Tensor(i) for i in np_ytrain])

    train_dataset = utils.TensorDataset(tensor_xtrain, tensor_ytrain)
    return utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


def check_prediction(labels, network_output):
    num_correct = 0
    for l, o in zip(labels, network_output):
        if (o == max(o)).nonzero()[0].item() == l.item():
            num_correct += 1
    return num_correct



def main():

    net = Net()
    loss_func = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=1e-3, momentum=0.98)

    # get training data
    train_dataloader = get_train_dataloader()
    test_dataloader = get_test_dataloader()

    for q in range(10):
        total_loss = 0
        loss_avg_devider = 0
        for i, data in enumerate(train_dataloader):
            # split data into input and label
            x, y = data
            y = y.squeeze_()
            labels = torch.Tensor(y)

            # run through network
            out = net(x)

            # calculate loss and backprop
            loss = loss_func(out, labels.long())
            # record loss for total
            total_loss += loss.item()
            loss_avg_devider += 1
            loss.backward()
            opt.step()
        print(f'Average loss: {total_loss / loss_avg_devider}')

        correct_n = 0
        total_n = 0
        for i, data in enumerate(test_dataloader):
            # split data into input and label
            x, y = data
            y = y.squeeze_()
            labels = torch.Tensor(y)

            # run through network
            out = net(x)

            #tally correct predications
            correct_n += check_prediction(labels, out)
            total_n += BATCH_SIZE

        print(f'Accuracy: {correct_n / total_n}') 
        print('')




if __name__ == '__main__':
    main()
