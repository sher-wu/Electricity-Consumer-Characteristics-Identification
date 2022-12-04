import numpy as np
import csv
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sys
from tabulate import tabulate

t_max = 1000
Feature = 27
Hidden_feature = 16
Target_y = 6
Category_Total = [0, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 3, 2, 3, 3, 2]
Category = Category_Total[Target_y]


class Logger(object):

    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(fileN, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


class RetailerDataset(Dataset):
    def __init__(self, in_feature, target_y, file_id):
        super(RetailerDataset, self).__init__()
        x_data = []
        y_data = []
        self.len = 0
        input_file = open("D:/cer_electricity/Train_Set/Retailer" + str(file_id + 1) + "_PPPCA_S.csv", 'r')
        input_reader = csv.reader(input_file)
        for row in input_reader:
            y_val = int(row[in_feature + target_y])
            if y_val > 0:
                x_data += [list(map(np.float32, row[1: in_feature + 1]))]
                y_data += [y_val - 1]
                self.len += 1
        input_file.close()
        self.x = torch.from_numpy(np.array(x_data))
        self.y = torch.LongTensor(np.array(y_data))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


dataset = [RetailerDataset(Feature, Target_y, i) for i in range(6)]
train_loader = [DataLoader(dataset=dataset[i], batch_size=16, shuffle=True, num_workers=0) for i in range(5)]
test_loader = DataLoader(dataset=dataset[5], batch_size=16, shuffle=False, num_workers=0)


class RetailerModel(nn.Module):
    def __init__(self, in_feature, out_category):
        super(RetailerModel, self).__init__()
        self.linear1 = nn.Linear(in_feature, Hidden_feature)
        self.linear2 = nn.Linear(Hidden_feature, out_category)
        self.dropout = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
        # self.bn = nn.BatchNorm1d(in_feature, affine=False)
        # self.bn2 = nn.LayerNorm(Hidden_feature, elementwise_affine=True)

    def forward(self, x):
        # x = self.bn(x)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        # x = self.bn2(x)
        x = self.linear2(x)
        return x


model = [RetailerModel(Feature, Category) for i in range(6)]
criterion = [nn.CrossEntropyLoss(reduction='mean') for i in range(6)]
# criterion = [nn.CrossEntropyLoss() for i in range(6)]
optimizer = [torch.optim.Adam(model[i].parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                              amsgrad=False) for i in range(6)]
# optimizer = [torch.optim.AdamW(model[i].parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
#                                amsgrad=False) for i in range(6)]


def pretty_vars(self):
    table = [[n, p.mean(), p.grad.mean()] for n, p in self.named_parameters() if p.grad is not None]
    return tabulate(table, headers=['layer', 'weights', 'grad'], tablefmt='pretty')


def train(epoch):
    total_loss = [0 for k in range(7)]
    total_num = [0 for k in range(6)]
    w1 = [torch.zeros(Hidden_feature, Feature) for i in range(5)]
    b1 = [torch.zeros(Hidden_feature) for i in range(5)]
    w2 = [torch.zeros(Category, Hidden_feature) for i in range(5)]
    b2 = [torch.zeros(Category) for i in range(5)]

    for k in range(5):
        optimizer[k].zero_grad()

        model[k].linear1.weight.data = model[5].linear1.weight.data
        model[k].linear1.bias.data = model[5].linear1.bias.data
        model[k].linear2.weight.data = model[5].linear2.weight.data
        model[k].linear2.bias.data = model[5].linear2.bias.data

        for retailer_t in range(3):
            for i, data in enumerate(train_loader[k], 0):
                inputs, labels = data
                total_num[k] += labels.size(0)
                y_pred = model[k](inputs)
                loss = criterion[k](y_pred, labels)
                loss.backward()
                total_loss[k] += loss.item()

        total_num[5] += total_num[k]
        # total_loss[6] += total_loss[k]
        # total_loss[k] /= total_num[k]
        # total_loss[5] += total_loss[k]
        w1[k] = model[k].linear1.weight.grad.data
        b1[k] = model[k].linear1.bias.grad.data
        w2[k] = model[k].linear2.weight.grad.data
        b2[k] = model[k].linear2.bias.grad.data

    optimizer[5].zero_grad()
    for k in range(5):
        a = total_num[k] / total_num[5]
        # a = total_loss[k] / total_loss[5]
        # a = total_loss[k] * total_num[k] / total_loss[6]
        model[5].linear1.weight.grad.data += w1[k] * a
        model[5].linear1.bias.grad.data += b1[k] * a
        model[5].linear2.weight.grad.data += w2[k] * a
        model[5].linear2.bias.grad.data += b2[k] * a
    # print(pretty_vars(model[5]))
    optimizer[5].step()

    # if epoch % 10 == 9:
    #     print(total_loss[0] + total_loss[1] + total_loss[2] + total_loss[3] + total_loss[4])


def test(epoch, max_record):
    model[5].eval()

    correct = 0
    total = 0
    cm = [[0 for i in range(Category)] for j in range(Category)]  # confusion matrix
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            y_pred = model[5](inputs)
            _, predicted = torch.max(y_pred, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for k in range(labels.size(0)):
                cm[predicted[k]][labels[k]] += 1

    mcc = 0
    if Category == 2:
        mcc = (cm[0][0] * cm[1][1] - cm[0][1] * cm[1][0]) / (
                (cm[0][0] + cm[1][0]) * (cm[0][0] + cm[0][1]) * (cm[1][0] + cm[1][1]) * (cm[0][1] + cm[1][1]) + 1e-8) ** 0.5
    else:
        mcc += (cm[0][0] * (cm[1][1] + cm[2][2]) - (cm[0][1] + cm[0][2]) * (cm[1][0] + cm[2][0])) / (
                (cm[0][0] + cm[1][0] + cm[2][0]) * (cm[0][0] + cm[0][1] + cm[0][2]) *
                (cm[1][0] + cm[2][0] + cm[1][1] + cm[2][2]) * (cm[0][1] + cm[0][2] + cm[1][1] + cm[2][2]) + 1e-8) ** 0.5
        mcc += (cm[1][1] * (cm[0][0] + cm[2][2]) - (cm[1][0] + cm[1][2]) * (cm[0][1] + cm[2][1])) / (
                (cm[1][1] + cm[0][1] + cm[2][1]) * (cm[1][1] + cm[1][0] + cm[1][2]) *
                (cm[0][1] + cm[2][1] + cm[0][0] + cm[2][2]) * (cm[1][0] + cm[1][2] + cm[0][0] + cm[2][2]) + 1e-8) ** 0.5
        mcc += (cm[2][2] * (cm[0][0] + cm[1][1]) - (cm[2][0] + cm[2][1]) * (cm[0][2] + cm[1][2])) / (
                (cm[2][2] + cm[0][2] + cm[1][2]) * (cm[2][2] + cm[2][0] + cm[2][1]) *
                (cm[0][2] + cm[1][2] + cm[0][0] + cm[1][1]) * (cm[2][0] + cm[2][1] + cm[0][0] + cm[1][1]) + 1e-8) ** 0.5
        mcc = mcc / 3

    if correct / total > max_record[0]:
        max_record[0] = correct / total
    if mcc > max_record[1]:
        max_record[1] = mcc
    print("Epoch:{} Accuracy:{:.3f} % Mcc:{:.5f} Cor/Tot:{} / {}".format(epoch, 100 * (correct / total), mcc, correct,
                                                                         total))

    model[5].train()


def main():
    logger = Logger('D:/Graduate/Log/log.log')

    for i, data in enumerate(train_loader[0], 0):  # initialize model5 w.grad b.grad
        inputs, labels = data
        y_pred = model[5](inputs)
        loss = criterion[5](y_pred, labels)
        loss.backward()
        optimizer[5].zero_grad()
        break

    max_record = [0, 0]

    for epoch in range(t_max):
        train(epoch)
        # if epoch % 10 == 9:
        test(epoch, max_record)

    print("Max_Accuracy:{:.4f} % Max_Mcc:{:.4f}".format(max_record[0], max_record[1]))

    logger.reset()


if __name__ == '__main__':
    main()
