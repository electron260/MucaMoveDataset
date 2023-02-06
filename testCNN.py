#Create a pytorch custom dataset with samples in each file of the directory and the label is the name of the file 
#In each file, each sample is separated by ; and each sample is composed of several list separated by a line
#The first list is the label of the sample and the others are the features of the sample
#The features are separated by a line





import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

#MyDataset takes a bool argument to know if it is a train dataset or a test dataset

#Each sample must be a tensor 
#Each label must be a string



class MyDataset(Dataset):
    def __init__(self, root_dir, train):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.files.sort()
        self.samples = []
        self.labels = []
        sample = []
        framenb = 0 
        labelchange = {"slideleft": 0, "slideright" : 1, "slideup" : 2, "slidedown" : 3, "longtouch" : 4}
        for file in self.files:
            if file.endswith('.txt'):
            
                with open(os.path.join(root_dir, file), 'r') as f:
                    for line in f:
                        
              
                        if line != '\n' :
                            print("len sample : ",len(sample), "framenb : ", framenb)
                            if line != ';\n':

                                lineprocessed = line.split(',')
                                lineprocessed[0] = lineprocessed[0].replace('[', '')
                                lineprocessed[-1] = lineprocessed[-1].replace(']\n', '')
                                lineprocessed = [i.replace(' ', '') for i in lineprocessed]
                                lineprocessed = [float(i) for i in lineprocessed]
                                sample.append(lineprocessed)
                        
                               
                                framenb += 1
                            

              
                        #lineprocessed = [float(i) for i in lineprocessed]
                      
                            if line == ";\n"  or framenb == 5 :
                                if framenb == 5 : 
                                    self.samples.append(sample)
                             
                                    self.labels.append(labelchange[str(file)[:-4]])
                                sample = []
                                framenb = 0
                          
                
                        
                
        if train:
            self.samples = self.samples[:int(0.8*len(self.samples))]
        else:
            self.samples = self.samples[int(0.8*len(self.samples)):]



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        print("tensor size : ", torch.Tensor(sample).size())
        return torch.Tensor(sample).unsqueeze(0), label

#Create a pytorch Convolutional NN taking tensor of 5 frames of 121 features as input and output a tensor of 5 float (the probability of each class)
#The input is a tensor of 5 frames of 121 features

#Try another one because this one give me as output RuntimeError: Given input size: (6x1x117). Calculated output size: (6x0x58). Output size is too small 
#I think it is because the size of the input is not a multiple of the size of the kernel
#I tried to change the size of the kernel but it doesn't work

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x






#The output is a tensor of 5 float (the probability of each class)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.unsqueeze(1)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    train_dataset = MyDataset("/Users/hugo/ArcticProject/CNNMOVE/MucaMoveDataset/dataset ",True)
    test_dataset = MyDataset("/Users/hugo/ArcticProject/CNNMOVE/MucaMoveDataset/dataset ",False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
    for i,v in enumerate(train_loader):
        print(v[0].size())
  
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, 10):
         train(model, train_loader, optimizer, epoch)
         test(model, test_loader)


if __name__ == '__main__':
 #open a file to read
    # with open("/Users/hugo/ArcticProject/CNNMOVE/MucaMoveDataset/dataset /slideright.txt") as f:
    #     for line in f : 
    #         print(type(line))
    main()