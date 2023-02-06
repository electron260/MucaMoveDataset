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
                        
                        lineprocessed = line.split(',')
                        lineprocessed[0] = lineprocessed[0].replace('[', '')
                        lineprocessed[-1] = lineprocessed[-1].replace(']\n', '')
                        lineprocessed = [i.replace(' ', '') for i in lineprocessed]
                        
                        if lineprocessed  != ['\n'] and lineprocessed != [';\n'] :
                            lineprocessed = [float(i) for i in lineprocessed]
                        # for i in lineprocessed :
                        #     print(i)

                    
                        #lineprocessed = [float(i) for i in lineprocessed]
                        if lineprocessed != ['\n'] and lineprocessed != [';\n'] :
                            if lineprocessed == "\n"  or framenb == 4 :
                                if framenb == 4 : 
                                    self.samples.append(sample)
                                    self.labels.append(labelchange[str(file)[:-4]])
                                sample = []
                                framenb = 0
                            else :
                            
                                sample = []
                                sample.append(lineprocessed)
                                framenb += 1
                            

                
                        
                
        if train:
            self.samples = self.samples[:int(0.8*len(self.samples))]
        else:
            self.samples = self.samples[int(0.8*len(self.samples)):]



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        #print("tensor  : ", torch.Tensor(sample[0]), "tensor size : ", torch.Tensor(sample[0]).unsqueeze(0).size())
        return torch.Tensor(sample[0]), label

#Create a pytorch CNN taking tensor of size 121 with a batch of 4 
#The output is a tensor of size 5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(1,0,2,3)
        print(x.size())
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.unsqueeze(1)
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
    train_dataset = MyDataset("/home/hugo/Bureau/ARTICPROJECT/CNNMove/MucaMoveDataset/dataset ",True)
    test_dataset = MyDataset("/home/hugo/Bureau/ARTICPROJECT/CNNMove/MucaMoveDataset/dataset ",False)
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
    main()
    # train_dataset = MyDataset("/home/hugo/Bureau/ARTICPROJECT/CNNMove/MucaMoveDataset/dataset ",True)
    # test_dataset = MyDataset("/home/hugo/Bureau/ARTICPROJECT/CNNMove/MucaMoveDataset/dataset ",False)
    # print( "train len  : ",len(train_dataset), " test len : ", len(test_dataset))
    # main()