#LIBRARY
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributions.dirichlet import Dirichlet
import torch.optim as optim
from torch.distributions.kl import kl_divergence as KL
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score
from sklearn.metrics import roc_curve,roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# LOADER
BATCH_SIZE_Train = 100
BATCH_SIZE_Test = 100


train_indomain = torchvision.datasets.MNIST('/gpfs01/berens/user/ywen/MNIST_part/', train=True, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.RandomErasing(p = 0.5, scale=(0.33, 0.33)),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))


train_OOD = torchvision.datasets.FashionMNIST('/gpfs01/berens/user/ywen/MNIST_part/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)) ]))


# P=0 test set

test_indomain = torchvision.datasets.MNIST('/gpfs01/berens/user/ywen/MNIST_part/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                               
                             ]))

test_OOD = torchvision.datasets.FashionMNIST('/gpfs01/berens/user/ywen/MNIST_part/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

# P=1 test set
test_indomain_er = torchvision.datasets.MNIST('/gpfs01/berens/user/ywen/MNIST_part/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.RandomErasing(p = 1, scale=(0.33, 0.33)),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))  
                             ]))

# CHANGE LABELS
train_OOD.targets += 10
test_OOD.targets += 10


# COMBINE AND ADD LOADER
train_dataset = torch.utils.data.ConcatDataset([train_indomain,train_OOD])
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE_Train, shuffle=True)

test1_dataset = torch.utils.data.ConcatDataset([test_indomain,test_OOD])
test2_dataset = torch.utils.data.ConcatDataset([test_indomain_er,test_OOD])

test_in_domain_loader = torch.utils.data.DataLoader(test1_dataset,batch_size=BATCH_SIZE_Test, shuffle=True)
test_erased_loader = torch.utils.data.DataLoader(test2_dataset,batch_size=BATCH_SIZE_Test, shuffle=True)

# DPN
class PriorNet_VGG(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
       
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256*3*3, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
  

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = self.maxpool(output)

        output = F.relu(self.conv2(output))
        output = self.maxpool(output)

        output = F.relu(self.conv3(output))
        output = self.maxpool(output)
        output = output.view(-1, 256*3*3)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
 
        return output
        # return F.softmax(output,dim=-1) # return probability

    def alphas(self, x): #alpha
        return torch.exp(self.forward(x))
  

    def uncertainty(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0
        entropy = -torch.sum(probs*torch.log(probs), dim=1)
        data_uncertainty = -torch.sum((alphas/alpha0)*(torch.digamma(alphas+1)-torch.digamma(alpha0+1)), dim=1)
        diff_entropy = torch.sum(
            torch.lgamma(alphas)-(alphas-1)*(torch.digamma(alphas)-torch.digamma(alpha0)),
            dim=1) - torch.lgamma(alpha0).T[0]
        return data_uncertainty,diff_entropy,entropy


# TRAINING
model= PriorNet_VGG()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.train()

def change_label(number):
  output_ls = []
  for i in number.numpy():
    label = np.ones(10)

    if int(i) > 9:  #  OOD
      label *= 1
      
    else:
      label[int(i)]= 100

    output = list(label)
    output_ls.append(output)
  return torch.Tensor(output_ls)

train_loss = []
num = []
for epoch in range(10):
 accuracy = 0
 for batch, (data, label) in enumerate(train_loader):
    optimizer.zero_grad()
          
    # predict
    alphas = model.alphas(data)
    
    # groundtruth
    target_alphas = change_label(label)

 
    loss = torch.mean(KL(Dirichlet(alphas), Dirichlet(target_alphas))) #RKL

    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())

 print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))


# TEST
print("Test set p=0")

model.eval()
accuracy = 0
accuracy_OOD = 0
label_test = []
differential =[]
datauncertainty =[]
entropy = []
MI = []
for batch, (data, label) in enumerate(test_in_domain_loader):
          
  # predict
  output = model.alphas(data)
  pred = torch.max(output,1).indices

  # ACCURACY 
  acc = 0
  for idx,lbl in enumerate(label):
    if lbl >= 10:
      label_test.append(1)
      
    else:
      label_test.append(0)
      if lbl==pred[idx]:
        acc+=1

  dc, dff, ent = model.uncertainty(data)
  ent_score = ent.detach().numpy()
  data_score = dc.detach().numpy()
  dff_score = dff.detach().numpy()
  MI_score = ent_score - data_score

  entropy.append(list(ent_score))
  differential.append(list(dff_score))
  datauncertainty.append(list(data_score))
  MI.append(list(MI_score))

    
  accuracy += acc
  
 
print('test accuracy in domain:',accuracy/10000)


entropy = list(np.concatenate(entropy).flat)
datauncertainty = list(np.concatenate(datauncertainty ).flat)
differential = list(np.concatenate(differential).flat)
MI = list(np.concatenate(MI).flat)


print('Entropy AUROC:',roc_auc_score(label_test,entropy))
print('Entropy AUPR:', average_precision_score(label_test,entropy))
print('Expect data uncertainty AUROC:',roc_auc_score(label_test,datauncertainty ))
print('Expect data uncertainty AUPR:', average_precision_score(label_test,datauncertainty ))
print('Entropy of mu AUROC:',roc_auc_score(label_test,differential))
print('Entropy of mu AUPR:', average_precision_score(label_test,differential))
print('Entropy of MI AUROC:',roc_auc_score(label_test,MI))
print('Entropy of MI AUPR:', average_precision_score(label_test,MI))
print('mean Entropy :',np.mean(entropy))
print('std Entropy:', np.std(entropy))
print('mean Expect data uncertainty:',np.mean(datauncertainty ))
print('std Expect data uncertainty:', np.std(datauncertainty ))
print('mean Entropy of mu:',np.mean(differential))
print('std Entropy of mu:', np.std(differential))
print('mean Entropy of MI:',np.mean(MI))
print('std Entropy of MI:', np.std(MI))


# TEST 2
print("Test set p=1")

model.eval()
accuracy = 0
accuracy_OOD = 0
label_test = []
differential =[]
datauncertainty =[]
entropy = []

MI = []
for batch, (data, label) in enumerate(test_erased_loader):
          
  # predict
   
  output = model.alphas(data)
  pred = torch.max(output,1).indices

  # ACCURACY 
  acc = 0
  for idx,lbl in enumerate(label):
    if lbl < 10:
      label_test.append(0)
      if lbl == pred[idx] :
        acc +=1 
    else:
      label_test.append(1)

  dc, dff, ent = model.uncertainty(data)
  ent_score = ent.detach().numpy()
  data_score = dc.detach().numpy()
  dff_score = dff.detach().numpy()
  MI_score = ent_score - data_score

  entropy.append(list(ent_score))
  differential.append(list(dff_score))
  datauncertainty.append(list(data_score))
  MI.append(list(MI_score))
    
  accuracy += acc
  
 
print('test accuracy in domain:',accuracy/10000)


entropy = list(np.concatenate(entropy).flat)
datauncertainty = list(np.concatenate(datauncertainty ).flat)
differential = list(np.concatenate(differential).flat)
MI = list(np.concatenate(MI).flat)

print('Entropy AUROC:',roc_auc_score(label_test,entropy))
print('Entropy AUPR:', average_precision_score(label_test,entropy))
print('Expect data uncertainty AUROC:',roc_auc_score(label_test,datauncertainty ))
print('Expect data uncertainty AUPR:', average_precision_score(label_test,datauncertainty ))
print('Entropy of mu AUROC:',roc_auc_score(label_test,differential))
print('Entropy of mu AUPR:', average_precision_score(label_test,differential))
print('Entropy of MI AUROC:',roc_auc_score(label_test,MI))
print('Entropy of MI AUPR:', average_precision_score(label_test,MI))
print('mean Entropy :',np.mean(entropy))
print('std Entropy:', np.std(entropy))
print('mean Expect data uncertainty:',np.mean(datauncertainty ))
print('std Expect data uncertainty:', np.std(datauncertainty ))
print('mean Entropy of mu:',np.mean(differential))
print('std Entropy of mu:', np.std(differential))
print('mean Entropy of MI:',np.mean(MI))
print('std Entropy of MI:', np.std(MI))
