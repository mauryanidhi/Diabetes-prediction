# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:37:23 2020

@author: Nidhi
"""

import pandas as pd
df=pd.read_csv('diabetes.csv')
df.head()
df.isnull().sum()
import seaborn as sns

X=df.drop('Outcome',axis=1).values### independent features
y=df['Outcome'].values###dependent features
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
df=pd.read_csv('diabetes.csv')
df.head()
y_train
#### Libraries From Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)
df.shape
#### Creating Modelwith Pytorch

class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=20,hidden2=20,out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x
###
torch.manual_seed(20)
model=ANN_Model()
model.parameters
## BACKWORD PROGAGATION
loss_function=nn.CrossEntropyLoss()
otimizer=torch.optim.Adam(model.parameters(),lr=.04)
import time
start_time=time.time()
epochs=10000
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss)
    if i%10==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    otimizer.zero_grad()
    loss.backward()
    otimizer.step()
print(time.time()-start_time)
### plot the loss function
import matplotlib.pyplot as plt

plt.plot(range(epochs),final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
#### Prediction In X_test data
predictions=[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        predictions.append(y_pred.argmax().item())
        print(y_pred.argmax().item())
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predictions)
cm

from sklearn.metrics import accuracy_score

accuracy_score(y_test,predictions)
from sklearn.metrics import recall_score
recall_score(y_test,predictions)


plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

#### Save the model
torch.save(model,'diabetes.pt')
#### Save And Load the model
model=torch.load('diabetes.pt')
model.eval()
### Predcition of new data point
list(df.iloc[0,:-1])
#### New Data
lst1=[6.0, 130.0, 72.0, 40.0, 0.0, 25.6, 0.627, 45.0]
new_data=torch.tensor(lst1)
#### Predict new data using Pytorch
with torch.no_grad():
    print(model(new_data))
    print(model(new_data).argmax().item())