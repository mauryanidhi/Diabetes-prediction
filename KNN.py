# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:26:00 2020

@author: Nidhi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#Load the dataset
df = pd.read_csv('diabetes.csv')

#Print the first 5 rows of the dataframe.
df.head()
#Let's observe the shape of the dataframe.
df.shape
df.info()
df.describe()
#outlier
df.plot(kind='box',figsize=(20,10))
df=df[df['SkinThickness']<80]
df=df[df['Insulin']<600]
df.loc[df['Insulin']==0,'Insulin']=df['Insulin'].mean()
df.loc[df['SkinThickness']==0,'SkinThickness']=df['SkinThickness'].mean()
df.loc[df['BloodPressure']==0,'BloodPressure']=df['BloodPressure'].mean()
df.loc[df['BloodPressure']==0,'BloodPressure']=df['BloodPressure'].mean()
#normalization
df=df/df.max()
                    
"""As observed above we have 768 rows and 9 columns.
 The first 8 columns represent the features and the last column 
 represent the target/label."""
 #Let's create numpy arrays for features and target
X = df.drop('Outcome',axis=1).values
y = df['Outcome'].values
X[:5]
y[:5]

#importing train_test_split for random split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)
#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,20)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 
#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=7)
#Fit the model
knn.fit(X_train,y_train)
#Get accuracy. Note: In case of classification algorithms score method represents accuracy.
knn.score(X_test,y_test)
#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.metrics import recall_score
recall_score(y_test,y_pred)