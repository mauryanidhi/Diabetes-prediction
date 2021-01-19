# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:38:35 2021

@author: Nidhi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("diabetes.csv")
data.shape
data.head()
data.describe()
#compare the minimum and maximum values with the average and plot boxplot
data1=data.drop('Outcome',axis=1)
data1.plot(kind='box', subplots=True, layout=(4,4), sharex=False,sharey=False ,figsize =(15,15))
plt.show()
#Analysis of variables
def bar_plot(variable):
    var =data[variable]
    varValue = var.value_counts()
    plt.figure(figsize=(15,7))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    
    plt.show()
    print("{}: \n {}".format(variable,varValue))
data.columns

category1 = ['Pregnancies','Age']
    
for c in category1:
    bar_plot(c)
    
#distribution of variables according to the target.
import seaborn as sns   
from matplotlib import pyplot
a4_dims = (18, 8)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.countplot(x='Age',hue='Outcome',data=data, linewidth=1,ax=ax)

a4_dims = (18, 8)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.countplot(x='Pregnancies',hue='Outcome',data=data, linewidth=1,ax=ax)

colors = {0:'#cd1076', 1:'#008080'}
fig, ax = plt.subplots()
grouped = data.groupby('Outcome')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter'
               ,x='Glucose', y='Age', label=key
               ,color=colors[key])
plt.show()

colors = {0:'#cd1076', 1:'#008080'}
fig, ax = plt.subplots()
grouped = data.groupby('Outcome')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter'
               ,x='BMI', y='Age', label=key
               ,color=colors[key])
plt.show()
#distribution of our target column
data['Outcome'].value_counts().plot(kind='pie',colors=['#2C4373', '#F2A74B'],autopct='%1.1f%%',figsize=(9,9))
plt.show
varValue = data.Outcome.value_counts()
print(varValue)

#balance it with Upsampling method
from sklearn.utils import resample
df_majority = data.loc[data.Outcome == 0].copy()
df_minority = data.loc[data.Outcome == 1].copy()
df_minority_upsampled = resample(df_minority,
                             replace=True,  # sample with replacement
                            n_samples=500,  # to match majority class
                            random_state=123) 
data = pd.concat([df_majority, df_minority_upsampled])

data['Outcome'].value_counts().plot(kind='pie',colors=['#F2A74B', '#cd919e'],autopct='%1.1f%%',figsize=(9,9))
plt.show
varValue = data.Outcome.value_counts()
print(varValue)

data.isnull().sum()

#feature selection

data["Outcome"] = data.Outcome
X = data.drop("Outcome",1)
y = data["Outcome"]
data.head()
plt.figure(figsize=(15,7))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor["Outcome"]) #absolute value
#High Correlations
relevant_features = cor_target[cor_target>=0.2]
relevant_features

newdata=data.drop(['BloodPressure', 'SkinThickness', 'Insulin','DiabetesPedigreeFunction'],axis=1)

newdata.head()

data=pd.DataFrame(newdata)
from sklearn.preprocessing import StandardScaler
X = data.iloc[:, 0:4]
Y = data.iloc[:, 4]
nd = StandardScaler()
nd.fit(X)
X =nd.transform(X)
print(Y)


from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import model_selection
                  
X = data.iloc[:, 0:4]
Y = data.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

#to plot a graph
accuracies ={} 
means={}
randoms={}

#XGBOOST Classifier
#Manual Tuning
from xgboost import XGBClassifier
accuracy = []
for n in range(5,11):
    xgb =XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.78,
                           colsample_bytree=1, max_depth=n)
    xgb.fit(X_train,y_train)
    prediction = xgb.predict(X_test)
    accuracy.append(accuracy_score(y_test, prediction))
print(accuracy)    
plt.plot(range(5,11), accuracy,color='#cd5555')
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.show()
#RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
xgb_params = {
    'learning_rate' : [0.08, 0.06, 0.04, 0.09],      
    'max_depth': range(2,30),
    'n_estimators': [100, 200, 300,500,1000]}
xgb =XGBClassifier()
xgb_randomcv_model=RandomizedSearchCV(estimator=xgb, param_distributions=xgb_params, n_iter=2, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X_train,y_train)
print(xgb_randomcv_model.best_params_)
print('xgb_randomcv_model accuracy = {}'.format(xgb_randomcv_model.best_score_))
random=xgb_randomcv_model.best_score_*100
randoms['XGBoost']=random

#GridSearchCV
from sklearn.model_selection import GridSearchCV
xgb_params = { 'learning_rate' : [0.08, 0.06, 0.04, 0.09],      
    'max_depth': range(1,40),
    'n_estimators': [100, 200, 300,500,1000]}
xgb =XGBClassifier()
xgb_gridcv_model = GridSearchCV(estimator=xgb, param_grid=xgb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X_train,y_train)
print(xgb_gridcv_model.best_params_)
print('rf gridcv model accuracy score = {}'.format(xgb_gridcv_model.best_score_))
acc=xgb_gridcv_model.best_score_ *100
accuracies[' XGBoost Gridsearch']=acc

#Cross Validation
kfold=model_selection.KFold(n_splits=5)
modelL=XGBClassifier(n_estimators=100, max_depth=11,learning_rate=0.09)
results=model_selection.cross_val_score(modelL,X,Y,cv=kfold)
print(results)
print(results.mean()*100)
mean=results.mean()*100
means['XGBoost']=mean

#Decision tree
#RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
dt_params = {'min_weight_fraction_leaf' : [0.0 , 0.2 , 0.4 , 0.6 ,0.8],
   'max_depth': range(1,40),
    'max_features': range(1,40),
    'min_samples_leaf': range(1,40),
    'max_leaf_nodes' : range(1,40)
    
    }
dt=DecisionTreeClassifier()
dt_randomcv_model=RandomizedSearchCV(estimator=dt, param_distributions=dt_params, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X_train,y_train)
print(dt_randomcv_model.best_params_)
print('rf_randomcv_model accuracy score = {}'.format(dt_randomcv_model.best_score_))
random=dt_randomcv_model.best_score_*100
randoms['Decision Tree']=random

#Cross Validation
kfold=model_selection.KFold(n_splits=5)
modelL=DecisionTreeClassifier(min_weight_fraction_leaf=0.0,max_features=3, min_samples_leaf=15,max_depth=7,max_leaf_nodes=28)
results=model_selection.cross_val_score(modelL,X,Y,cv=kfold)
print(results)
print(results.mean()*100)
mean=results.mean()*100
means['Decision Tree']=mean


#Random forest
#RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
rf_params = {
   'max_depth': range(1,40),
    'max_features': range(1,40),
    'min_samples_leaf': range(1,20),
    'min_samples_split': range(1,20),
    'n_estimators': [100, 200, 300,500,1000]}
rf=RandomForestClassifier()
rf_randomcv_model=RandomizedSearchCV(estimator=rf, param_distributions=rf_params, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X_train,y_train)
print(rf_randomcv_model.best_params_)
print('rf_randomcv_model accuracy score = {}'.format(rf_randomcv_model.best_score_))
random=rf_randomcv_model.best_score_*100
randoms['Random Forest']=random

#Cross Validation
kfold=model_selection.KFold(n_splits=5)
modelL=RandomForestClassifier(n_estimators=100,min_samples_split=14, min_samples_leaf=4,max_depth=7)
results=model_selection.cross_val_score(modelL,X,Y,cv=kfold)
print(results)
print(results.mean()*100)
mean=results.mean()*100
means['Random Forest']=mean
#knn
#RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn_params = {'n_neighbors' : range(1,10)
   }
knn=KNeighborsClassifier()
knn_randomcv_model=RandomizedSearchCV(estimator=knn, param_distributions=knn_params, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X_train,y_train)
print(knn_randomcv_model.best_params_)
print('rf_randomcv_model accuracy score = {}'.format(knn_randomcv_model.best_score_))
random=knn_randomcv_model.best_score_*100
randoms['KNN']=random

#GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn_params = {'n_neighbors' : range(1,10),
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto','ball_tree','kd_tree','brute'],
              'p' : [1,2]
   }
knn=KNeighborsClassifier()
knn_gridcv_model=GridSearchCV(estimator=knn, param_grid=knn_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X_train,y_train)
print(knn_gridcv_model.best_params_)
print('rf_randomcv_model accuracy score = {}'.format(knn_gridcv_model.best_score_)) 
acc=knn_gridcv_model.best_score_ *100
accuracies['KNN Gridsearch']=acc

#Cross Validation
kfold=model_selection.KFold(n_splits=5)
modelL=KNeighborsClassifier(n_neighbors= 1)
results=model_selection.cross_val_score(modelL,X,Y,cv=kfold)
print(results)
print(results.mean()*100)
mean=results.mean()*100
means['KNN']=mean
#Logistic
#GridSearchCV
from sklearn.linear_model import LogisticRegression
lr_params = {'penalty' : ['l1','l2', 'elasticnet','none'],
              'C' : range(1,7),
            'solver' :['newton-cg','lbfgs','liblinear','sag','saga'],
             'max_iter' : [100,200],
             'multi_class' : ['ovr','multinomial']
   }
lr=LogisticRegression()
lr_gridcv_model=GridSearchCV(estimator=lr, param_grid=lr_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X_train,y_train)
print(lr_gridcv_model.best_params_)
print('rf_gridcv_model accuracy score = {}'.format(lr_gridcv_model.best_score_)) 
random=lr_gridcv_model.best_score_*100
randoms['Logistic Regression']=random

#Cross Validation
kfold=model_selection.KFold(n_splits=5)
model=LogisticRegression(C=2,max_iter=100,multi_class='ovr',penalty='l2',solver='liblinear')
results=model_selection.cross_val_score(model,X,Y,cv=kfold)
print(results)
print(results.mean()*100)
mean=results.mean()*100
means['Logistic Regression']=mean

#svm
#Gridsearch CV
from sklearn.svm import SVC
svc_params= {'C' : [0.1,0.2,0.3,0.001,0.003],
             'kernel': ['linear','poly','rbf','sigmoid']}
svc=SVC()
svc_gridcv_model=GridSearchCV(estimator=svc, param_grid=svc_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X_train,y_train)
print(svc_gridcv_model.best_params_)
print('rf_gridcv_model accuracy score = {}'.format(svc_gridcv_model.best_score_)) 
acc=svc_gridcv_model.best_score_ *100
accuracies['SVC Gridsearch']=acc

#Cross Validation
kfold=model_selection.KFold(n_splits=5)
model=SVC(C=0.1,kernel='linear')
results=model_selection.cross_val_score(model,X,Y,cv=kfold)
print(results)
print(results.mean()*100)
mean=results.mean()*100
means['SVC']=mean

#Comparisons
#GridSearch CV
colors = ["#C06C84", "#5E1742", "#005D8E"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,101,5))
plt.ylabel("GridSearch Scores%")
plt.xlabel("\n\n Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()

#RandomSearch CV
colors = ["#00008b", "#00e5ee", "#cd1076", "#008080","#cd5555",'black']

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,101,5))
plt.ylabel("Random Search Scores %")
plt.xlabel("\n\n Algorithms")
sns.barplot(x=list(randoms.keys()), y=list(randoms.values()), palette=colors)
plt.show()

#Cross Validation
colors = ["#C06C84", "#5E1742", "#005D8E", "#00ADB5","#3E606F","#EFAB1F"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,101,5))
plt.ylabel("Cross Validation Scores %")
plt.xlabel("\n\n Algorithms")
sns.barplot(x=list(means.keys()), y=list(means.values()), palette=colors)
plt.show()
