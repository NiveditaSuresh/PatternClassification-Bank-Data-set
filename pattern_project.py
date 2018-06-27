import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import genfromtxt
from sklearn import discriminant_analysis
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import feature_extraction
from sklearn import datasets
import math

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.decomposition import PCA

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#READING RAW DATA
my_data=pd.read_csv('bank-additional.csv',delimiter=',')
X= my_data

#IDENTIFYING UNKNOWNS
for i in range (0,20):
    for j in range (0,4119):
        if(X.ix[j,i] =='unknown'):
            X.ix[j,i] = '0'


#CATEGORICAL STRINGS TO CATEGORICAL INTEGERS            
le = preprocessing.LabelEncoder()
for i in range(1,10):    
    Y = le.fit_transform(X.ix[:,i])
    X.ix[:,i] =Y

    
Y = le.fit_transform(X.ix[:,13])
X.ix[:,13] =Y
Y = le.fit_transform(X.ix[:,19])
X.ix[:,19] =Y

Y = X.as_matrix()
Y = np.array(Y)
label_data = Y[:,19]
Y1 = Y[:,0:19]

Y , Y_test, label_train, label_test = train_test_split(Y1, label_data, stratify= label_data, test_size=0.25)

n_class1 = 0
n_class2 = 0

#Classifying data based on label
for i in range(0,Y.shape[0]):
    if(label_train[i]==0):
        n_class1 = n_class1+1
    else:
        n_class2 =n_class2+1
         

print()

X1 = np.empty((0,19))
X2 = np.empty((0,19))
#Y, label_train = SMOTE().fit_sample(Y, label_train)

#Classifying data based on label
for i in range(0,Y.shape[0]):
    if(label_train[i]==0):
        X1 = np.append(X1,[Y[i,:]],axis=0)
    else:
        X2= np.append(X2,[Y[i,:]],axis=0)
         

print()


#FILLING MISSING DATA

#Training Data; Imuptation with most frequent in a class
imp = Imputer(missing_values = 0,strategy = 'most_frequent',axis = 0)
Y1 = imp.fit_transform(X1[:,1:7])
X1[:,1:7] =Y1
Y2 = imp.fit_transform(X2[:,1:7])
X2[:,1:7] = Y2


X = np.append(X1,X2,axis =0)



#Test Data; Imputation with most frequent in all classes
imp = Imputer(missing_values = 0,strategy = 'most_frequent',axis = 0)
Y1 = imp.fit_transform(Y_test[:,1:7])
Y_test[:,1:7] =Y1


#One Hot Encoding for Categorical data

ohe = preprocessing.OneHotEncoder(n_values='auto', categorical_features= [1,2,3,4,5,6,7,8,9,13],sparse = False,handle_unknown='error')
ohe.fit(X)
X_processed = ohe.transform(X)
Y_test_processed = ohe.transform(Y_test)


#Normalize

ss = StandardScaler()
ss.fit(X_processed[:,47:56])
Y = ss.transform(X_processed[:,47:56])
X_processed[:,47:56] = Y

Y = ss.transform(Y_test_processed[:,47:56])
Y_test_processed[:,47:56] = Y

feature_train = np.array(X_processed)
feature_test = np.array(Y_test_processed)

#Feature selection

#fda = SelectKBest(score_func=f_classif,k=50)
#fda.fit(X_processed,label_train)
#X_reduced= fda.transform(X_processed)
#Y_test_reduced = fda.transform(Y_test_processed)


#Feature reduction

#feat_redn = PCA(n_components = 30)
#feat_redn.fit(X_processed)
#X_reduced = feat_redn.transform(X_processed)
#Y_test_reduced = feat_redn.transform(Y_test_processed)

#feature_train = np.array(X_reduced)
#feature_test = np.array(Y_test_reduced)

# Classification

#SVM CLASSIFIER
print()
print()
print("SVM CLASSIFIER")
print()

c_range = np.logspace(-2,2,20,base=10)
g_range = np.logspace(-2,2,20)
mean_acc = np.zeros((20, 20))
stdv_acc = np.zeros((20, 20))
for j in range(0,20):
    for k in range(0,20):
        c = c_range[j]        
        g = g_range[k]        
        Kfold = StratifiedKFold(n_splits = 5,shuffle = True)
        i=0;
        mean = 0
        stdv = 0
        for train_index,valid_index in Kfold.split(feature_train,label_train):
            feature_train_cv,feature_valid_cv = feature_train[train_index],feature_train[valid_index]
            label_train_cv,label_valid_cv = label_train[train_index],label_train[valid_index]
            model = SVC(C=c,gamma = g)
            model.fit(feature_train_cv,label_train_cv)
            label_valid_pred_cv = model.predict(feature_valid_cv)
            acc_valid =  accuracy_score(label_valid_cv,label_valid_pred_cv)
            mean= mean+acc_valid
            stdv = stdv + math.pow(acc_valid,2)
            i=i+1
        mean = mean/i
        stdv = math.sqrt((stdv/i)-math.pow(mean,2))
        mean_acc[j,k] = mean
        stdv_acc[j,k] = stdv


maxacc=0
for j in range(0,20):
    for k in range(0,20):
        if mean_acc[j,k]>=maxacc :
            maxacc = mean_acc[j,k]
            c_optimal = c_range[j]
            g_optimal = g_range[k]

count=0
minstdv=100
for j in range(0,20):
    for k in range(0,20):
        if mean_acc[j,k]==maxacc :
            count=count+1
            if stdv_acc[j,k]<=minstdv:
                minstd = stdv_acc[j,k]
                c_optimal = c_range[j]
                g_optimal = g_range[k]
                
       

model = SVC(C=c_optimal,gamma=g_optimal,class_weight='balanced')
model.fit(feature_train,label_train)

label_train_pred = model.predict(feature_train)
acc_train =  accuracy_score(label_train,label_train_pred)
print("Training Accuracy using SVM Classifier: ",acc_train)
roc_train = roc_auc_score(label_train,label_train_pred)
print("Training AUC using SVM Classifier: ",roc_train)
f1_train = f1_score(label_train,label_train_pred)
print("Training F1 SCORE using SVM Classifier: ",f1_train)
print(classification_report(label_train,label_train_pred))
print()
print(confusion_matrix(label_train,label_train_pred))
print()

label_test_pred = model.predict(feature_test)
acc_test =  accuracy_score(label_test,label_test_pred)
print("Testing Accuracy using SVM CLassifier: ",acc_test)
roc_test = roc_auc_score(label_test,label_test_pred)
print("Testing AUC using SVM Classifier: ",roc_test)
f1_test = f1_score(label_test,label_test_pred)
print("Testing F1 SCORE using SVM Classifier: ",f1_test)
print(classification_report(label_test,label_test_pred))
print()
print(confusion_matrix(label_test,label_test_pred))
print()
print()
print("NAIVE BAYES CLASSIFIER")
print()

#NAIVE BAYES

p1 = n_class1/(n_class1+n_class2)
p2 = n_class2/(n_class1+n_class2)
p = np.array([p1,p2])
model = GaussianNB()
model.fit(feature_train,label_train)

label_train_pred = model.predict(feature_train)
acc_train =  accuracy_score(label_train,label_train_pred)
print("Training Accuracy using NAIVE BAYES Classifier: ",acc_train)
roc_train = roc_auc_score(label_train,label_train_pred)
print("Training AUC using NAIVE BAYES Classifier: ",roc_train)
f1_train = f1_score(label_train,label_train_pred)
print("Training F1 SCORE using NAIVE BAYES Classifier: ",f1_train)
print()
print(classification_report(label_train,label_train_pred))
print()
print(confusion_matrix(label_train,label_train_pred))
print()

label_test_pred = model.predict(feature_test)
acc_test =  accuracy_score(label_test,label_test_pred)
print("Testing Accuracy using NAIVE BAYES CLassifier: ",acc_test)
roc_test = roc_auc_score(label_test,label_test_pred)
print("Testing AUC using NAIVE BAYES Classifier: ",roc_test)
f1_test = f1_score(label_test,label_test_pred)
print("Testing F1 SCORE using NAIVE BAYES Classifier: ",f1_test)
print(classification_report(label_test,label_test_pred))
print()
print(confusion_matrix(label_test,label_test_pred))


print()
print()
print("PERCEPTRON CLASSIFIER")
print()

#PERCEPTRON

model = SGDClassifier(loss='perceptron', eta0=1,max_iter=1000, learning_rate='constant', penalty=None,class_weight= "balanced")
model.fit(feature_train,label_train)

label_train_pred = model.predict(feature_train)
acc_train =  accuracy_score(label_train,label_train_pred)
print("Training Accuracy using PERCEPTRON Classifier: ",acc_train)
roc_train = roc_auc_score(label_train,label_train_pred)
print("Training AUC using PERCEPTRON Classifier: ",roc_train)
f1_train = f1_score(label_train,label_train_pred)
print("Training F1 SCORE using PERCEPTRON Classifier: ",f1_train)
print()
print(classification_report(label_train,label_train_pred))
print()
print(confusion_matrix(label_train,label_train_pred))
print()

label_test_pred = model.predict(feature_test)
acc_test =  accuracy_score(label_test,label_test_pred)
print("Testing Accuracy using PERCEPTRON CLassifier: ",acc_test)
roc_test = roc_auc_score(label_test,label_test_pred)
print("Testing AUC using PERCEPTRON Classifier: ",roc_test)
f1_test = f1_score(label_test,label_test_pred)
print("Testing F1 SCORE using PERCEPTRON Classifier: ",f1_test)
print(classification_report(label_test,label_test_pred))
print()
print(confusion_matrix(label_test,label_test_pred))
print()
print()

print("RANDOM FOREST CLASSIFIER")
print()
#RANDOM FOREST CLASSIFIER

dep_range = np.logspace(1,4,10,base=1)

mean_acc = np.zeros((1, 4))
stdv_acc = np.zeros((1, 4))
for j in range(0,4):
    dep = dep_range[j]
    Kfold = StratifiedKFold(n_splits = 5,shuffle = True)
    i=0;
    mean = 0
    stdv = 0
    for train_index,valid_index in Kfold.split(feature_train,label_train):
        feature_train_cv,feature_valid_cv = feature_train[train_index],feature_train[valid_index]
        label_train_cv,label_valid_cv = label_train[train_index],label_train[valid_index]
        model = RandomForestClassifier(n_estimators=10,max_depth=dep,class_weight='balanced')
        model.fit(feature_train_cv,label_train_cv)
        label_valid_pred_cv = model.predict(feature_valid_cv)
        acc_valid =  f1_score(label_valid_cv,label_valid_pred_cv)
        mean= mean+acc_valid
        stdv = stdv + math.pow(acc_valid,2)
        i=i+1
        
    mean = mean/i
    stdv = math.sqrt((stdv/i)-math.pow(mean,2))
    mean_acc[0,j] = mean
    stdv_acc[0,j] = stdv
    


maxacc=0
for d in range(0,4):
    if mean_acc[0,d]>=maxacc :
        maxacc = mean_acc[0,d]
        dep_optimal = dep_range[d]

count=0
minstdv=100
for d in range(0,4):
    if mean_acc[0,d]==maxacc :
        count=count+1
        if stdv_acc[0,d]<=minstdv:
            minstd = stdv_acc[0,d]
            dep_optimal = dep_range[d]

model = RandomForestClassifier(n_estimators=10,max_depth=dep_optimal,class_weight='balanced')
model.fit(feature_train,label_train)

label_train_pred = model.predict(feature_train)
acc_train =  accuracy_score(label_train,label_train_pred)
print("Training Accuracy using Random Forest Classifier: ",acc_train)
roc_train = roc_auc_score(label_train,label_train_pred)
print("Training AUC using Random Forest Classifier: ",roc_train)
f1_train = f1_score(label_train,label_train_pred)
print("Training F1 SCORE using Random Forest Classifier: ",f1_train)
print()
print(classification_report(label_train,label_train_pred))
print()
print(confusion_matrix(label_train,label_train_pred))
print()

label_test_pred = model.predict(feature_test)
acc_test =  accuracy_score(label_test,label_test_pred)
print("Testing Accuracy using Random Forest CLassifier: ",acc_test)
roc_test = roc_auc_score(label_test,label_test_pred)
print("Testing AUC using Random Forest Classifier: ",roc_test)
f1_test = f1_score(label_test,label_test_pred)
print("Testing F1 SCORE using Random Forest Classifier: ",f1_test)
print(classification_report(label_test,label_test_pred))
print()
print(confusion_matrix(label_test,label_test_pred))
