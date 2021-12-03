#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.datasets import load_breast_cancer


# In[2]:


# Setup

breast = load_breast_cancer()

breast_data = breast.data

breast_input = pd.DataFrame(breast_data)

breast_labels = breast.target
labels = np.reshape(breast_labels,(569,1))

final_breast_data = np.concatenate([breast_data,labels],axis=1)


# In[3]:


breast_dataset = pd.DataFrame(final_breast_data)

features = breast.feature_names
features_labels = np.append(features,'label') 

breast_dataset.columns = features_labels

#breast_dataset['label'].replace(0, 'Benign',inplace=True) 
#breast_dataset['label'].replace(1, 'Malignant',inplace=True)

#breast_dataset.head()


# In[4]:


from sklearn.model_selection import train_test_split

np.random.seed(0)
train, test = train_test_split(breast_dataset, train_size=0.8, test_size=0.2, random_state=np.random)


# In[5]:


y_train = train.pop("label").values
y_test = test.pop("label").values


# In[6]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler() 

x_train = scale.fit_transform(train.values) 
x_test = scale.transform(test.values) 


# In[7]:


# Question 1

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=np.random) 
classifier.fit(x_train, y_train)
y_pred_1 = classifier.predict(x_test)


# In[8]:


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_1)) 
print("Precision:",metrics.precision_score(y_test, y_pred_1)) 
print("Recall:",metrics.recall_score(y_test, y_pred_1))


# In[ ]:





# In[9]:


# Question 2

from sklearn.decomposition import PCA

x_2 = breast_dataset.loc[:,features].values 
y_2 = breast_dataset.loc[:,['label']].values 

x_2 = scale.fit_transform(x_2)
#y_2 = scale.fit_transform(y_2)

K = 8

components = []
for i in range(K):
    temp = "PC"
    temp += str(i)
    components.append(temp)

pca = PCA(n_components=K) 
principalComponents = pca.fit_transform(x_2) 
principalDf = pd.DataFrame(data = principalComponents, columns = components) 

breast_dataset_2 = pd.concat([principalDf, breast_dataset[['label']]], axis = 1)


# In[10]:


train_2, test_2 = train_test_split(breast_dataset_2, train_size=0.8, test_size=0.2, random_state=np.random)

y_train_2 = train_2.pop("label").values
y_test_2 = test_2.pop("label").values

x_train_2 = scale.fit_transform(train_2.values) 
x_test_2 = scale.transform(test_2.values) 

classifier_2 = LogisticRegression(random_state=np.random) 
classifier_2.fit(x_train_2, y_train_2)
y_pred_2 = classifier_2.predict(x_test_2)

print("Accuracy:",metrics.accuracy_score(y_test_2, y_pred_2)) 
print("Precision:",metrics.precision_score(y_test_2, y_pred_2)) 
print("Recall:",metrics.recall_score(y_test_2, y_pred_2))


# In[11]:


# These values were obtained by testing various k values using the above sections
accuracy = [0.9210526315789473, 0.9649122807017544, 0.9473684210526315, 0.9824561403508771,
            0.9824561403508771, 0.9824561403508771, 0.9824561403508771, 0.9912280701754386,
            0.9824561403508771, 0.9473684210526315]
precision = [0.9146341463414634, 0.9444444444444444, 0.927536231884058, 0.9710144927536232,
             0.9714285714285714, 0.96875, 0.9868421052631579, 0.987012987012987, 0.9753086419753086,
             0.9210526315789473]
recall = [0.974025974025974,  1.0, 0.9846153846153847, 1.0, 1.0, 1.0, 0.9868421052631579, 1.0,  1.0,  1.0]

plt.plot(range(1, 11), accuracy, color='blue', label="Accuracy")
plt.plot(range(1, 11), precision, color='red', label="Precision")
plt.plot(range(1, 11), recall, color='yellow', label="Recall")
plt.grid()
plt.legend()
plt.xlabel('# of Features')
plt.title('Metrics over various K values')


# In[12]:


# Question 3

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1) 

x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x_2, y_2, train_size=0.8, test_size=0.2, random_state=np.random)

y_train_3 = y_train_3.reshape(len(y_train_3),)

lda.fit(x_train_3, y_train_3) 
y_pred_3 = lda.predict(x_test_3) 

print("Accuracy:",metrics.accuracy_score(y_test_3, y_pred_3)) 
print("Precision:",metrics.precision_score(y_test_3, y_pred_3)) 
print("Recall:",metrics.recall_score(y_test_3, y_pred_3))


# In[ ]:





# In[ ]:




