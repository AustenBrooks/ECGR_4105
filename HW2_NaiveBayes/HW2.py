#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


diabetes = pd.read_csv("diabetes.csv")


# In[3]:


from sklearn.model_selection import train_test_split

np.random.seed(0)
train, test = train_test_split(diabetes, train_size=0.8, test_size=0.2, random_state=np.random)


# In[4]:


y_train = train.pop("Outcome").values
y_test = test.pop("Outcome").values


# In[5]:


from sklearn.preprocessing import StandardScaler 
scale = StandardScaler() 

x_train = scale.fit_transform(train) 
x_test = scale.transform(test) 


# In[6]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0) 
classifier.fit(x_train, y_train) 


# In[7]:


y_pred_1 = classifier.predict(x_test)


# In[8]:


from sklearn.metrics import confusion_matrix 
cnf_matrix_1 = confusion_matrix(y_test, y_pred_1)


# In[9]:


from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_1)) 
print("Precision:",metrics.precision_score(y_test, y_pred_1)) 
print("Recall:",metrics.recall_score(y_test, y_pred_1))


# In[10]:


import seaborn as sns 
class_names=[0,1] # name  of classes 
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names)

# create heatmap 
sns.heatmap(pd.DataFrame(cnf_matrix_1), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[11]:


# Problem 2

from sklearn.naive_bayes import GaussianNB 
classifier_NB = GaussianNB() 
classifier_NB.fit(x_train, y_train) 


# In[12]:


y_pred_2 = classifier.predict(x_test)


# In[13]:


from sklearn.metrics import confusion_matrix 
cnf_matrix_2 = confusion_matrix(y_test, y_pred_2)


# In[14]:


from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_2)) 
print("Precision:",metrics.precision_score(y_test, y_pred_2)) 
print("Recall:",metrics.recall_score(y_test, y_pred_2))


# In[15]:


import seaborn as sns 
class_names=[0,1] # name  of classes 
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names)

# create heatmap 
sns.heatmap(pd.DataFrame(cnf_matrix_2), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label') 


# In[16]:


# Problem 3

diabetes_3 = pd.read_csv("diabetes.csv")
y = diabetes_3.pop("Outcome").values
x = diabetes_3.values


# In[18]:


from sklearn.model_selection import KFold
K = 10
kf = KFold(n_splits=K)

classifier_3 = LogisticRegression(random_state=0, max_iter=1000) 

x_test_3 = None
y_test_3 = None

for train_index, test_index in kf.split(x):
    x_train_3, x_test_3 = x[train_index], x[test_index]
    y_train_3, y_test_3 = y[train_index], y[test_index]
    
    # train the model on k-1 folds
    if test_index[-1] != len(y)-1:        
        classifier_3.fit(x_train_3, y_train_3)

y_pred_3 = classifier_3.predict(x_test_3)

print("Accuracy:",metrics.accuracy_score(y_test_3, y_pred_3)) 
print("Precision:",metrics.precision_score(y_test_3, y_pred_3)) 
print("Recall:",metrics.recall_score(y_test_3, y_pred_3))


# In[ ]:




