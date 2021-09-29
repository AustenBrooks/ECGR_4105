#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# calculates the loss given an input, an output, and the theta values
# returns a scalar loss
def loss_function(xVal, yVal, theta):
    hVal = np.dot(xVal, theta)
    hVal = np.reshape(hVal, (len(hVal), 1))
    
    J = np.sum((yVal - hVal)**2)
    
    J /= (2*len(yVal))
    
    return J


# In[3]:


# calculates the theta values to give the lowest loss
# returns an array of the theta values and an array of the cost history
def gradient_descent(xVal, yVal, xValTest, yValTest, theta, epochs, alpha):
    costHist = np.zeros(epochs)
    costHistTest = np.zeros(epochs)
    
    for i in range(epochs):
        hVal = xVal.dot(theta)
        error = hVal - yVal
        theta = theta - (alpha / len(yVal)) * xVal.transpose().dot(error)
        
        costHist[i] = loss_function(xVal, yVal, theta)
        costHistTest[i] = loss_function(xValTest, yValTest, theta)
    
    return theta, costHist, costHistTest


# In[4]:


# loss function with param penalties
def loss_function_penal(xVal, yVal, theta, lamb):
    hVal = np.dot(xVal, theta)
    hVal = np.reshape(hVal, (len(hVal), 1))
    
    penalties = theta
    penalties[0] = 0
    
    J = np.sum((yVal - hVal)**2)
    
    J += lamb * np.sum(penalties**2)
    
    J /= (2*len(yVal))
    
    return J


# In[5]:


# gradient descent with the parameter penalties implemented
def gradient_descent_param_penalties(xVal, yVal, xValTest, yValTest, theta, epochs, alpha, lamb):
    costHist = np.zeros(epochs)
    costHistTest = np.zeros(epochs)
    
    for i in range(epochs):
        hVal = xVal.dot(theta)
        error = hVal - yVal
        theta0 = theta[0]
        theta = theta * (1 - lamb * alpha / len(yVal)) - (alpha / len(yVal)) * xVal.transpose().dot(error)
        theta[0] = theta0 - (alpha / len(yVal)) * xVal.transpose().dot(error)[0]
        
        costHist[i] = loss_function_penal(xVal, yVal, theta, lamb)
        costHistTest[i] = loss_function(xValTest, yValTest, theta)
    
    return theta, costHist, costHistTest


# In[6]:


housing = pd.read_csv("Housing.csv")


# In[7]:


varList=['mainroad','guestroom','basement','hotwaterheating','airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes':1,"no":0})

housing[varList] = housing[varList].apply(binary_map)


# In[8]:


from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size=0.7, test_size=0.3, random_state=np.random)


# In[9]:


num_vars = ['bedrooms', 'bathrooms', 'stories', 'parking', 'price'] #removed area to prevent overflow
num_vars_b = ['bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price'] #removed area to prevent overflow

df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]

df_NewtrainB = df_train[num_vars_b]
df_NewtestB = df_test[num_vars_b]

df_Normtrain = df_train[num_vars]
df_Normtest = df_test[num_vars]

df_NormtrainB = df_train[num_vars_b]
df_NormtestB = df_test[num_vars_b]


# In[10]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#scaler = StandardScaler()
scaler = MinMaxScaler()
df_Normtrain[num_vars] = scaler.fit_transform(df_Normtrain[num_vars])
df_Normtest[num_vars] = scaler.fit_transform(df_Normtest[num_vars])

df_NormtrainB[num_vars] = scaler.fit_transform(df_NormtrainB[num_vars])
df_NormtestB[num_vars] = scaler.fit_transform(df_NormtestB[num_vars])


# In[11]:


# 1A
y_Newtrain = df_Newtrain.pop('price')
x_Newtrain = df_Newtrain
y_Newtest = df_Newtest.pop('price')
x_Newtest = df_Newtest

y = y_Newtrain.values
y = np.reshape(y, (len(y), 1))

x0 = np.ones((len(y), 1))
xVal_1A = x0

for i in range(len(num_vars) - 1):
    xi = df_Newtrain.values[:, i]
    xi = np.reshape(xi, (len(y),1))
    xVal_1A = np.hstack((xVal_1A, xi))
    
y_test = y_Newtest.values
y_test = np.reshape(y_test, (len(y_test), 1))

x0 = np.ones((len(y_test), 1))
xVal_1A_test = x0

for i in range(len(num_vars) - 1):
    xi = df_Newtest.values[:, i]
    xi = np.reshape(xi, (len(y_test),1))
    xVal_1A_test = np.hstack((xVal_1A_test, xi))

# 1B
y_NewtrainB = df_NewtrainB.pop('price')
x_NewtrainB = df_NewtrainB
y_NewtestB = df_NewtestB.pop('price')
x_NewtestB = df_NewtestB

y_1B = y_NewtrainB.values
y_1B = np.reshape(y_1B, (len(y_1B), 1))

x0 = np.ones((len(y_1B), 1))
xVal_1B = x0

for i in range(len(num_vars_b) - 1):
    xi = df_NewtrainB.values[:, i]
    xi = np.reshape(xi, (len(y_1B),1))
    xVal_1B = np.hstack((xVal_1B, xi))
    
y_test_1B = y_NewtestB.values
y_test_1B = np.reshape(y_test_1B, (len(y_test_1B), 1))

x0 = np.ones((len(y_test_1B), 1))
xVal_1B_test = x0

for i in range(len(num_vars_b) - 1):
    xi = df_NewtestB.values[:, i]
    xi = np.reshape(xi, (len(y_test_1B),1))
    xVal_1B_test = np.hstack((xVal_1B_test, xi))


# In[12]:


# 2A
y_Normtrain = df_Normtrain.pop('price')
x_Normtrain = df_Normtrain
y_Normtest = df_Normtest.pop('price')
x_Normtest = df_Normtest

y_2A = y_Normtrain.values
y_2A = np.reshape(y_2A, (len(y_2A), 1))

x0 = np.ones((len(y_2A), 1))
xVal_2A = x0

for i in range(len(num_vars) - 1):
    xi = x_Normtrain.values[:, i]
    xi = np.reshape(xi, (len(y_2A),1))
    xVal_2A = np.hstack((xVal_2A, xi))
    
y_test_2A = y_Normtest.values
y_test_2A = np.reshape(y_test_2A, (len(y_test_2A), 1))

x0 = np.ones((len(y_test_2A), 1))
xVal_2A_test = x0

for i in range(len(num_vars) - 1):
    xi = x_Normtest.values[:, i]
    xi = np.reshape(xi, (len(y_test_2A),1))
    xVal_2A_test = np.hstack((xVal_2A_test, xi))

# 2B
y_NormtrainB = df_NormtrainB.pop('price')
x_NormtrainB = df_NormtrainB
y_NormtestB = df_NormtestB.pop('price')
x_NormtestB = df_NormtestB

y_2B = y_NormtrainB.values
y_2B = np.reshape(y_2B, (len(y_2B), 1))

x0 = np.ones((len(y_2B), 1))
xVal_2B = x0

for i in range(len(num_vars_b) - 1):
    xi = x_NormtrainB.values[:, i]
    xi = np.reshape(xi, (len(y_2B),1))
    xVal_2B = np.hstack((xVal_2B, xi))
    
y_test_2B = y_NormtestB.values
y_test_2B = np.reshape(y_test_2B, (len(y_test_2B), 1))

x0 = np.ones((len(y_test_2B), 1))
xVal_2B_test = x0

for i in range(len(num_vars_b) - 1):
    xi = x_NormtestB.values[:, i]
    xi = np.reshape(xi, (len(y_test_2B),1))
    xVal_2B_test = np.hstack((xVal_2B_test, xi))


# In[13]:


epochs = 500
alpha = 0.02


# In[14]:


# 1A grad desc
theta_1A = np.zeros(shape = [len(num_vars), 1], dtype = float)
theta_1A, cost_history_1A_train, cost_history_1A_test = gradient_descent(xVal_1A, y, xVal_1A_test, y_test, theta_1A, epochs, alpha)


# plot the loss over epochs 1A
plt.plot(range(1, epochs + 1), cost_history_1A_train, color='blue')
plt.plot(range(1, epochs + 1), cost_history_1A_test, color='red')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[15]:


# 1B grad desc
theta_1B = np.zeros(shape = [len(num_vars_b), 1], dtype = float)
theta_1B, cost_history_1B_train, cost_history_1B_test = gradient_descent(xVal_1B, y, xVal_1B_test, y_test_1B, theta_1B, epochs, alpha)


# plot the loss over epochs 1B
plt.plot(range(1, epochs + 1), cost_history_1B_train, color='blue')
plt.plot(range(1, epochs + 1), cost_history_1B_test, color='red')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[16]:


# 2A grad desc
theta_2A = np.zeros(shape = [len(num_vars), 1], dtype = float)
theta_2A, cost_history_2A_train, cost_history_2A_test = gradient_descent(xVal_2A, y_2A, xVal_2A_test, y_test_2A, theta_2A, epochs, alpha)

# plot the loss over epochs 2A
plt.plot(range(1, epochs + 1), cost_history_2A_train, color='blue')
plt.plot(range(1, epochs + 1), cost_history_2A_test, color='red')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[17]:


# 2B grad desc
theta_2B = np.zeros(shape = [len(num_vars_b), 1], dtype = float)
theta_2B, cost_history_2B_train, cost_history_2B_test = gradient_descent(xVal_2B, y_2B, xVal_2B_test, y_test_2B, theta_2B, epochs, alpha)

# plot the loss over epochs 2A
plt.plot(range(1, epochs + 1), cost_history_2B_train, color='blue')
plt.plot(range(1, epochs + 1), cost_history_2B_test, color='red')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[18]:


# 3A grad desc
lamb = 1

theta_3A = np.zeros(shape = [len(num_vars), 1], dtype = float)
theta_3A, cost_history_3A_train, cost_history_3A_test = gradient_descent_param_penalties(xVal_2A, y_2A, xVal_2A_test, y_test_2A, theta_3A, epochs, alpha, lamb)

# plot the loss over epochs 3A
plt.plot(range(1, epochs + 1), cost_history_3A_train, color='blue')
plt.plot(range(1, epochs + 1), cost_history_3A_test, color='red')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[19]:


# 3B grad desc

theta_3B = np.zeros(shape = [len(num_vars_b), 1], dtype = float)
theta_3B, cost_history_3B_train, cost_history_3B_test = gradient_descent_param_penalties(xVal_2B, y_2B, xVal_2B_test, y_test_2B, theta_3B, epochs, alpha, lamb)

# plot the loss over epochs 3B
plt.plot(range(1, epochs + 1), cost_history_3B_train, color='blue')
plt.plot(range(1, epochs + 1), cost_history_3B_test, color='red')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[ ]:




