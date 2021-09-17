#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
def gradient_descent(xVal, yVal, theta, epochs, alpha):
    costHist = np.zeros(epochs)
    for i in range(epochs):
        hVal = xVal.dot(theta)
        error = hVal - yVal
        
        theta = theta - (alpha/len(yVal)) * xVal.transpose().dot(error)        
        costHist[i] = loss_function(xVal, yVal, theta)
    return theta, costHist
    


# In[4]:


# read in data set to d3, 99 rows, 4 cols
d3 = pd.read_csv("D3.csv")
x1 = d3.values[:,0]
x2 = d3.values[:,1]
x3 = d3.values[:,2]
y = d3.values[:,3]

m = len(x1)


# In[5]:


# temp value to hold x0
temp = np.ones((m,1))

# turning the data into vertical arrays
x1 = np.reshape(x1,(m,1))
x2 = np.reshape(x2,(m,1))
x3 = np.reshape(x3,(m,1))
y = np.reshape(y,(m,1))

# create arrays of [x0, x1], [x0, x2], etc
xVal1 = np.hstack((temp, x1))
xVal2 = np.hstack((temp, x2))
xVal3 = np.hstack((temp, x3))

theta1 = np.zeros(shape = [2,1], dtype = float)
theta2 = np.zeros(shape = [2,1], dtype = float)
theta3 = np.zeros(shape = [2,1], dtype = float)


# In[6]:


# training params
epochs = 1000
alpha = 0.01


# In[7]:


# calc and plot the linear regression model for the first column x values with the data points
theta1, cost_history1 = gradient_descent(xVal1, y, theta1, epochs, alpha)

plt.scatter(xVal1[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(xVal1[:,1],xVal1.dot(theta1),color='green',label='Linear Regression')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.title('Linear Regression Fit')
plt.legend()


# In[8]:


# plot the loss over epochs
plt.plot(range(1,epochs+1),cost_history1,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[9]:


# calc and plot the linear regression model for the second column x values with the data points
theta2, cost_history2 = gradient_descent(xVal2, y, theta2, epochs, alpha)

plt.scatter(xVal2[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(xVal2[:,1],xVal2.dot(theta2),color='green',label='Linear Regression')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.title('Linear Regression Fit')
plt.legend()


# In[10]:


# plot the loss over epochs

plt.plot(range(1,epochs+1),cost_history2,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[11]:


# calc and plot the linear regression model for the third column x values with the data points
theta3, cost_history3 = gradient_descent(xVal3, y, theta3, epochs, alpha)

plt.scatter(xVal3[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(xVal3[:,1],xVal3.dot(theta3),color='green',label='Linear Regression')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.title('Linear Regression Fit')
plt.legend()


# In[12]:


# plot the loss over epochs

plt.plot(range(1,epochs+1),cost_history3,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[18]:


print("Theta values from x1 \n", theta1)
print("Theta values from x2 \n", theta2)
print("Theta values from x3 \n", theta3)


# In[14]:


# Question 2

# create an array of [x0, x1, x2, x3]
xVal = np.hstack((temp, x1, x2, x3))
theta = np.zeros(shape = [4,1], dtype = float)

# run the linear regression
theta, cost_history = gradient_descent(xVal, y, theta, epochs, alpha)


# In[15]:


# plot the loss over epochs

plt.plot(range(1,epochs+1),cost_history,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of epochs')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[16]:


print("estimation of (1, 1, 1)", theta[0] + theta[1] + theta[2] + theta[3])
print("estimation of (2, 0, 4)", theta[0] + 2 * theta[1] + 0 * theta[2] + 4 * theta[3])
print("estimation of (3, 2, 1)", theta[0] + 3 * theta[1] + 2 * theta[2] + theta[3])


# In[ ]:




