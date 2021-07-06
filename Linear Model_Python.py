#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing packages to use
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv('regrex1.csv')


# In[3]:


dataset.shape


# In[4]:


dataset.describe ()


# In[5]:


dataset.plot(x='x', y='y', style='.')  
plt.title('Example Linear Model')  
plt.xlabel('X')  
plt.ylabel('Y')  
plt.show()


# In[6]:


x = dataset['x'].values.reshape(-1,1)
y = dataset['y'].values.reshape(-1,1)


# In[7]:


x_train = x
y_train = y


# In[8]:


regressor = LinearRegression()  
regressor.fit(x_train, y_train) #training the algorithm


# In[9]:


y_pred = regressor.predict(x)


# In[10]:


plt.scatter(x, y,  color='black', marker='.')
plt.plot(x, y_pred, color='black', linewidth=1)
plt.show()


# In[ ]:




