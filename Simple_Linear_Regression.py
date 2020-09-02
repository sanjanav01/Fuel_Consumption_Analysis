#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')


# In[4]:


df = pd.read_csv("FuelConsumption.csv")
df.head()


# In[ ]:


df.describe()


# In[9]:


cdf = df [['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)


# In[10]:


viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# In[12]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


# In[13]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color = 'red')
plt.xlabel("Engine Size")
plt.ylabel("Emissions")
plt.show()


# In[14]:


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color = 'green')
plt.xlabel("Cylinder")
plt.ylabel("Emissions")
plt.show()


# In[15]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[16]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='purple')
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()


# In[17]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

print ('Ceofficients:', regr.coef_)
print ('Intercept:', regr.intercept_)


# In[32]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'orange')
plt.plot(train_x, regr.coef_*train_x + regr.intercept_, '-g')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()


# In[21]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_p = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_p - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_p - test_y)**2))
print("R2 - score: %.2f" % r2_score(test_p, test_y))


# In[ ]:




