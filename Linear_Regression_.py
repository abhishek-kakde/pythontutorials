#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation 
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML





# In[3]:


#Load the dataset
boston=load_boston()

#Description of the dataset 
print(boston.DESCR)


# In[4]:


#Put the data into pandas DataFrames
features = pd.DataFrame(boston.data,columns=boston.feature_names)
features



# In[5]:


features['AGE']



# In[6]:


target = pd.DataFrame(boston.target,columns=['target'])
target 



# In[7]:


max(target['target'])



# In[8]:


min(target['target'])


# In[9]:


# Concatenate features and target into a single DataFrame 
# axis = 1 makes it concatenate column wise
df=pd.concat([features,target],axis=1)
df


# In[10]:


#Use round(decimals=2) to set the precision to 2 decimal places 
df.describe().round(decimals=2)


# In[11]:


# calculate correlation between every column of the data 
corr = df.corr('pearson')

# Take absolute values of correlations
corrs = [abs(corr[attr]['target']) for attr in list(features)]

# Make a list of pairs [(corr,feature)]
l = list(zip(corrs, list (features)))

# slot the list of pairs in reverse/descending order, 
# with the correlation value as the key of sorting 
l.sort(key = lambda x : x[0], reverse = True)

# "Unzip" pairs to 2 lists
# zip(*l)- takes the list that looks like [[a,b,c], [d,e,f], [g,h,i]]
# and returns [[a,d,g], [b,e,h], [c,f,i]]
corrs, labels= list(zip((*l)))

#plot correlation with respect to target variable as a bar graph
index = np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index, corrs, width = 0.5)
plt.xlabel("Attrributes")
plt.ylabel("Correlation with the target variable")
plt.xticks(index, labels)
plt.show()






# In[59]:


#NORMALIZATION 
X=df['LSTAT'].values
Y=df['target'].values


# In[60]:


#Before Normalization 
print(Y[:5])


# In[61]:


x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X.reshape(-1, 1))
X = X[:, -1]
y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y.reshape(-1, 1))
Y = Y[:, -1]



# In[62]:


# After normalization 
print(Y[:5])


# In[63]:


def error(m, x, c, t):
    N = x.size
    e = sum(((m * x + c ) - t) ** 2)
    return e * 1/(2 * N)



# In[64]:


#SPLITTING THE DATASET 
#0.2 indicates 20% data is randomly sampled as testing data 
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2)


# In[65]:


def update(m, x, c, t, learning_rate):
    grad_m = sum (2 * ((m * x + c) - t) * x)
    grad_c = sum (2 * ((m * x + c) - t))
    m = m - grad_m * learning_rate 
    c = c - grad_c * learning_rate
    return m, c


# In[66]:


def gradient_descent(init_m, init_c, x, t, learning_rate, iterations, error_threshold):
    m = init_m
    c = init_c
    error_values = list()
    mc_values = list()
    for i in range(iterations):
        e = error(m, x, c, t)
        if e < error_threshold:
            print('Error less than the threshold. Stopping gradient descent')
            break 
        error_values.append(e)
        m, c = update(m, x, c, t, learning_rate)
        mc_values.append((m, c))
    return m, c, error_values, mc_values


# In[67]:


get_ipython().run_cell_magic('time', '', 'init_m = 0.9\ninit_c = 0\nlearning_rate =  0.001\niterations =  250\nerror_threshold =  0.001 \n\nm, c, error_values, mc_values = gradient_descent(init_m, init_c, xtrain, ytrain, learning_rate, iterations, error_threshold)\n\n\n')


# In[68]:


#plot the regression curve
plt.scatter(xtrain, ytrain, color = 'b')
plt.plot(xtrain, (m*xtrain+c), color= 'r')


# In[69]:


#plot error
plt.plot(np.arange(len(mc_values)),error_values)
plt.ylabel("Error")
plt.xlabel("Iterations")


# In[70]:


#MODEL TRAINING VISUALIZATION 
#As the number of iterations increases, changes in the lines are less noticable.
#in order to reduce the processing time for the animation, it is advised to choose smaller values 
mc_values_anim = mc_values[0:250:5]



# In[71]:


fig, ax = plt.subplots()

ln, = plt.plot([], [], 'ro-', animated=True)

def init():
    plt.scatter(xtrain, ytrain, color = 'b')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    return ln, 

def update_frame(frame): 
    m, c = mc_values_anim[frame]
    x1, y1 = -0.5, m*-.5+c
    x2, y2 = 1.5, m*1.5+c
    ln.set_data([x1, x2] , [y1, y2])
    return ln, 

anim= FuncAnimation(fig, update_frame, frames=range(len(mc_values_anim)), init_func=init, blit=False)

HTML(anim.to_html5_video())





# In[72]:


# PREDICTIONS OF PRICES 

# Calculate the predictions on the test set as a vectorized operation

predicted = (m * xtest) + c 



# In[73]:


#Compute the MSE for the predicted values on the testing set 
mean_squared_error(ytest, predicted)


# In[74]:


#Put xtest, ytest and predicted values into single dataframe so that we
#can see the predicted values along the testing set 
p = pd.DataFrame(list(zip(xtest, ytest, predicted)), columns = ['x', 'target_y', 'predicted_y'])
p.head()


# In[75]:


#PLOTING PREDICTED VALUES AGAINST THE TARGET VALUES 
plt.scatter(xtest, ytest, color = 'b')
plt.plot(xtest, predicted, color= 'r')


# In[76]:


#REVERT NORMALIZATION TO OBTAIN THE PREDICTED PRICES OF HOUSES IN $1000s
# Reshape to change the shape that is required by the scaler 
predicted = np.array(predicted).reshape(-1,1)
xtest = xtest.reshape(-1, 1)
ytest = ytest.reshape(-1, 1)

xtest_scaled = x_scaler.inverse_transform(xtest)
ytest_scaled = y_scaler.inverse_transform(ytest)
predicted_scaled = y_scaler.inverse_transform(predicted)

#This is to remove the extra dimension 
xtest_scaled = xtest_scaled [:, -1]
ytest_scaled = ytest_scaled [:, -1]
predicted_scaled = predicted_scaled [:, -1]
p = pd.DataFrame(list(zip(xtest, ytest, predicted)), columns = ['x', 'target_y', 'predicted_y'])
p = p.round(decimals = 2)
p.head()


# In[ ]:




