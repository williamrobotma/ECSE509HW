#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
import os

data_dir ='data'


# # Question 2.1 - Load Data

# Download prostate data from https://web.stanford.edu/~hastie/ElemStatLearn/data.html
# 
# Take x1 = lcavol, x2 = lweight, x3 = age, x4 = lbph, x5 = svi, x6 = lcp, x7 = gleason, x8 = pgg45,
# and define the response variable y = lpsa. Standardize all variables. Show the 8 scatter plot of y against each x1, . . . , x8

# In[2]:


prostate_df = pd.read_csv(os.path.join(data_dir, 'prostate.data'),sep='\t', index_col=0)
prostate_df


# We don't need train so we drop it

# In[3]:


prostate_df = prostate_df.drop('train', axis=1)
prostate_df


# In[4]:


var_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y']
prostate_df = prostate_df.reset_index(drop=True)
prostate_df.columns = var_names
prostate_df


# Standardize

# In[5]:


from sklearn.preprocessing import StandardScaler

prostate_df.iloc[:,:] = StandardScaler().fit_transform(prostate_df.values)

prostate_df


# In[6]:


fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize = (10, 5))

for i, ax in enumerate(axs.flat):
    ax.scatter(prostate_df.iloc[:,i], prostate_df.iloc[:,-1])
    ax.set_xlabel(prostate_df.columns[i])
    ax.set_ylabel(prostate_df.columns[-1])
    
fig.suptitle('Scatterplots for each feature against y')
plt.show()


# In[7]:


prostate_df.to_pickle(os.path.join(data_dir, 'prostate_df.pkl'))


# # Question 2.2

# Use a Markov chain Monte Carlo sampler to take samples from the posterior of w in Question 1 over this data set and visualize your samples, do they converge?
# 

# In[8]:


prostate_df = pd.read_pickle(os.path.join(data_dir, 'prostate_df.pkl'))


# ## Find estimation of $\lambda$ and $\sigma^2$ using least squares

# Fit a linear model to the data (we know it to be linear)

# In[9]:


from sklearn.linear_model import LinearRegression

X,y = prostate_df.drop('y', axis=1), prostate_df['y']
lr = LinearRegression(fit_intercept=False).fit(X,y)


# Using least squares predicted value as the mean, find variance

# $\sigma^2=(\Sigma_{i=1}^n(\hat{y_i}-y_i)^2)/(n-1)$

# In[10]:


var_y = np.square(lr.predict(X) - y).sum()/(y.size - 1)


# $\lambda=(\Sigma_{i=1}^N(w_i)^2)/(N-1)$

# Find the variance of all the ws

# In[11]:


w_hat = lr.coef_
lam = np.square(w_hat).sum()/(w_hat.size - 1)


# In[12]:


print(f'Variance of y: {var_y}, Lambda: {lam}')


# Create MCMC sampler

# In[13]:


from scipy.stats import multivariate_normal as mn
np.random.seed(235)

mu_w = np.zeros(X.shape[1])
cov_w = np.eye(X.shape[1])*lam
cov_y = np.eye(X.shape[0])*var_y
scale = 0.005

# This is our PDF from 1.4, but log transformed
posterior_log = lambda w, X, y : np.log(mn.pdf(y, np.matmul(X, w), cov_y)) + np.log(mn.pdf(w, mu_w, cov_w))
# Random walk transitions
RW = lambda w, scale: np.random.multivariate_normal(w, scale*np.eye(len(w)))
# Initialize w according to its known multivariate normal distribution
w_0 = np.random.multivariate_normal(mu_w, cov_w)
w = w_0


# In[14]:


iterations = 10000
history = np.zeros((iterations, X.shape[1]))
accepted = np.zeros(iterations, dtype=bool)
for i in range(iterations):
    log_p = posterior_log(w, X, y)
    
    # walk w
    w_new = RW(w, scale)
    log_p_new = posterior_log(w_new, X, y)
    
    # sample a minimum ratio 
    r = np.random.uniform(0,1)

    # transform back from log to normal
    if np.exp(log_p_new - log_p) > r:
        w = w_new
        accepted[i] = True
    
    history[i] = w_new

    # print(w)


# In[15]:


accepted[:-10]


# In[30]:


fig, axs = plt.subplots(X.shape[1], 2, figsize = (15,15))

for i, ax in enumerate(axs):
    iteration = np.arange(iterations)
    ax[0].plot(iteration[~accepted], history[~accepted,i], 'bx', label='Rejected',alpha=0.5)
    ax[0].plot(iteration[accepted], history[accepted,i], 'r.', label='Accepted',alpha=1)
    ax[0].set_xlabel('iterations')
    ax[0].set_ylabel(f'w_{i}')
    ax[0].legend()
    ax[0].set_title('Trace of Samples')
    
    ax[1].hist(history[len(history)//2:,i], bins=100)
    # ax[1].hist(history[accepted,i][len(history[accepted,i])//2:], bins=100) # drop 1st 50% in hist
    ax[1].set_ylabel('Samples')
    ax[1].set_xlabel(f'w_{i}')
    ax[1].set_title('Distribution of Samples')
    
# fig.suptitle('Traces and histogram for each element of w')
plt.tight_layout()
plt.show()


# Compare MSE of initial w vs that from linear regression to see if it converged:

# In[17]:


accepted_ws = history[~accepted]
w_est = np.mean(accepted_ws[len(accepted_ws) // 2:], axis=0)
w_est2 = np.mean(history[len(history) // 2:], axis=0)
print("MSEs for each estimate against w obtained from least squares:")
print("initial w: ", np.mean((w_0-w_hat)**2))
print("average w across entire history: ", np.mean((w_est-w_hat)**2))
print("average w across last 50% of samples: ", np.mean((w_est2-w_hat)**2))


# # Question 2.3

# In[18]:


# calculate posterior likelihood for each sample
log_ps = np.array([posterior_log(his, X, y) for his in history])
w_MAP = history[np.argmax(log_ps)]


# In[ ]:


# calculate posterior likelihood using equation using lambda' = 0


# In[27]:


w_MAP2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
w_MAP.tolist(), w_MAP2.tolist()


# # Question 2.4

# In[20]:


w_est.shape


# In[21]:


prostate_df = pd.read_pickle(os.path.join(data_dir, 'prostate_df.pkl'))


# In[22]:


from sklearn.decomposition import PCA

prostate_pc = PCA().fit_transform(prostate_df.drop('y', axis=1))


# In[23]:


prostate_pc.shape


# In[24]:


plt.scatter(prostate_pc[:, 0], prostate_pc[:, 1])
plt.xlabel('z1')
plt.ylabel('z2')
plt.title('First two principle components of prostate data features')
plt.show()


# In[ ]:





# In[ ]:




