# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
# movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
movies = pd.read_csv('C:/Users/Marcelo/ICLOUD~1/Work/CASA-P~1/0_PROJ~1/Study/MESTRADO/Udemy/DEEP_L~1/VOLUME~2/PART5-~1/SECTIO~3/P16-BO~1/BOLTZM~1/ml-1m/movies.dat', sep = '::',header = None, engine = 'python', encoding = 'latin-1')

# users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('C:/Users/Marcelo/ICLOUD~1/Work/CASA-P~1/0_PROJ~1/Study/MESTRADO/Udemy/DEEP_L~1/VOLUME~2/PART5-~1/SECTIO~3/P16-BO~1/BOLTZM~1/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('C:/Users/Marcelo/ICLOUD~1/Work/CASA-P~1/0_PROJ~1/Study/MESTRADO/Udemy/DEEP_L~1/VOLUME~2/PART5-~1/SECTIO~3/P16-BO~1/BOLTZM~1/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


# Preparing the training set and the test set
#training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = pd.read_csv('C:/Users/Marcelo/ICLOUD~1/Work/CASA-P~1/0_PROJ~1/Study/MESTRADO/Udemy/DEEP_L~1/VOLUME~2/PART5-~1/SECTIO~3/P16-BO~1/BOLTZM~1/ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

#test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = pd.read_csv('C:/Users/Marcelo/ICLOUD~1/Work/CASA-P~1/0_PROJ~1/Study/MESTRADO/Udemy/DEEP_L~1/VOLUME~2/PART5-~1/SECTIO~3/P16-BO~1/BOLTZM~1/ml-100k/u1.test', delimiter = '\t', engine='python')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
# self (the object created afterwards); nv (number of visible nodes); nh (number of hidden nodes)
# W (weights); a (bias for probability nh given nv); b (bias for probabiçity nv given nh)
# sample_h: it was created a sigmoid function to activate the hidden nodes given the visible nodes
# the result will be the probability of the hidden nodes given the visible nodes (p_h_given_v)
# to be mathematically correct, get the transpose of the weights (W) by using t() 
# return bernoulli samples of the weights from hidden nodes given visible nodes
# sample_v: it was done the same as done in sample_h but now applied to the visible nodes given the hidden nodes
# with sample_v it is not necessary to get the transpose of the weights
# All common training algorithms for RBMs approximate the log-likelihood gradient
# given some data and perform gradient ascent on these approximations
# in order to minimize the energy (source: "A Tutorial on Energy-Based Learning" - Yann LeCun et al.).
# So it was necessary to build a function to train by using contrastive divergence learning
# which obtains estimates after running the chain for just a few steps (it is sufficient for model training)
# source: "An Introduction to Restricted Boltzmann Machines" - Asja Fischer and Christian Igel
# Function train is built by implementing Algorithm 1. k-step contrastive divergence (Ficher and Igel paper, p.28).
# From the algorithm, I implemented only lines 8, 9 and 10, 
# which were the lines responsible for computing the constrative divergence.

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
# Creating the parameters for the function __init__ takes action
# nh and batch size are tunable, I defined 100 nh (hidden nodes) and 100 batch_size.
nv = len(training_set[0])
nh = 100
#nh = len(training_set[0])
batch_size = 100
rbm = RBM(nv, nh)

# To deal with "RuntimeError: The size of tensor a (1682) must match the size of tensor b (100) at non-singleton dimension 1"
# It was needed to invert the transpose from hidden nodes to visible nodes

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nv, nh) # was torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
   
    def sample_h(self, x):
        wx = torch.mm(x, self.W) # was torch.mm(x, self.W.t())    
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W.t()) # was torch.mm(y, self.W)  
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
 
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# Training the RBM
# the third for loop computs the Gibbs chain (based on Bengio and Delalleau's theorem)
# also, the third loop is the implementation of lines 5 and 6 from Algorithm 1. k-step contrastive divergence
# to normalize the train loss, divide it by the counter (s).

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))

# Save/Load Entire Model
# source: https://pytorch.org/tutorials/beginner/saving_loading_models.html
# Save: torch.save(model, PATH)
PATH = 'C:/Users/Marcelo/ICLOUD~1/Work/CASA-P~1/0_PROJ~1/Study/MESTRADO/Udemy/DEEP_L~1/VOLUME~2/PART5-~1/SECTIO~3/P16-BO~1/BOLTZM~1'

# torch.save(model, PATH)

# Load
# Model class must be defined somewhere
# model = torch.load(PATH)
# model.eval()

# Evaluating the Boltzmann Machine
# the two ways of evaluating our RBM are with the RMSE and the Average Distance.

# RMSE:
# The RMSE (Root Mean Squared Error) is calculated as the root of the mean 
# of the squared differences between the predictions and the targets.

# The code that computes the RMSE:

# Training phase

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Test phase

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
        s += 1.
print('test loss: '+str(test_loss/s))

# Using the RMSE, this RBM would obtain an error around 0.46. 
# But be careful, although it looks similar, one must not confuse the RMSE and the Average Distance. 
# A RMSE of 0.46 doesn’t mean that the average distance between the prediction and the ground truth is 0.46. 
# In random mode we would end up with a RMSE around 0.72. 
# An error of 0.46 corresponds to 75% of successful prediction.

# Average Distance:

# Training phase:

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0])) # Average Distance here
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Test Phase

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) # Average Distance here
        s += 1.
print('test loss: '+str(test_loss/s))

# With this metric, we obtained an Average Distance of 0.24, which is equivalent to about 75% of correct prediction.

# Hence, it works very well and there is a predictive power.

# If you want to check that 0.25 corresponds to 75% of success, you can run the following test:

import numpy as np
u = np.random.choice([0,1], 100000)
v = np.random.choice([0,1], 100000)
u[:50000] = v[:50000]
sum(u==v)/float(len(u)) # -> you get 0.75
np.mean(np.abs(u-v)) # -> you get 0.25

