## NSC 3270/6270 Spring 2021
## Final Project

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as R

# import tools for basic keras networks 
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# supress unnecessary warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#############################################################################################################################
## PART 1 - First, we create a single-layer neural network in Keras and train it to discriminate
## two classes of input patterns. Then, we create a testing array and use that to create a visualization
## of what the network has learned. The network should have a single output node.
## 
## To train the neural network, we build an Np x Nin array of training patterns and an Np x Nout
## array of teacher values. Keras adds the bias parameters automatically when it creates the network.
## We train the network using a validation_split=0.1.
##
## We then create a graph showing how 'loss' (error on the training data), and 'val_loss' (error on the held-out
## validation data) change over the course of training.  X-axis shows Epochs, and Y-axis is the loss value.
## Then we generate the meshgrid equally-spaced testing array, generate predictions from this test array using
## the network.predict() function in Keras, and display the results.
#############################################################################################################################

## generate N random samples from a 2D multivariate normal distribution
## with mean [mx, my]
## with covariance matrix [[  sx*sx, r*sx*sy],
##                         [r*sx*sy,   sy*sy]]

def gensamples(N, mx, my, sx, sy, r):
    M   = np.array([mx, my])
    Cov = np.array([[  sx*sx, r*sx*sy],
                  [r*sx*sy,   sy*sy]])
    return (R.multivariate_normal(M, Cov, size=N, check_valid='warn'))

## create two classes of training patterns using gensamples().

N = 500
mx0 = 0.;   my0 = 0.;   sx0 = 1.;   sy0 = 1.;   r0 = 0.
sample0 = gensamples(N, mx0, my0, sx0, sy0, r0)
mx1 = 5.;   my1 = 5.;   sx1 = 1.;   sy1 = 1.;   r1 = 0.
sample1 = gensamples(N, mx1, my1, sx1, sy1, r1)

def plottrain(sample0, sample1):
    # plot example
    plt.plot(sample0[:,0],sample0[:,1],'b+',sample1[:,0],sample1[:,1],'r+')
    plt.xlabel('dim1')
    plt.ylabel('dim2')
    plt.axis('equal')
    plt.axis('square')
    plt.legend(('class 0', 'class 1'), loc='lower right')
    xymin = -5;   xymax = 10
    plt.ylim((xymin,xymax))
    plt.xlim((xymin,xymax))
    plt.show()

# this will show a plot with the two sets of training patterns

plottrain(sample0, sample1)

######################################
## Code to set up and train network ##
######################################
Np, nin = sample0.shape
train = np.concatenate((sample0, sample1), axis=0)                                    #creates training array
teach = np.concatenate((np.zeros((Np,1)), np.ones((Np, 1))), axis=0)                  #creates teacher array

nin  = train.shape[1]
nout = teach.shape[1]  

network = models.Sequential()                                                          #creates/sets up network
network.add(layers.Dense(nout, activation='sigmoid',input_shape=(nin,)))               #single layer

network.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])                     #compiles network
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)                                               #selects optimizer
history = network.fit(train, teach, verbose=True, validation_split=.1, epochs=600, batch_size=300)    #trains network

val_loss = history.history['val_loss'] 
loss = history.history['loss']
plt.figure()
plt.plot(loss, label='loss')                    #plots loss over the course of training
plt.plot(val_loss, label='val_loss')            #plots val_loss over the course of training
plt.legend()
plt.ylabel("Loss Value")
plt.xlabel("Epoch Number")

##########################################
## Code to generate network predictions ##
##########################################
xymin = -5
xymax = 10
Npts = 50
xv, yv = np.meshgrid(np.linspace(xymin, xymax, Npts), np.linspace(xymin, xymax, Npts))
test = np.concatenate((xv.reshape((Npts*Npts,1)), yv.reshape(Npts*Npts,1)), axis=1)          #creates test array

out = network.predict(test)          #generates network predictions

# this function creates a filled contour plot using xv, yv, and the network output
# predictions (out) from test

def plottest(xv, yv, out, sample0, sample1):
    # reshape out
    zv = out.reshape(xv.shape)

    # create figure
    fig = plt.figure()
    plt.contourf(xv, yv, zv, levels=xv.shape[0], cmap=plt.cm.gist_yarg)
    plt.plot(sample0[:,0],sample0[:,1],'b+',sample1[:,0],sample1[:,1],'r+')    
    
    plt.xlabel('dim1')
    plt.ylabel('dim2')
    plt.axis('equal')
    plt.axis('square')
    xymin = -5;   xymax = 10
    plt.ylim((xymin,xymax))
    plt.xlim((xymin,xymax))
    plt.show()

# plot filled contour plot of test predictions

plottest(xv, yv, out, sample0, sample1)
#############################################################################################################################




#############################################################################################################################
## PART 2 - We now create and train a neural network with Keras to learn the XOR classification problem.
#############################################################################################################################


###########################################################################
## Code to create the training patterns and teachers for the XOR problem ##
###########################################################################
trainXOR0=np.concatenate((sample0, sample1), axis=0)                          #creates first set of patterns

sample2 = gensamples(N, 0, 5, 1, 1, 0)
sample3 = gensamples(N, 5, 0, 1, 1, 0)  
trainXOR1=np.concatenate((sample2, sample3), axis=0)                          #creates second set of patterns

trainXOR= np.concatenate((trainXOR0, trainXOR1), axis=0)                       #combines two training pattern arrays
teachXOR=np.concatenate((np.zeros((2*Np,1)), np.ones((2*Np, 1))), axis=0)      #creates teacher array

######################################
## Code to set up and train network ##
######################################
nin  = trainXOR.shape[1]
nout = teachXOR.shape[1]
nhid = 10                                  #nodes in hidden layer

network = models.Sequential()                                                   #sets up network
network.add(layers.Dense(nhid, activation='relu', input_shape=(nin,)))          #hidden layer
network.add(layers.Dense(nout, activation='sigmoid'))

network.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])                               #compiles network
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)                                                         #optimizer
history = network.fit(trainXOR, teachXOR, verbose=True, validation_split=.1, epochs=700, batch_size=200)        #trains network

##########################################
## Code to generate network predictions ##
##########################################
out = network.predict(test)
plottest(xv, yv, out, trainXOR0, trainXOR1)
#############################################################################################################################



#############################################################################################################################
## PART 3 - Here we create our own classification problem (two classes) that is
## more complicated than a simple XOR problem.
#############################################################################################################################

N=500
samp1 = gensamples(N, 1, 5, 0.5, 1, 0)
samp2 = gensamples(N, 8, 2, 1, 0.3, 0.4)
samp3 = gensamples(N, 2.5, -2, 0.6, 1, 0.7)
set1 = np.concatenate((samp1, samp2, samp3), axis=0)         #creates first set    

samp4 = gensamples(N, 7, -3, 0.9, 0.6, 0.2)
samp5 = gensamples(N, 5, 7, 0.6, 2, 0.9)
samp6 = gensamples(N, -1, -1, 1, 0.8, 0.99)
set2 = np.concatenate((samp4, samp5, samp6), axis=0)         #creates second set

trainpats = np.concatenate((set1, set2), axis=0)                                    #combines sets to make training array
teachpats = np.concatenate((np.zeros((3*N,1)), np.ones((3*N, 1))), axis=0)          #creates teacher array

nin  = trainpats.shape[1]
nout = teachpats.shape[1]
nhid = 30                              #nodes in hidden layer

network = models.Sequential()                                                   #sets up network
network.add(layers.Dense(nhid, activation='relu', input_shape=(nin,)))          #hidden layer
network.add(layers.Dense(nout, activation='sigmoid'))

network.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])                               #compiles network
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)                                                         #optimizer
history = network.fit(trainpats, teachpats, verbose=True, validation_split=.1, epochs=800, batch_size=200)      #trains network

out = network.predict(test)                  #network predictions
plottest(xv, yv, out, set1, set2)            
#############################################################################################################################
