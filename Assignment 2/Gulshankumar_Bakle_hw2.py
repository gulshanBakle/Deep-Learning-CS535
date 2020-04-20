#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np


# In[2]:


import pickle
import numpy as np
import matplotlib.pyplot as plt


# In[3]:



# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step


class LinearTransform(object):

    def __init__(self, w, b):
        self.w=w*0.01
        self.b=b
        #print(self.W,self.b)

    def forward(self, x):
        self.x=x
        ltransform= np.dot(self.x,self.w)+self.b
        return ltransform
# DEFINE forward function

    def backward(self, grad_output, ip):
        de_by_dz1=np.dot(np.transpose(ip),grad_output)
        return de_by_dz1
    
    def update_wt_bias(self,old_momentum_w1,old_momentum_b1,de_by_dw,de_by_db,learning_rate,momentum,l2_penalty):
        
        momentum_w= momentum*old_momentum_w1 - learning_rate*(de_by_dw+ l2_penalty*self.w)
        momentum_b = momentum*old_momentum_b1 - learning_rate*(de_by_db + l2_penalty*self.b)
        self.w = self.w+momentum_w
        self.b =self.b+momentum_b
        
        return momentum_w,momentum_b
        
# DEFINE backward function
# ADD other operations in LinearTransform if needed


# This is a class for a ReLU layer max(x,0)
class Relu(object):
    def __init__(self):
        pass
    def forward(self,x):
        self.x=x
        Reluop= np.maximum(0,self.x)
        return Reluop
    # DEFINE forward function

    def backward(self, grad_output,lt1):
        lt1=(lt1>0)*1
        #self.Reluop[self.Reluop > 0] = 1
        de_by_dl1 = np.multiply(grad_output,lt1)
        return de_by_dl1
    # DEFINE backward function
# ADD other operations in ReLU if needed



# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self):
        #self.y=None
        #self.sigmoidf=None
        print()
    
    def forward(self, x):
        self.x=x
        self.sigmoidf=1/(1+np.exp(-self.x))
        return self.sigmoidf
# DEFINE forward function
    def backward(self,y,grad_output):
        self.y=y
        delta=(self.sigmoidf - self.y)
        return delta

    
# DEFINE backward function
# ADD other operations and data entries in SigmoidCrossEntropy if needed


# In[4]:


"""
Layer 1 is hidden layer
Layer 2 is classification or prediction layer
"""

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units,output_units):
        
    # INSERT CODE for initializing the network
        
        self.input_dims=input_dims
        self.hidden_units=hidden_units
        
        self.w1=np.random.randn(self.input_dims,self.hidden_units)
        self.b1=np.random.randn(1,hidden_units)
        self.w2=np.random.randn(self.hidden_units,1)
        self.b2=np.random.randn(1,1)
        
        #self.input_to_hidden_layer = LinearTransform(np.random.randn(input_dims, hidden_units), np.ones((1, hidden_units)))
        #self.hidden_to_output_layer = LinearTransform(np.random.randn(hidden_units, 1), np.ones((1, 1)))

        self.ip_to_hl=LinearTransform(self.w1,self.b1)
        self.hl_to_op=LinearTransform(self.w2,self.b2)
        
        self.relu=Relu()
        self.sigmoidce=SigmoidCrossEntropy()
        
        self.momentum_w1=0
        self.momentum_b1=0
        self.momentum_w2=0
        self.momentum_b2=0
        
        
        self.L1=0.0
        self.L2=0.0
        
    def train(self, x_batch, y_batch, learning_rate, momentum,l2_penalty):
        

#Training for only training data if flag=0


#Forward propogation

        
        ltransform_layer_1=self.ip_to_hl.forward(x_batch)
        reluOp=self.relu.forward(ltransform_layer_1)     #output of layer 1
        ltransform_layer_2=self.hl_to_op.forward(reluOp)
        sigmoidOp=self.sigmoidce.forward(ltransform_layer_2)    #output for layer layer 2
        y_pred_train=np.clip(sigmoidOp,1e-12, 1 - (1e-12))
        self.y_train_pred=y_pred_train
#Back propogation
            
            
        de_by_dl2=self.sigmoidce.backward(y_batch,sigmoidOp)
        dl2_by_dw2=reluOp
        de_by_dw2=self.hl_to_op.backward(de_by_dl2,dl2_by_dw2)
        de_by_db2=de_by_dl2
            
        dl2_by_dz1=self.hl_to_op.w
        de_by_dz1=np.dot(de_by_dl2,np.transpose(dl2_by_dz1))
        de_by_dl1=self.relu.backward(de_by_dz1,ltransform_layer_1)
        de_by_dw1=self.ip_to_hl.backward(de_by_dl1,x_batch)
        de_by_db1=de_by_dl1
            
            

#Update the weights

        self.momentum_w1,self.momentum_b1=self.ip_to_hl.update_wt_bias(self.momentum_w1,self.momentum_b1,de_by_dw1,de_by_db1,learning_rate,momentum,l2_penalty)
            
        self.momentum_w2,self.momentum_b2=self.hl_to_op.update_wt_bias(self.momentum_w2,self.momentum_b2,de_by_dw2,de_by_db2,learning_rate,momentum,l2_penalty)
            
            
#         self.momentum_w1= momentum*self.momentum_w1 - learning_rate*(de_by_dw1 + l2_penalty*self.w1)
#         self.momentum_b1= momentum*self.momentum_b1 - learning_rate*(de_by_db1 + l2_penalty*self.b1)
#         self.momentum_w2= momentum*self.momentum_w2 - learning_rate*(de_by_dw2 + l2_penalty*self.w2)
#         self.momentum_b2= momentum*self.momentum_b2 - learning_rate*(de_by_db2 + l2_penalty*self.b2)
        
        
            
#         self.w1 += self.momentum_w1
#         self.b1= self.b1+self.momentum_b1
#         self.w2=self.w2+self.momentum_w2
#         self.b2=self.b2+self.momentum_b2
        
       # print(self.w1)
#         import time
#         time.sleep(3)

            
#Calculate Losses

        self.training_loss = self.cross_entropy_loss(y_batch, self.y_train_pred, l2_penalty)    
        self.training_misclassification = self.misclassification_rate(y_batch, self.y_train_pred)
        return self.training_loss, self.training_misclassification    
 
    def evaluate(self, x, y,momentum,l2_penalty):
        
        ltransform_layer_1=self.ip_to_hl.forward(x)
        reluOp=self.relu.forward(ltransform_layer_1)     #output of layer 1
        ltransform_layer_2=self.hl_to_op.forward(reluOp)
        sigmoidOp=self.sigmoidce.forward(ltransform_layer_2)    #output for layer layer 2
        y_test=np.clip(sigmoidOp,1e-12, 1 - (1e-12))
        
        self.y_test_pred=y_test
        
        self.test_loss=self.cross_entropy_loss(y,self.y_test_pred,l2_penalty)
        self.test_misclassification=self.misclassification_rate(y,self.y_test_pred)
        
        return self.test_loss,self.test_misclassification
        
 
    def cross_entropy_loss(self,y,y_pred,l2_penalty):
        
        cross_entropy = (np.sum(np.square(self.ip_to_hl.w)) + np.sum(np.square(self.hl_to_op.w))) * (l2_penalty/ (2 * len(y))) + (np.dot(np.transpose(y), np.log(y_pred + 1e-12)) + np.dot((1 - np.transpose(y)), np.log(1 - y_pred + 1e-12)))
         #print(cumulative)

        avg_batch_cross_entropy_loss = (-1.0)* np.sum(cross_entropy)/len(y)
        
        return avg_batch_cross_entropy_loss
   
    
    
    def misclassification_rate(self,y,y_pred):
        
        y_pred = np.where(y_pred >= 0.5, np.ones(y_pred.shape), np.zeros(y_pred.shape))
        test = (y_pred == y)
        num_of_misclassifications = len((np.where(test==False))[0]) #shortcut
        #print("checkssss",misclassifications)
        #time.sleep(11
        return num_of_misclassifications
    
    
    
    
    
# INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed


# In[21]:


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        with open('cifar_2class_py2.p','rb') as f:
            u=pickle._Unpickler(f)
            u.encoding='latin1'
            dict=u.load()
            for i in dict:    
                print(i, dict[i].shape)

    train_x = dict['train_data']
    train_y = dict['train_labels']
    test_x = dict['test_data']
    test_y = dict['test_labels']

    num_examples, input_dims = train_x.shape
    test_exp=2000
    
    
    train_x=(train_x-np.amin(train_x,axis=0))/(np.amax(train_x,axis=0)-np.min(train_x,axis=0))
    test_x=(test_x-np.amin(test_x,axis=0))/(np.amax(test_x,axis=0)-np.min(test_x,axis=0))
    
    # Data normalization
    #x_train_max = np.max(train_x, axis=0)
    #train_x = train_x / x_train_max
    #x_test_max = np.max(test_x, axis=0)
    #test_x = test_x / x_test_max

    #x_train_mean = np.mean(train_x, axis=0)
    #train_x = (train_x - x_train_mean)
    #x_test_mean = np.mean(test_x, axis=0)
    #test_x = (test_x - x_test_mean)
    
    
    
    num_epochs = 10
    num_batches = 100
    learning_rate=0.0001
    momentum=0.8
    l2_penalty=0.0001
    #batch_size=32
    exp_per_batch=num_examples/num_batches
    exp_per_batch_test=len(test_x)/num_batches
    mlp = MLP(input_dims, hidden_units=32,output_units=1)
    
    #batch_train_loss, batch_train_accuracy, batch_val_accuracy, batch_val_loss = [], [], [], []

    for epoch in range(num_epochs):
        print("Epoch  ", epoch)
        misclassifiction_in_training=0.0
        misclassifiction_in_testing=0.0
        training_loss=0.0
        testing_loss=0.0
        #batch_loss,batch_acc=[],[]
        #num_of_batches=num_examples//batch_size
        #for b in range(num_of_batches):
         #   train_acc,train_loss=mlp.train()
        
        

    #Lets train our network here....
    


        for b in range(num_batches):
            batch_start=int((num_examples)/num_batches)*b
            batch_end=int((num_examples)/num_batches)*(b+1)
            
            x_batch=train_x[batch_start:batch_end]
            y_batch=train_y[batch_start:batch_end]
            
            batch_loss,batch_error=mlp.train(x_batch,y_batch,learning_rate, momentum,l2_penalty)
            
            training_loss=training_loss+batch_loss
            misclassifiction_in_training=misclassifiction_in_training+batch_error
            
            #print("training loss \t misclassificatrion in training", format(training_loss,misclassifiction_in_training))
            
        training_loss=training_loss/num_batches
        #print("Trainng loss for epoch "+str(epoch)+"is" +str(training_loss))
    #Lets test our network here....
   
        for b in range(int(exp_per_batch_test)):
            batch_start=int((num_examples)/num_batches)*b
            batch_end=int((num_examples)/num_batches)*(b+1)
            
            x_test_batch=test_x[batch_start:batch_end]
            y_test_batch=test_y[batch_start:batch_end]
            
            test_batch_loss,test_batch_error=mlp.evaluate(x_test_batch,y_test_batch, momentum,l2_penalty)
            
            testing_loss=testing_loss+test_batch_loss
            misclassifiction_in_testing=misclassifiction_in_testing+test_batch_error
            
            #print("testing loss \t misclassificatrion in testing".format(testing_loss,misclassifiction_in_testing))
          
        testing_loss=testing_loss/num_batches
        #print("Testing loss per epoch "+str(epoch)+"is"+str(testing_loss))

        sys.stdout.flush()
        # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        print()
        train_accuracy=(num_examples-misclassifiction_in_training)/num_examples
        test_accuracy=(test_exp-misclassifiction_in_testing)/test_exp
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(training_loss,100. * train_accuracy,))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(testing_loss,100. * test_accuracy,))


# In[57]:


ta_hu=[80.15,80.5,80.85,81.8,82.25,83]
hu=[16,32,64,128,512,1024]
plt.plot(hu,ta_hu)
plt.ylabel('Test accuracy')
plt.xlabel('number of hidden units')
plt.title('Hidden units vs test accuracy')
plt.show()


# In[14]:


ta_lr=[78.80,80.80,78.45,76.9,]
learning_rates=[0.00001,0.0001,0.0004,0.001,]
plt.plot(ta_lr,learning_rates)
plt.ylabel('Learning rate')
plt.xlabel('test accuracy')
plt.title('Learning rates vs test accuracy')
plt.show()


# In[22]:


ta_bs=[75.5,77.2,80.80,78.2,79.3]
bs=[10,50,100,200,500]
plt.plot(bs,ta_bs)
plt.ylabel('test accuarcy')
plt.xlabel('batch sizes')
plt.title('batch size vs test accuarcy')
plt.show()

