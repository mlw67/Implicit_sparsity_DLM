# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:51:42 2020

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cvx
import os
import pandas as pd
import seaborn as sns
import sys

################### Generate data ###################

def gen_problem(k,n,d,sigma):
    #wstar = np.zeros(d)
    #support = np.random.choice(np.arange(d), k, replace=False)
    #wstar[support] = np.random.randn(k)
    wstar = np.append(np.array([8,6,4,2,10]),np.zeros(d-k))
    print(wstar)
    X = np.random.randn(n,d)
    y = np.dot(X,wstar) #+ sigma*np.random.randn(n)
    return X,y,wstar#,support

class power_uc_iter(object):
    def __init__(self, power, eta=0.01, MAXIter=int(1e7), tol=1e-6):
        self.power = power
        self.eta = eta
        self.MAXIter = MAXIter
        self.tol = tol
        
    def objective(self,X,y):
        return (1./2.)*np.linalg.norm(np.dot(X, self.u**self.power) - y)**2

    def gradient_u(self,X,y):
        residual = np.dot(X, self.u**self.power) - y
        return 1./(X.shape[0])*self.power*np.multiply(np.dot(np.transpose(X), residual), self.u**(self.power-1))

    def fit(self,u0,X,y):
        self.u = u0
        u_list = []
        u_list.append(self.u**self.power)
        for t in range(self.MAXIter):
            if t % int(1000) == 0:
                print('\t{:d}k iterations, Loss: {:6.6f}\r'.format(int(t/1000),self.objective(X,y)),end='')
                #print('',self.u**self.power)
            self.u = self.u - self.eta*self.gradient_u(X,y)
            
            if t % 100 == 99:
                u_list.append(self.u**self.power)
                if self.objective(X,y) < self.tol:
                    return self.u,u_list, t+1
        print('\n')
        return self.u,u_list,t+1

    def predict(self, X):
        return np.dot(X, self.u**self.power)

class power_uc_res_iter(object):
    def __init__(self, power, eta=0.01, MAXIter=int(1e7), tol=1e-6):
        self.power = power
        self.eta = eta
        self.MAXIter = MAXIter
        self.tol = tol
        
    def objective(self,X,y):
        return (1./2.)*np.linalg.norm(np.dot(X, (self.u+np.ones(len(self.u)))**self.power) - y)**2

    def gradient_u(self,X,y):
        residual = np.dot(X, (self.u+np.ones(len(self.u)))**self.power) - y
        return 1./(X.shape[0])*self.power*np.multiply(np.dot(np.transpose(X), residual),(self.u+np.ones(len(self.u)))**(self.power-1))

    def fit(self,u0,X,y):
        self.u = u0
        u_list = []
        u_list.append(self.u**self.power)
        for t in range(self.MAXIter):
            if t % int(1000) == 0:
                print('\t{:d}k iterations, Loss: {:6.6f}\r'.format(int(t/1000),self.objective(X,y)),end='')
                print('',(self.u+np.ones(len(self.u)))**self.power)
            self.u = self.u - self.eta*self.gradient_u(X,y)
            u_list.append((self.u+np.ones(len(self.u)))**self.power)
            if t % 100 == 99:
                if self.objective(X,y) < self.tol:
                    return self.u,u_list, t+1
        print('\n')
        return self.u,u_list,t+1

    def predict(self, X):
        return np.dot(X, self.u**self.power)


class power_ui_iter(object):
    def __init__(self, layer_num, eta=0.01, MAXIter=int(1e7), tol=1e-6):
        self.layer_num = layer_num
        self.eta = eta
        self.MAXIter = MAXIter
        self.tol = tol
        self.u_mul = None
        
    def objective(self,X,y):
        return (1./2.)*np.linalg.norm(np.dot(X, self.u_mul) - y)**2

    def gradient_ui(self,X,y,i):      
        u_mul_i = self.u_mul_f(i)       
        residual = np.dot(X, self.u_mul) - y
        return 1./(X.shape[0])*np.multiply(np.dot(np.transpose(X), residual),u_mul_i)
    
    def u_mul_f(self,i):
        u = np.ones(len(self.layer_u[0]))
        for ui in self.layer_u[:i]:
            u = np.multiply(u,ui)
        return u
    
    def fit(self,u0,X,y):
        self.layer_u = u0
        self.u_mul = self.u_mul_f(self.layer_num)
        print(self.u_mul,'\n')
        u_list = []
        for t in range(self.MAXIter):
            if t % int(1000) == 0:
                print('\t{:d}k iterations, Loss: {:6.6f}\r'.format(int(t/1000),self.objective(X,y)),end='')               
                print('',self.u_mul)
            
            for i in range(self.layer_num):
                self.layer_u[i] = self.layer_u[i] - self.eta*self.gradient_ui(X,y,i)
            
            self.u_mul = self.u_mul_f(self.layer_num)
            
            if t % 100 == 99:
                u_list.append(self.u_mul)
                if self.objective(X,y) < self.tol:
                    return self.layer_u,u_list, t+1
        print('\n')
        return self.layer_u,u_list,t+1

    def predict(self, X):
        return np.dot(X, self.u_mul)
    

if __name__ == "__main__":
        
    
    tol = 1e-5
    MAXIter = 5e6
    
    res_error_list = []
    
    n = 100
    k = 5 # sparsity
    d = 1000 # dimension
    sigma = 0.1 # label noise
    
    #n = 20
    #k = 5 # sparsity
    #d = 40 # dimension
    #sigma = 0.01 # label noise
    
    X,y,W_star = gen_problem(k,n,d,sigma) #,support
    
    
    #power  = 2
    #eta = 0.0001
    #power_uc = power_uc_iter(power=power, eta=eta,MAXIter=int(1e7))
    #
    #u,u_list,T = power_uc.fit(u0,X,y)
    
    ##
    tol = 1e-8
    
    eta = 0.00001
    power  = 2
    power_uc1 = power_uc_iter(power=power, eta=eta, MAXIter=int(MAXIter),tol=tol)
    
    u0 = np.ones(W_star.shape)*pow(0.0001,1/power)
    #u0[4:] = 0
    u1,u_list1,T = power_uc1.fit(u0.copy(),X.copy(),y.copy())
    u_list1 = np.array(u_list1)
    u_list1 = pd.DataFrame(u_list1)
    
    ## 
    eta = 0.00001
    power  = 3
    power_uc2 = power_uc_iter(power=power, eta=eta, MAXIter=int(MAXIter),tol=tol)
    
    u0 = np.ones(W_star.shape)*pow(0.0001,1/power)
    u2,u_list2,T = power_uc2.fit(u0.copy(),X.copy(),y.copy())
    u_list2 = np.array(u_list2)
    u_list2 = pd.DataFrame(u_list2)
    
    ## 
    eta = 0.00001
    power  = 4
    power_uc3 = power_uc_iter(power=power, eta=eta, MAXIter=int(MAXIter),tol=tol)
    
    u0 = np.ones(W_star.shape)*pow(0.0001,1/power)
    u3,u_list3,T = power_uc3.fit(u0.copy(),X.copy(),y.copy())
    u_list3 = np.array(u_list3)
    u_list3 = pd.DataFrame(u_list3)
    
    ## 
    eta = 0.00001
    power  = 5
    power_uc4 = power_uc_iter(power=power, eta=eta, MAXIter=int(MAXIter),tol=tol)
    
    u0 = np.ones(W_star.shape)*pow(0.0001,1/power)
    u4,u_list4,T = power_uc4.fit(u0.copy(),X.copy(),y.copy())
    u_list4 = np.array(u_list4)
    u_list4 = pd.DataFrame(u_list4)
    
    u_list = [u_list1,u_list2,u_list3,u_list4]
    
    
    Iter_n_plot = 20000
    
    plt.figure(figsize=(20,6),dpi=200) #
    
    for power_i in range(len([2,3,4,5])):
        plt.subplot(int('14'+str(power_i+1)))
        for i in range(d):
            
            label_i = 'power '+str(power_i+2)
            plt.title(label_i)
            plt.plot(u_list[power_i].iloc[:Iter_n_plot:500,i])
            #plt.plot(u_list1.iloc[:Iter_n_plot:100,i],label='power 2 $w_2$')
        #plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('W')
 
    plt.savefig('../fig/Imp_sparse_w.jpg')
    plt.show()    

    
    #plt.subplot(132)
    #plt.plot(u_list2.iloc[:Iter_n_plot:100,0],label='power 3 $w_1$')
    #plt.plot(u_list2.iloc[:Iter_n_plot:100,1],label='power 3 $w_2$')
    #plt.legend()
    #plt.xlabel('Iteration')
    #plt.ylabel('W')
    #
    #plt.subplot(133)
    #plt.plot(u_list3.iloc[:Iter_n_plot:100,0],label='power 4 $w_1$')
    #plt.plot(u_list3.iloc[:Iter_n_plot:100,1],label='power 4 $w_2$')
    #plt.legend()
    #plt.xlabel('Iteration')
    #plt.ylabel('W')
    

    
    
    ## Sol 2
    #X,y,W_star = gen_problem(k,n,d,sigma)
    
    #power  = 3
    #u0 = np.ones(W_star.shape[0])*0.0001
    #
    #u_init = []
    #for i in range(power):
    #    u_init.append(u0)
    #
    ##u0[4:] = 0
    #
    #eta = 0.001
    #
    #power_ui = power_ui_iter(layer_num=power, eta=eta, MAXIter=int(1e7))
    #
    #u,u_list,T = power_ui.fit(u_init,X,y)
    #
    #u_list = np.array(u_list)
    #u_list = pd.DataFrame(u_list)
    #Iter_n_plot = -9000
    #
    #plt.figure()
    #u_list.iloc[Iter_n_plot:,0].plot()
    
    #X,y,W_star = gen_problem(k,n,d,sigma)
    #
    #eta = 0.0001
    #power1  = 2
    #u0 = np.ones(W_star.shape[0])*0.0001
    #
    #u_init1 = []
    #for i in range(power1):
    #    u_init1.append(u0)
    #
    #power_uc1 = power_ui_iter(layer_num=power1, eta=eta, MAXIter=int(1e6),tol=1e-5)
    #u1,u_list1,T = power_uc1.fit(u_init1.copy(),X.copy(),y.copy())
    #
    ### 
    #eta = 0.0001
    #power2  = 3
    #u_init2 = []
    #for i in range(power2):
    #    u_init2.append(u0)
    #
    #power_uc2 = power_ui_iter(layer_num=power2, eta=eta, MAXIter=int(1e6),tol=1e-5)
    #u2,u_list2,T = power_uc2.fit(u_init2.copy(),X.copy(),y.copy())
    #
    #
    #u_list1 = np.array(u_list1)
    #u_list1 = pd.DataFrame(u_list1)
    #
    #u_list2 = np.array(u_list2)
    #u_list2 = pd.DataFrame(u_list2)
    #
    #Iter_n_plot = 500
    #
    #plt.figure(figsize=(16,6)) #
    #
    #plt.subplot(121)
    #plt.plot(u_list1.iloc[:Iter_n_plot:,0],label='power 2 $w_1$')
    #plt.plot(u_list1.iloc[:Iter_n_plot:,1],label='power 2 $w_2$')
    #plt.legend()
    #plt.xlabel('Iteration')
    #plt.ylabel('W')
    #
    #plt.subplot(122)
    #plt.plot(u_list2.iloc[:Iter_n_plot:,0],label='power 3 $w_1$')
    #plt.plot(u_list2.iloc[:Iter_n_plot:,1],label='power 3 $w_2$')
    #plt.legend()
    #plt.xlabel('Iteration')
    #plt.ylabel('W')
    #
    #plt.savefig('I_w.jpg')
    #plt.show()
    
    ## 