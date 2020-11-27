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
    #wstar[support] = 1#np.abs(np.random.randn(k))*10
    support = [0,1,2,3,4]
    wstar = np.append(np.array([1,1,1,1,0.1]),np.zeros(d-k)) #8,6,4,2,10
    
    print(wstar)
    np.random.seed(0)
    X = np.random.randn(n,d)
    y = np.dot(X,wstar) + sigma*np.random.randn(n)
    return X,y,wstar,support

class power_uc_iter(object):
    def __init__(self, power, eta=0.01, MAXIter=int(1e7), tol=1e-6):
        self.power = power
        self.eta = eta
        self.MAXIter = MAXIter
        self.tol = tol
        
    def objective(self,X,y):
        return (1./2.)*np.linalg.norm(np.dot(X, self.u**self.power) - y)**2

    def objective_r(self,wstar):
        return (1./2.)*np.linalg.norm(wstar - self.u**self.power)**2

    def gradient_u(self,X,y):
        residual = np.dot(X, self.u**self.power) - y
        return 1./(X.shape[0])*self.power*np.multiply(np.dot(np.transpose(X), residual), self.u**(self.power-1))

    def fit(self,u0,X,y,wstar):
        self.u = u0
        u_list = []
        u_list.append(self.u**self.power)
        for t in range(self.MAXIter):
            if t % int(1000) == 0:
                print('\t{:d}k iterations, Loss: {:6.6f}, Rec: {:6.6f}\r'.format(int(t/1000),self.objective(X,y),self.objective_r(wstar)),end='')
                #print('',self.u**self.power)
            self.u = self.u - self.eta*self.gradient_u(X,y)
            if t % 100 == 99:
                if self.objective_r(wstar) < self.tol:
                    return self.u,u_list, t+1,self.objective_r(wstar)
        print('\n')
        return self.u,u_list,t+1,self.objective_r(wstar)

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

    def objective_r(self,wstar):
        return (1./2.)*np.linalg.norm(wstar - self.u**self.power)**2

    def gradient_u(self,X,y):
        residual = np.dot(X, (self.u+np.ones(len(self.u)))**self.power) - y
        return 1./(X.shape[0])*self.power*np.multiply(np.dot(np.transpose(X), residual),(self.u+np.ones(len(self.u)))**(self.power-1))

    def fit(self,u0,X,y,wstar):
        self.u = u0
        u_list = []
        u_list.append(self.u**self.power)
        for t in range(self.MAXIter):
            if t % int(1000) == 0:
                print('\t{:d}k iterations, Loss: {:6.6f}, Rec: {:6.6f}\r'.format(int(t/1000),self.objective(X,y),self.objective_r(wstar)),end='')
                #print('',(self.u+np.ones(len(self.u)))**self.power)
            self.u = self.u - self.eta*self.gradient_u(X,y)
            if t % 100 == 99:
                if self.objective_r(wstar) < self.tol:
                    return self.u,u_list, t+1,self.objective_r(wstar)
        print('\n')
        return self.u,u_list,t+1,self.objective_r(wstar)

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
                if self.objective(X,y) < self.tol:
                    return self.layer_u,u_list, t+1
        print('\n')
        return self.layer_u,u_list,t+1

    def predict(self, X):
        return np.dot(X, self.u_mul)
    

def objective_res(w_res,wstar):
    return (1./2.)*np.linalg.norm(wstar - w_res)**2


def recon_error(w_res,wstar,support):
    error_all = (1./2.)*np.linalg.norm(wstar - w_res)**2
    error_unzero = (1./2.)*np.linalg.norm(wstar[support] - w_res[support])**2
    idx = [i for i in range(len(wstar)) if i not in support]
    error_zero = (1./2.)*np.linalg.norm(wstar[idx] - w_res[idx])**2
    return error_all,error_unzero,error_zero
    

if __name__ == "__main__":
    
    ## hyper
    tol = 1e-5
    MAXIter = 5e6
    
    res_error_list = []
    
    #n = 80
    n_list = list(range(20,140,20))  #[100,]
    for n in n_list:
        print(n)
        k = 5 # sparsity
        d = 100 # dimension
        sigma = 0.1 # label noise
        
        X,y,W_star,support = gen_problem(k,n,d,sigma) #,support
        
        ## 
        
        eta0 = 0.1
        
        ## OLS
        w_ols = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        res_error_ols = objective_res(w_ols,W_star.copy())
        error_all_ols, error_unzero_ols, error_zero_ols = recon_error(w_ols,W_star,support)
        
        init_scalar = 1e-8
        
    #    ##
    #    power  = 1
    #    power_uc0 = power_uc_iter(power=power, eta=eta**power, MAXIter=int(MAXIter),tol=tol)
    #    
    #    u0 = np.ones(W_star.shape)*pow(init_scalar,1/power)
    #    #u0[4:] = 0
    #    u0_,u_list0,T0,res_error0 = power_uc0.fit(u0.copy(),X.copy(),y.copy(),W_star.copy())
    #    error_all_pow1, error_unzero_pow1, error_zero_pow1 = recon_error(u0_**power,W_star,support)
        
        ##
        power  = 2
        power_uc1 = power_uc_iter(power=power, eta=0.01**power, MAXIter=int(MAXIter),tol=tol)
        
        u0 = np.ones(W_star.shape)*pow(init_scalar,1/power)
        #u0[4:] = 0
        u1,u_list1,T1,res_error1 = power_uc1.fit(u0.copy(),X.copy(),y.copy(),W_star.copy())
        error_all_pow2, error_unzero_pow2, error_zero_pow2 = recon_error(u1**power,W_star,support)
        
        ##
        power  = 3
        power_uc2 = power_uc_iter(power=power, eta=eta0**power, MAXIter=int(MAXIter),tol=tol)
        
        u0 = np.ones(W_star.shape)*pow(init_scalar,1/power)
        u2,u_list2,T2,res_error2 = power_uc2.fit(u0.copy(),X.copy(),y.copy(),W_star.copy())
        error_all_pow3, error_unzero_pow3, error_zero_pow3 = recon_error(u2**power,W_star,support)
        
        ##
        power  = 4
        power_uc3 = power_uc_iter(power=power, eta=eta0**power, MAXIter=int(MAXIter),tol=tol)
        
        u0 = np.ones(W_star.shape)*pow(0.0001,1/power)
        u3,u_list3,T3,res_error3 = power_uc3.fit(u0.copy(),X.copy(),y.copy(),W_star.copy())
        error_all_pow4, error_unzero_pow4, error_zero_pow4 = recon_error(u3**power,W_star,support)
        
        ##
        power  = 5
        power_uc4 = power_uc_iter(power=power, eta=eta0**power, MAXIter=int(MAXIter),tol=tol)
        
        u0 = np.ones(W_star.shape)*pow(0.0001,1/power)
        u4,u_list4,T4,res_error4 = power_uc3.fit(u0.copy(),X.copy(),y.copy(),W_star.copy())
        error_all_pow5, error_unzero_pow5, error_zero_pow5 = recon_error(u4**power,W_star,support)
        
        
        res_error_lasso_best = 100
        
        for alpha_i in [1e-6,1e-5,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]: #1e-8,1e-7,
            lasso = Lasso(alpha=alpha_i, fit_intercept=False, normalize=False, max_iter=MAXIter, 
                          tol=tol, random_state=0)#alpha=alpha_i, 
        
            lasso.fit(X.copy(),y.copy())
            res_error_lasso = objective_res(lasso.coef_,W_star.copy())
            if res_error_lasso_best > res_error_lasso:
                res_error_lasso_best = res_error_lasso
                coef_best = lasso.coef_
                best_alpha = alpha_i
        
        error_all_lasso, error_unzero_lasso, error_zero_lasso = recon_error(coef_best,W_star,support)
        print(coef_best)
        print(res_error_lasso_best)
        print(best_alpha)
        
        res_error_list.append([res_error_ols,res_error1,res_error2,res_error3,res_error4,
                               res_error_lasso_best])
        
    res_error_df = pd.DataFrame(res_error_list)
    res_error_df.columns = ['OLS','Power 2','Power 3','Power 4','Power 5','Lasso'] #,'Power 1'
    res_error_df.iloc[:,2:].plot()
    plt.show()
    
    plt.figure(figsize=(12,8),dpi=200)
    for i in range(1,6):
        plt.plot(res_error_df.iloc[:,i],'*--',label=res_error_df.columns[i])
    
    plt.xticks(range(len(res_error_df)),n_list,fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('# of observations',fontsize=16)
    plt.ylabel('Reconstruction error',fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig('../fig/Sparse_Reconstruction_w_error.jpg')
    plt.show()
