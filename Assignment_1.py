#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.stats import norm
from random import sample
import math


# In[3]:


# import data
AB_test_df = pd.read_csv('AB_test_data.csv')


# ### Pre-test

# In[5]:


A_df = AB_test_df.loc[AB_test_df.Variant=='A']
A_df_2 = A_df.loc[A_df.date>='2020-01-01']
A_df_1 = A_df.loc[A_df.date<'2020-01-01']

p_a_1 = np.sum(A_df_1.purchase_TF)/len(A_df_1)
p_a_2 = np.sum(A_df_2.purchase_TF)/len(A_df_2)
alpha = 0.1

z_stats = (p_a_2-p_a_1)/np.sqrt(p_a_1*(1-p_a_1)/len(A_df_2))
z_alpha = norm.ppf(1-alpha/2)

if z_stats<z_alpha:
    print("Accept Null Hypothesis")
else:
    print("Reject Null Hypothesis")


# ### 1st question

# Alternative B improved conversion rates over alternative A  
# 
# $H_{0}$: conversion rate of B <= conversion rate of A  
# $H_{1}$: conversion rate of B > conversion rate of A  
# 
# alpha = 0.05  
# t_alpha = 1.9600071176773137  
# t_stats = 8.692151285198767  
# t_stats > t_alpha, so the we reject the $H_{0}$

# In[13]:


A_df = AB_test_df.loc[AB_test_df.Variant=='A']
B_df = AB_test_df.loc[AB_test_df.Variant=='B']
N_a = len(A_df)
N_b = len(B_df)

p_a = np.sum(A_df.purchase_TF)/len(A_df)
p_b = np.sum(B_df.purchase_TF)/len(B_df)


# In[14]:


p_a


# In[15]:


# alpha = 0.05
z_alpha =norm.ppf(0.95)


# In[16]:


print(z_alpha)


# In[17]:


# H0: p_b<=p_a
# H1: p_b>p_a
z_stats = (p_b-p_a)/np.sqrt(p_a*(1-p_a)/N_b)


# In[18]:


print(z_stats)


# In[19]:


np.absolute(z_stats)>np.absolute(z_alpha)


# ### 2nd Question Yao/Ray way

# Optimal Size: 1158

# In[22]:


B_population = list(B_df.purchase_TF)


# In[23]:


# set the coefficient
alpha =0.05
confidence = 1-alpha/2
power = 0.8
difference = p_b-p_a # the minimum detecable difference

# calculate the minimum n
z_alpha = norm.ppf(confidence)
z_beta = norm.ppf(power)
p_bar = (p_a+p_b)/2
n_star = (z_alpha*np.sqrt(2*p_bar*(1-p_bar))+z_beta*np.sqrt(p_a*(1-p_a)+p_b*(1-p_b)))**2/difference**2
n_star = math.ceil(n_star)


# In[24]:


n_star


# In[25]:


# define function for simulation:
def AB_test_with_minimum():
    B_sample = sample(B_population,n_star)
    N_b = len(B_sample)
    p_b = np.sum(B_sample)/len(B_sample)
    z_stats  =  np.absolute(p_a-p_b)/np.sqrt(p_a*(1-p_a)/N_b)
    
    if z_stats>z_alpha:
        return 1,z_stats
    else:
        return 0,z_stats


# 10 times: reject H0 in all 10 times

# In[27]:


outcomes = [AB_test_with_minimum() for i in range(1,11)]
outcomes


# ### 2nd Question Nuja way

# In[29]:


def get_power(n, p1, p2, cl):
    alpha = 1 - cl
    qu = norm.ppf(1 - alpha/2)
    diff = abs(p2-p1)
    bp = (p1+p2) / 2
    
    v1 = p1 * (1-p1)
    v2 = p2 * (1-p2)
    bv = bp * (1-bp)
    
    power_part_one = norm.cdf((n**0.5 * diff - qu * (2 * bv)**0.5) / (v1+v2) ** 0.5)
    power_part_two = 1 - norm.cdf((n**0.5 * diff + qu * (2 * bv)**0.5) / (v1+v2) ** 0.5)
    
    power = power_part_one + power_part_two
    
    return (power)


# In[30]:


# here we create a get optimal sample size function
def get_sample_size(power, p1, p2, cl, max_n=1000000):
    n = 1 
    while n <= max_n:
        tmp_power = get_power(n, p1, p2, cl)
        if tmp_power >= power: 
            return n 
        else: 
            n = n + 1
    return "Increase Max N Value"


# In[31]:


get_sample_size(0.8, p_a,p_b,0.95)


# ### 3rd Question

# In all of the 10 times, we are able to stop the test prior to using the full samples.  
# Average iteration time is around 360 

# In[33]:


ln_A = np.log(1/0.05)
ln_B = np.log(0.2)
ln_1_xi = np.log(p_b/p_a)
ln_0_xi = np.log((1-p_b)/(1-p_a))


# In[34]:


# define the function for the number
def SPRT():
    B_sample = sample(B_population,n_star)
    ln_lamba = 0
    i=0
    for record in B_sample:
        if ln_lamba<ln_A and ln_lamba>ln_B:
            if record==0:
                ln_lamba=ln_lamba+ln_0_xi
            else:
                ln_lamba=ln_lamba+ln_1_xi
            i+=1
        elif ln_lamba>=ln_A:
            return(i,'accept H1')
            break
        else:
            return(i,'accept H0')
            break


# In[35]:


sprt_result = [SPRT() for i in range(1,11)]
sprt_sample_result =  [record[0] for record  in sprt_result]
sprt_test_result =  [record[1] for record in sprt_result]
np.mean(sprt_sample_result)


# In[36]:


sprt_test_result

