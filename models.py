#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 12:31:14 2020

@author: mornitzan
"""



#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


#%%
###########################################################################
###########################################################################
###########################################################################
# Independent gene expression profiles
###########################################################################
###########################################################################
###########################################################################
#%%
#%% independent sequences (final figure)

# set number of cells and genes
num_cells = 2**10
num_genes = 500

# sample a gene expression matrix
cur_profiles_ind = np.random.randn(num_cells,num_genes);

# compute eigenvalues 
cov_mat = np.cov(cur_profiles_ind)
eigenvalues = np.linalg.eig(cov_mat)[0]

emp_ev = np.sort(eigenvalues)[::-1]
emp_ev = emp_ev/emp_ev[0]

#% corresponding MP distribution  
c=np.float(num_cells)/num_genes;
a=(1-np.sqrt(c))**2;
b=(1+np.sqrt(c))**2;
n=50; #bins
weights1 , bins = np.histogram(eigenvalues,bins=np.linspace(a,b,n))
f = weights1/ np.float(weights1.sum()) 

#% Theoretical pdf
F = np.multiply((1./(2*np.pi*bins*c)),np.sqrt(np.multiply((b-bins),(bins-a))));
F = F / np.nansum(F)
F[np.isnan(F)]=0

sampled_ev = np.random.choice(bins, p=F, size=len(emp_ev))
sampled_ev = np.sort(sampled_ev)[::-1]
sampled_ev = sampled_ev/sampled_ev[1]

#figures

fonts = 20
fonts_ticks = 15

plt.figure(figsize=(5,4))
plt.loglog(emp_ev,'-', label='empirical slope')
plt.loglog(sampled_ev,'--', label='predicted slope')
plt.xlabel('rank',fontsize=fonts)
plt.ylabel(r'$\lambda$',fontsize=fonts)
plt.ylim(10**-3,10**1)
plt.xlim(1,500)
plt.legend(fontsize=fonts_ticks)  
plt.xticks(fontsize=fonts_ticks)
plt.yticks(fontsize=fonts_ticks)
plt.show() 

plt.figure(figsize=(5,4))
plt.hist(bins[1:],weights = f,bins=bins[:-1],alpha=0.2, label='simulated data')
plt.hist(bins,weights = F,bins=bins,alpha=0.2, label='MP')
plt.legend(loc='upper right', fontsize=fonts_ticks)
plt.xlabel(r'$\lambda$', fontsize=fonts)
plt.ylabel(r'P($\lambda$)', fontsize=fonts)
plt.xticks(fontsize=fonts_ticks)
plt.yticks(fontsize=fonts_ticks)
plt.xlim(bins[1],bins[-1]*1.01)
plt.show()  


#%%
###########################################################################
###########################################################################
###########################################################################
# Lineage model
###########################################################################
###########################################################################
###########################################################################
#%%

num_genes = 500
num_mutations = 25
#%
initial_profile = np.random.randint(0,2,num_genes)*2 - 1
genes_mutated = np.random.choice(num_genes, num_mutations, replace=False)
initial_profile[genes_mutated] = -(initial_profile[genes_mutated])
cur_profiles_lin = np.copy(initial_profile)

num_bifurcations = 10
for k in range(num_bifurcations):
    cur_profiles_lin = np.vstack((cur_profiles_lin,cur_profiles_lin))
    for j in range(cur_profiles_lin.shape[0]):
        genes_mutated = np.random.choice(num_genes, num_mutations, replace=False)
        cur_profiles_lin[j,genes_mutated] = -(cur_profiles_lin[j,genes_mutated])

# compute eigenvalues 
cov_mat = np.cov(cur_profiles_lin)
ev = np.real(np.linalg.eig(cov_mat)[0])
ranked_ev = np.sort(ev)[::-1] 

q = 2
p = num_genes            
m = num_mutations    
alpha = np.e**(np.float(-2*q*m) / p*(q-1))
slope_here = -np.log(2*alpha) / np.log(2)
x_here = np.arange(1,len(ranked_ev)+1)    
computed_line = (ranked_ev[1]/(x_here[1]**slope_here))* x_here**slope_here

ranked_ev_norm = ranked_ev/ranked_ev[1]
temp_all_comp = computed_line/computed_line[1]

#% corresponding MP distribution  
num_cells = cur_profiles_lin.shape[0]
c=np.float(num_cells)/num_genes;
a=(1-np.sqrt(c))**2;
b=(1+np.sqrt(c))**2;
n=50; #bins
weights1 , bins = np.histogram(ranked_ev,bins=np.linspace(a,b,n))
f = weights1/ np.float(weights1.sum()) 

#% Theoretical pdf
F = np.multiply((1./(2*np.pi*bins*c)),np.sqrt(np.multiply((b-bins),(bins-a))));
F = F / np.nansum(F)
F[np.isnan(F)]=0

sampled_ev = np.random.choice(bins, p=F, size=len(emp_ev))
sampled_ev = np.sort(sampled_ev)[::-1]
sampled_ev = sampled_ev/sampled_ev[1]

#figures

fonts = 20
fonts_ticks = 15

plt.figure(figsize=(5,4))

plt.loglog(ranked_ev_norm, label='empirical slope')
plt.loglog(temp_all_comp,'--', label='predicted slope')
plt.xlabel('rank',fontsize=fonts)
plt.ylabel(r'$\lambda$', fontsize=fonts)
plt.legend(fontsize=fonts_ticks)  
plt.xticks(fontsize=fonts_ticks)
plt.yticks(fontsize=fonts_ticks)
plt.xlim(1,500)
plt.ylim(10**-3,10**1)
plt.show()  
       
plt.figure(figsize=(5,4))
plt.hist(bins[1:],weights = f,bins=bins[:-1],alpha=0.2, label='simulated data')
plt.hist(bins,weights = F,bins=bins,alpha=0.2, label='MP')
plt.legend(loc='upper right', fontsize=fonts_ticks)
plt.xlabel(r'$\lambda$', fontsize=fonts)
plt.ylabel(r'P($\lambda$)', fontsize=fonts)
plt.xticks(fontsize=fonts_ticks)
plt.yticks(fontsize=fonts_ticks)
plt.xlim(bins[1],bins[-1]*1.01)
plt.show()       
            


#%%
###########################################################################
###########################################################################
###########################################################################
# Regulatory interactions model
###########################################################################
###########################################################################
###########################################################################
#%%

def compute_energy(profile, interaction_mat, suggested_ind): 
    ind_here = np.where(interaction_mat[suggested_ind,:])
    x = suggested_ind
    y = ind_here[0]
    sum_here = - np.dot(interaction_mat[x,y],y) * profile[x]
    return sum_here


num_genes=500;
num_cells = 2**10
interaction_mat_binary = np.random.rand(num_genes,num_genes)
interaction_mat_binary[interaction_mat_binary < 0.1] = 1
interaction_mat_binary[interaction_mat_binary < 1] = 0
interaction_strength = np.random.rand(num_genes,num_genes)*2 - 1
interaction_mat = np.multiply(interaction_mat_binary,interaction_strength)
interaction_mat = np.triu(interaction_mat , k=1)

num_proposed_mutations = 1000
    
cur_profiles_reg = np.zeros((num_cells , num_genes))

for iter in range(num_cells):
    if (iter/100.).is_integer():
        print(iter)
    initial_profile = np.random.randint(0,2,num_genes)*2 - 1
    cur_profile = np.copy(initial_profile)
    
    for j in range(num_proposed_mutations):
        proposed_profile = np.copy(cur_profile)
        suggested_ind = np.random.choice(num_genes, 1, replace=False)
        proposed_profile[suggested_ind] = -(proposed_profile[suggested_ind])

        energy_proposed = compute_energy(proposed_profile, interaction_mat, suggested_ind)
        energy_current = compute_energy(cur_profile, interaction_mat, suggested_ind)
        delta_energy = energy_proposed - energy_current
        prob_acceptance = min(1,np.e**(-delta_energy))
        if np.random.rand()<prob_acceptance:
            cur_profile = np.copy(proposed_profile)
        
    cur_profiles_reg[iter,:] = cur_profile
    

# compute eigenvalues 
cov_mat = np.cov(cur_profiles_reg)
ev = np.real(np.linalg.eig(cov_mat)[0])
ranked_ev = np.sort(ev)[::-1] 
            
#% corresponding MP distribution     
c=np.float(num_cells)/num_genes;
a=(1-np.sqrt(c))**2;
b=(1+np.sqrt(c))**2;
n=50; #bins
weights1 , bins = np.histogram(ranked_ev,bins=np.linspace(a,b,n))
f = weights1/ np.float(weights1.sum()) 

#% Theoretical pdf
F = np.multiply((1./(2*np.pi*bins*c)),np.sqrt(np.multiply((b-bins),(bins-a))));
F = F / np.nansum(F)
F[np.isnan(F)]=0

sampled_ev = np.random.choice(bins, p=F, size=len(emp_ev))
sampled_ev = np.sort(sampled_ev)[::-1]
sampled_ev = sampled_ev/sampled_ev[1]

ranked_ev = np.sort(ranked_ev)[::-1]
ranked_ev_norm = ranked_ev/ranked_ev[1]

#figures
fonts = 20
fonts_ticks = 15

plt.figure(figsize=(5,4))

plt.loglog(ranked_ev_norm, label='empirical slope')
plt.xlabel('rank',fontsize=fonts)
plt.ylabel(r'$\lambda$', fontsize=fonts)
plt.xticks(fontsize=fonts_ticks)
plt.yticks(fontsize=fonts_ticks)
plt.xlim(1,500)
plt.ylim(10**-3,10**1)
plt.show()  
       
plt.figure(figsize=(5,4))
plt.hist(bins[1:],weights = f,bins=bins[:-1],alpha=0.2, label='simulated data')
plt.hist(bins,weights = F,bins=bins,alpha=0.2, label='MP')
plt.legend(loc='upper right', fontsize=fonts_ticks)
plt.xlabel(r'$\lambda$', fontsize=fonts)
plt.ylabel(r'P($\lambda$)', fontsize=fonts)
plt.xticks(fontsize=fonts_ticks)
plt.yticks(fontsize=fonts_ticks)
plt.xlim(bins[1],bins[-1]*1.01)
plt.show()       

