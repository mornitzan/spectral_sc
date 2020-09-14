from __future__ import print_function

#########
# about #
#########

__version__ = "0.1.1"
__author__ = ["Mor Nitzan"]


###########
# imports #
###########

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors

#############
# functions #
#############

def mat_preprocessed(dge_full, gene_names, num_var):
    dge_full_post = (dge_full.T.astype(float) / np.nansum(dge_full,axis=1)).T
    var_genes = np.argsort(np.divide(np.nanvar(dge_full_post.T,axis=1),np.nanmean(dge_full_post.T,axis=1)+0.0001))
    ind_var = var_genes[-num_var:]
    dge_full_post = dge_full_post[:,ind_var]
    dge_full_post = stats.zscore(dge_full_post, axis=0) 
    gene_names_post = gene_names[ind_var]
    return dge_full_post, gene_names_post, ind_var

def mat_to_ev(dge):
    cov_mat = np.cov(dge)
    ev = np.real(np.linalg.eig(cov_mat)[0])
    ranked_ev = np.sort(ev)[::-1]  
    return ranked_ev

def mat_to_ev_v(dge):
    cov_mat = np.cov(dge)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
    order_ev = np.flip(np.argsort(eigenvalues),axis=0)
    v_ordered = eigenvectors[:,order_ev]
    ev_ordered = eigenvalues[order_ev]
    return ev_ordered, v_ordered

def mat_to_ev_v_genes(gene_names_post):
    try:
        gene_names_post = [i.strip('\"') for i in gene_names_post]
    except:
        1
    gene_names_here = np.array(gene_names_post)        
    gene_names_here = np.array([x.lower() for x in gene_names_here.tolist()])
    return gene_names_here

def v_to_top_genes(v_ordered, num_eigenvectors):    
    temp = np.sum(np.abs(v_ordered)[:,0:num_eigenvectors],axis=1)
    top_genes = np.argsort(temp)[::-1]
    return top_genes

def plot_ev_vs_rank(ranked_ev, dataset_name):
    plt.figure(figsize=(5,4))
    plt.loglog(range(1,len(ranked_ev)+1),ranked_ev)
    plt.xlabel('rank', fontsize=fonts)
    plt.ylabel(r'$\lambda$', fontsize=fonts)
#    plt.title(dataset_name, fontsize=fonts)
    plt.xlim(1,len(ranked_ev)); 
    plt.ylim(10**-4,1); 
    plt.yticks(fontsize=fonts_ticks)
    plt.xticks(fontsize=fonts_ticks)
    plt.show() 

def plot_ev_vs_rank_mult(ranked_ev_rand, ranked_ev_nei, dataset_name):
    nei_num = ranked_ev_rand.shape[1]
    plt.figure(figsize=(5,4))
    plt.loglog(range(1,nei_num+1),ranked_ev_rand[0,0:].T,color='black',label='random selection')
    plt.loglog(range(1,nei_num+1),ranked_ev_rand[1:,0:].T,color='black')
    plt.loglog(range(1,nei_num+1),ranked_ev_nei[0,0:].T,color='red',label = 'neighbors')
    plt.loglog(range(1,nei_num+1),ranked_ev_nei[1:,0:].T,color='red')
    plt.xlabel('rank', fontsize=fonts)
    plt.ylabel(r'$\lambda$', fontsize=fonts)
    plt.title(str(dataset_name), fontsize=fonts)
    plt.legend(fontsize=fonts_ticks)
    plt.xlim(1,nei_num); 
    plt.ylim(10**-3,1); 
    plt.yticks(fontsize=fonts_ticks)
    plt.xticks(fontsize=fonts_ticks)
    plt.show() 

def getdata(dat_name):
    if dat_name == 'epidermis':
        file_name = 'spectral_sc/datasets/GSE67602_Joost_et_al_expression.txt'
        temp = np.loadtxt(file_name,dtype='str')    
        dge_full = temp[1:,:]; dge_full = dge_full[:,1:]
        dge_full = dge_full.T.astype(float)
        gene_names = temp[:,0]; gene_names = gene_names[1:]
        dataset_name = 'epidermis'
    
    return dge_full, gene_names, dataset_name

def compute_energy(profile, interaction_mat, suggested_ind): 
    ind_here = np.where(interaction_mat[suggested_ind,:])
    x = suggested_ind
    y = ind_here[0]
    sum_here = - np.dot(interaction_mat[x,y],y) * profile[x]
    return sum_here

########
# main #
########
    
if __name__ == '__main__':
    pass



