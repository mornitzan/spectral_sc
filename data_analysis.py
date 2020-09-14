#########
# about #
#########

__version__ = "0.1.1"
__author__ = ["Mor Nitzan"]


###########
# imports #
###########

from sc_pl_functions import *

#%% 

# get data
dat_name = 'epidermis'; 
dge_full, gene_names, dataset_name = getdata(dat_name)
num_cells = dge_full.shape[0]

#%%
# basic preprocessing
num_var=500 #number of top variable genes
dge_full_post, gene_names_post, ind_var = mat_preprocessed(dge_full, gene_names, num_var)

#%%
# compute eigenvalues
ranked_ev = mat_to_ev(dge_full_post.T)
ranked_ev_norm = ranked_ev/ranked_ev[0]

#%%
# figure: eigenvalues vs rank
fonts_ticks = 15
fonts = 20

plot_ev_vs_rank(ranked_ev_norm, dataset_name)

#%% permutation

# permute gene expression matrix
dge_full_perm = np.copy(dge_full_post)
for i in range(num_var):
    dge_full_perm[:,i] = dge_full_perm[np.random.permutation(num_cells),i]

# compute eigenvalues
ranked_ev_perm = mat_to_ev(dge_full_perm.T)
ranked_ev_perm_norm = ranked_ev_perm/ranked_ev_perm[0]

# figure: eigenvalues vs rank of permuted data
plot_ev_vs_rank(ranked_ev_perm_norm, (dataset_name+', permuted'))


#%% corresponding MP distribution  

c = np.float(num_var)/num_cells;   

n=50;
a=(1-np.sqrt(c))**2;
b=(1+np.sqrt(c))**2;
weights1, bins1  = np.histogram(ranked_ev,bins=np.linspace(ranked_ev.min(),b*2,n))
f_last = weights1/ np.float(weights1.sum())

#% Theoretical pdf
F = np.multiply((1./(2*np.pi*bins1*c)),np.sqrt(np.maximum(np.zeros(len(bins1)),np.multiply((b-bins1),(bins1-a)))));
F /= np.nansum(F)

# figure: distribution of eigenvalues of data and MP
plt.figure(figsize=(5,4))
bins_here = (bins1[:-1]+bins1[1:])/2
plt.hist(bins_here,weights = f_last,bins=bins_here,alpha=0.2, label='data')
plt.hist(bins1,weights = F,bins=bins1,alpha=0.2, label='MP')
plt.legend(loc='upper right', fontsize=fonts)
plt.xlabel(r'$\lambda$', fontsize=fonts)
plt.ylabel(r'$P(\lambda)$', fontsize=fonts)
plt.xticks(fontsize=fonts_ticks)
plt.yticks(fontsize=fonts_ticks)
plt.xlim(0,bins1[-1]*1.01)
plt.show()  

#%% Comparing eigenvalue distributions of randomly selected cells vs neighboring cells

# construct a KNN graph
nei_num = 50 #choose number of neighbros
nbrs = NearestNeighbors(n_neighbors=nei_num).fit(dge_full_post)
distances, indices = nbrs.kneighbors(dge_full_post)

num_iter = 10 #number of iterations

# compute eigenvalues for subgroups of neighboring cells
ranked_ev_nei = np.zeros((num_iter,nei_num))
for i in range(num_iter):
    node = np.random.randint(num_cells) #select a random cell
    ind_here = indices[node,:] #retrieve the cell's neighbors
    dge_partial = dge_full_post[ind_here,:] #get the partial gene expression matrix for that group
    dge_partial = stats.zscore(dge_partial, axis=0) 
    dge_partial[np.isnan(dge_partial)] = 0
    ranked_ev = mat_to_ev(dge_partial) #compute eigenvalues
    ranked_ev_nei[i,:] = ranked_ev/ranked_ev[0]

# compute eigenvalues for subgroups of randomly selected cells
ranked_ev_rand = np.zeros((num_iter,nei_num))
for i in range(num_iter):
    ind_here = np.random.choice(num_cells, nei_num, replace=False) #choose a random group of cells
    dge_partial = dge_full_post[ind_here,:] #get the partial gene expression matrix for that group
    dge_partial = stats.zscore(dge_partial, axis=0) 
    dge_partial[np.isnan(dge_partial)] = 0
    ranked_ev = mat_to_ev(dge_partial) #compute eigenvalues
    ranked_ev_rand[i,:] = ranked_ev/ranked_ev[0]

#% figure: eigenvalue vs rank
plot_ev_vs_rank_mult(ranked_ev_rand, ranked_ev_nei, dataset_name)

#%% Generate ranked list of genes dominating the high modes of the data

num_eigenvectors = 300 #number of eigenvectors to include in the analysis

ev_ordered, v_ordered = mat_to_ev_v(dge_full_post.T)

gene_names_here = mat_to_ev_v_genes(gene_names_post)

temp = np.sum(np.abs(v_ordered)[:,:num_eigenvectors],axis=1)
top_genes = np.argsort(temp)[::-1]

tem = gene_names_here[top_genes]
my_list = tem.tolist()
output_file_name = 'output_genes/' + str(dataset_name) + str(num_eigenvectors)+'_sum.txt'
with open(output_file_name, 'w') as f:
    for item in my_list:    
        f.write("%s\n" % item)
        
        
