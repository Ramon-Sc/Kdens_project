import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import umap
import csv
import time
import os
import argparse

'''
Data augmentation using Umaps density preserving feature "densmap"

point of reference for level of synth sample validity:
 min(dens) in original data

if dens_candidate <=  min(dens_orginal_points):
        candidate.prob_of_inclusion_in_synthdata  =  0

'''
def kdens_augment(
    scaled_data,
    num_candidates,
    umap_k,kdens_k,
    std_gaussian_noise
    ):
    #retrieve positive and negative examples from data via boolean mask
    mask_pos = (scaled_data[:, 0] == 1)
    mask_neg = (scaled_data[:, 0] == 0)

    # first column (y-label) dropped
    X_pos = scaled_data[mask_pos,1:]
    X_neg = scaled_data[mask_neg,1:]

    if X_pos.shape[0] <= umap_k or X_neg.shape[0] <= umap_k or X_pos.shape[0] <= kdens_k or X_neg.shape[0] <= kdens_k:
        print("number of examples in one class smaller than value for k nn")
        return 0

    num_pos = X_pos.shape[0]
    num_neg = X_neg.shape[0]

    num_features = X_pos.shape[1]

#candidate generation###########################################################
    rng = np.random.default_rng()

    candidates_pos = (
    rng.choice(X_pos,num_candidates)
    +rng.normal(0,std_gaussian_noise,size=(num_candidates,num_features))
    )

    candidates_neg = (
    rng.choice(X_neg,num_candidates)
    +rng.normal(0,std_gaussian_noise,size=(num_candidates,num_features))
    )

    candidates = np.row_stack((candidates_pos,candidates_neg))

#dim reduction via UMAP#########################################################
    X_pos_transformed,X_neg_transformed,candidates_transformed,MIN_dens_pos,MIN_dens_neg = dim_reduction(
    X_pos,X_neg,num_pos,num_neg,candidates,umap_k,kdens_k)

#actual upsampling##############################################################
    synth_pos,synth_neg = select_candidates(
    X_pos_transformed,
    X_neg_transformed,
    candidates_transformed,
    candidates,
    kdens_k,
    MIN_dens_pos,
    MIN_dens_neg,
    )

    return synth_pos,synth_neg

def preprocessing(path):
    #import CSV file
    data = np.genfromtxt(path, delimiter = ',', dtype = 'U')

    #drop first row (featurelabels) and first column (indexes)
    data = data[1:,1:]

    #hacky removal of leading and trailing qoute-char, if present
    data = np.char.lstrip(data,'"')
    data = np.char.rstrip(data,'"')

    #cast everything to float
    data=data.astype(float)

    #minmax feature scaling
    #first column (label) scaled as well, uneccessarry and ugly but obviously wouldn't cause any problems since labels stay 0 and 1 after "scaling"
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    return data,scaler


def dim_reduction(
    X_pos,
    X_neg,
    num_pos,
    num_neg,
    candidates,
    umap_k,
    kdens_k
    ):
    print("running UMAP...")

    data = np.vstack((X_pos,X_neg,candidates))

    fit = umap.UMAP(densmap=True,n_neighbors=umap_k)
    u = fit.fit_transform(data)
    ###


    MIN_dens_pos = np.sort(
    pairwise_distances(u[:num_pos,:], metric='euclidean')[:,-kdens_k-1:].mean(axis=0)
    )[-1]

    MIN_dens_neg = np.sort(
    pairwise_distances(u[num_pos:num_pos+num_neg,:], metric='euclidean')[:,-kdens_k-1:].mean(axis=0)
    )[-1]


    ############################################################################
    # from matplotlib import colors
    # import matplotlib.pyplot as plt
    # num_candidates_pos = int((candidates.shape[0])/2)
    # num_candidates_neg = int((candidates.shape[0])/2)
    # labels = np.concatenate(
    # (
    # np.full(X_pos.shape[0],1),
    # np.full(X_neg.shape[0],2),
    # np.full(num_candidates_pos,3),
    # np.full(num_candidates_neg,4),
    # )
    # )
    # colormap  =  colors.ListedColormap(['darkblue','red','blue','orange'])
    # plt.scatter(u[:,0], u[:,1],c = labels,s = 25,cmap = colormap)
    # plt.show()
    ############################################################################


    return u[:num_pos],u[num_pos:num_pos+num_neg],u[num_pos+num_neg:],MIN_dens_pos,MIN_dens_neg

def inclusion_choice(mean_dist_to_k_ex,MIN_dens):

    '''
    very simple linear choice function
    probably better to have a zero-mean gaussian where std_dev is chosen such that:
        gaussian(max_avg_dist_to_k_nn)  =  some multiple (<1) of max(gaussian)
    '''
    if mean_dist_to_k_ex < MIN_dens:
        scaled_mean_dist_to_k_ex = mean_dist_to_k_ex/MIN_dens
    else:
        scaled_mean_dist_to_k_ex = 1

    p = [1-scaled_mean_dist_to_k_ex,scaled_mean_dist_to_k_ex]

    return np.random.choice([True,False],p = p)


def select_candidates(
    X_pos_transformed,
    X_neg_transformed,
    candidates_transformed,
    candidates,
    kdens_k,
    MIN_dens_pos,
    MIN_dens_neg,
    ):
    print("selecting candidates...")
    #sorted euclidean distances synthetic example to every original example:
    dist_to_pos = np.sort(
    pairwise_distances(X_pos_transformed,candidates_transformed,metric='euclidean'),
    axis = 0)

    dist_to_neg = np.sort(
    pairwise_distances(X_neg_transformed,candidates_transformed,metric='euclidean'),
    axis = 0)


    #mean dist to k nearest points
    mean_dist_to_k_pos = dist_to_pos[:kdens_k,:].mean(axis = 0)
    mean_dist_to_k_neg = dist_to_neg[:kdens_k,:].mean(axis = 0)

    #optional: set dist to max dist if dist to other classes nearest points is smaller
    # idx_losers_pos = np.argwhere(mean_dist_to_k_pos>mean_dist_to_k_neg)
    # idx_losers_neg = np.argwhere(mean_dist_to_k_pos<mean_dist_to_k_neg)
    #
    # mean_dist_to_k_pos[idx_losers_pos] = MIN_dens_pos
    # mean_dist_to_k_neg[idx_losers_neg] = MIN_dens_neg

    v_inclusion_choice = np.vectorize(inclusion_choice)

    #boolean mask with values from inclusion choice - True for candidates whose density is "high enough"
    mask_winners_pos = v_inclusion_choice(mean_dist_to_k_pos,MIN_dens_pos)
    mask_winners_neg = v_inclusion_choice(mean_dist_to_k_neg,MIN_dens_neg)

    #caveat: hacky implementation: the sets "winners_pos" and "winners_neg" can theoretically intersect i.e. the same example can be assigned to both classes
    winners_pos = candidates[mask_winners_pos]
    winners_neg = candidates[mask_winners_neg]

    num_pos_synths = winners_pos.shape[0]
    num_neg_synths = winners_neg.shape[0]

    print("# pos synth examples: ",num_pos_synths)
    print("# neg synth examples: ",num_neg_synths)

    synth_pos = np.column_stack((np.ones(num_pos_synths),winners_pos))
    synth_neg = np.column_stack((np.zeros(num_neg_synths),winners_neg))

    return synth_pos,synth_neg

if __name__ == '__main__':

    #runtime
    st = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("-nc","--num_cand",help = "number of candidates generated from parent points", type=int, default=1000,)
    parser.add_argument("-uk","--umap_k",help="umap k", type=int, default=30,)
    parser.add_argument("-kdk","--kdens_k",help="kdens_k", type=int, default=30,)
    parser.add_argument("-gnstd","--gnoise_std",help="standard dev gaussian noise", type=int, default=0.01,)
    parser.add_argument("-i","--input",help="input data",type=str,required=True)

    args = parser.parse_args()

##VARS##########################################################################
    path = args.input

    #number of samples drawn from each class
    num_candidates = args.num_cand

    #would it make more sense to make theses two always be the same value, since umaps k is  =  =  densmaps k?
    umap_k = args.umap_k
    kdens_k = args.kdens_k

    std_gaussian_noise = args.gnoise_std

    path_csv_outdir = os.getcwd()

#preprocessing##################################################################
    scaled_data,scaler = preprocessing(path)

#kdens_augmentation#############################################################
    synth_pos,synth_neg = kdens_augment(scaled_data,num_candidates,umap_k,kdens_k,std_gaussian_noise)

    synth_pos = scaler.inverse_transform(synth_pos)
    synth_neg = scaler.inverse_transform(synth_neg)

#writing synth examples to csv file#############################################
    dsname = path.split("/")[-1]
    dsname = dsname.split(".")[0]
    print("writing csv file to: ",path_csv_outdir)
    with open(
    path_csv_outdir
    +"/"
    +"syn_"
    +str(dsname)
    +"_"+str(num_candidates)
    +"_"+str(umap_k)
    +"_"+str(kdens_k)
    +"_"+str(std_gaussian_noise)
    +".csv"
    ,"w+") as f:

        write = csv.writer(f)

        for synth in synth_pos:
            write.writerow(synth)

        for synth in synth_neg:
            write.writerow(synth)

################################################################################
    #runtime
    et = time.time()
    elapsed_time = et-st
    print('Execution time: ',elapsed_time,'seconds')
