import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import umap
import csv
'''
Data augmentation using Umaps density preserving feature "densmap"

still hacky synth_example classs assignment just commented out
this whole thing is not efficient but works for concept evaluation

'''
def main():
    #VARS
    #path="/home/ramon/Masterthesis/kdens_data/heroin_new.csv"
    path="/home/ramon/Masterthesis/kdens_data/testdata/gauss.csv"
    path_csv_outdir="/home/ramon/Masterthesis/Kdens_project/generated_data/densmap_augment_data/generated_"
    num_candidates=100000
    umap_k=30
    kdens_k=20

    X_pos,X_neg,num_pos,num_neg,data_min,data_max,MEAN_dist_pos,MEAN_dist_neg,num_features=preprocessing(path)

    candidates=np.random.rand(num_candidates,num_features)

    X_pos_transformed,X_neg_transformed,candidates_transformed=dim_reduction(
    X_pos,X_neg,num_pos,num_neg,candidates,umap_k)

    X_synth_pos,X_synth_neg=select_candidates(
    X_pos_transformed,
    X_neg_transformed,
    candidates_transformed,
    candidates,
    kdens_k,
    MEAN_dist_pos,
    MEAN_dist_neg,
    data_min,
    data_max
    )

    dsname=path.split("/")[-1]
    print("writing to csv file...")
    with open(path_csv_outdir+str(dsname),"w+") as f:
        write=csv.writer(f)

    #todo write takes forever dunno why ...fixit
        for synth_pos in X_synth_pos:
            write.writerow(synth_pos)

        #less relevant as neg is not the minority class
        #for synth_neg in X_synth_neg:
            #write.writerow(X_synth_neg)


def preprocessing(path):
    #import CSV file
    data=np.genfromtxt(path, delimiter=',', dtype='U')

    #drop first row (featurelabels) and first column (indexes)
    data=data[1:,1:]

    #hacky removal of leading and trailing qoute-char, if present
    data=np.char.lstrip(data,'"')
    data=np.char.rstrip(data,'"')

    #cast everything to float
    data.astype(float)

    #minmax feature scaling
    #first column (label) scaled as well, uneccessarry and ugly but obviously wouldn't cause any problems since labels stay 0 and 1 after "scaling"
    scaler=MinMaxScaler()
    scaler.fit(data)
    data=scaler.transform(data)

    #mins and maxes used later on
    data_min=scaler.data_min_
    data_max=scaler.data_max_

    #retrieve positive and negative examples from data via boolean mask
    mask_pos=(data[:, 0] == 1)
    mask_neg=(data[:, 0] == 0)

    # first column (y-label) dropped
    X_pos=data[mask_pos,1:]
    X_neg=data[mask_neg,1:]

    num_pos=X_pos.shape[0]
    num_neg=X_neg.shape[0]

    num_features=X_pos.shape[1]

    MEAN_dist_pos=squareform(pdist(X_pos, metric='euclidean')).mean()
    MEAN_dist_neg=squareform(pdist(X_neg, metric='euclidean')).mean()

    return X_pos,X_neg,num_pos,num_neg,data_min,data_max,MEAN_dist_pos,MEAN_dist_neg,num_features

def dim_reduction(
X_pos,
X_neg,
num_pos,
num_neg,
candidates,
umap_k
):
    print("running UMAP")
    data=np.vstack((X_pos,X_neg,candidates))

    fit=umap.UMAP(densmap=True,n_neighbors=umap_k)
    u=fit.fit_transform(data)
    return u[:num_pos],u[num_pos:num_pos+num_neg],u[num_pos+num_neg:]

def inclusion_choice(mean_dist_to_k_ex,MEAN_dist):
    if mean_dist_to_k_ex < MEAN_dist:
        scaled_mean_dist_to_k_ex=mean_dist_to_k_ex/MEAN_dist
    else:
        scaled_mean_dist_to_k_ex=1

    p=[1-scaled_mean_dist_to_k_ex,scaled_mean_dist_to_k_ex]
    #alternative: log scaled probabilites
    # logbase=100
    # p=[min(1,-math.log(scaled_mean_dist,logbase)),
    # 1-min(1,-math.log(scaled_mean_dist,logbase))]

    #include example with prob based on mean dist
    return np.random.choice([True,False],p=p)


def select_candidates(
X_pos_transformed,
X_neg_transformed,
candidates_transformed,
candidates,
kdens_k,
MEAN_dist_pos,
MEAN_dist_neg,
data_min,
data_max
):
    print("selecting candidates")
    #sorted euclidean distances synthetic example to every original example:
    dist_to_pos=np.sort(
    pairwise_distances(X_pos_transformed,candidates_transformed,metric='euclidean'),
    axis=0)

    dist_to_neg=np.sort(
    pairwise_distances(X_neg_transformed,candidates_transformed,metric='euclidean'),
    axis=0)

    #mean dist to k nearest points
    mean_dist_to_k_pos=dist_to_pos[:kdens_k,:].mean(axis=0)
    mean_dist_to_k_neg=dist_to_neg[:kdens_k,:].mean(axis=0)

    #optional set dist to max dist if dist to other classes nearest points is smaller
    # idx_losers_pos=np.argwhere(mean_dist_to_k_pos>mean_dist_to_k_neg)
    # idx_losers_neg=np.argwhere(mean_dist_to_k_pos<mean_dist_to_k_neg)
    #
    # mean_dist_to_k_pos[idx_losers_pos]=max_dist_pos
    # mean_dist_to_k_neg[idx_losers_neg]=max_dist_neg

    v_inclusion_choice=np.vectorize(inclusion_choice)

    #use booleanmask to keep samples whose mean dist to the nearest k points is smaller than respective dist in the other class
    mask_winners_pos=v_inclusion_choice(mean_dist_to_k_pos,MEAN_dist_pos)
    mask_winners_neg=v_inclusion_choice(mean_dist_to_k_neg,MEAN_dist_neg)

    #caveat: hacky implementation th sets winners_pos and _neg can theoretically intersect i.e. the same example can be assigned to both classes
    winners_pos=candidates[mask_winners_pos]
    winners_neg=candidates[mask_winners_neg]

    num_pos_synths=winners_pos.shape[0]
    num_neg_synths=winners_neg.shape[0]

    print("# pos synth examples: ",num_pos_synths)
    print("# neg synth examples: ",num_neg_synths)

    # add label and rescale
    #yes im "rescaling" labels here, see comment in preprocessing
    rescaling_factor=(data_max-data_min)+data_min
    rescaling_factor=1

    X_synth_pos=np.column_stack((np.ones(num_pos_synths),winners_pos))*rescaling_factor
    X_synth_neg=np.column_stack((np.zeros(num_neg_synths),winners_neg))*rescaling_factor

    return X_synth_pos,X_synth_neg

if __name__ == '__main__':
    main()
