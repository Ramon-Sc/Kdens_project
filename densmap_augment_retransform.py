import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import umap
'''
Data augmentation using Umaps density preserving feature "densmap"
'''
def main():
    #VARS
    path="/home/ramon/kdens_data/cervical_new.csv"
    #path="/home/ramon/kdens_data/testdata/gauss.csv"

    umap_k=30
    kdens_k=1
    num_syn_pos=20
    num_syn_neg=20

    #preprocessing
    X_pos,X_neg,num_pos,num_neg,data_min,data_max,max_dist_pos,max_dist_neg,num_features=preprocessing(path)


    #UMAP DIM REDUCTION
    X_pos_transformed,X_neg_transformed,mapper,x_low,x_high,y_low,y_high=dim_reduction(
    X_pos,X_neg,num_pos,num_neg,umap_k
    )

    #example generation
    synth_examples=gen_examples(
    X_pos_transformed,X_neg_transformed,
    kdens_k,
    max_dist_pos,max_dist_neg,
    num_syn_pos,num_syn_neg,
    data_max,data_min,
    mapper,
    x_low,x_high,
    y_low,y_high
    )

def preprocessing(path):
    #import CSV file
    data = np.genfromtxt(path, delimiter=',', dtype='U')

    #drop first row (featurelabels) and first column (indexes)
    data=data[1:,1:]

    #hacky removal of leading and trailing qoute-char, if present
    data=np.char.lstrip(data,'"')
    data=np.char.rstrip(data,'"')

    #cast everything to float
    data.astype(float)

    #minmax feature scaling
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

    max_dist_pos=np.amax(squareform(pdist(X_pos, metric='euclidean')))
    max_dist_neg=np.amax(squareform(pdist(X_neg, metric='euclidean')))

    return X_pos,X_neg,num_pos,num_neg,data_min,data_max,max_dist_pos,max_dist_neg,num_features

def dim_reduction(X_pos,X_neg,num_pos,num_neg,umap_k):

    data=np.vstack((X_pos,X_neg))
    mapper=umap.UMAP(densmap=True,n_neighbors=umap_k)
    u=mapper.fit_transform(data)

    x_low=min(u[:,0])
    x_high=max(u[:,0])
    y_low=min(u[:,1])
    y_high=max(u[:,1])

    return u[:num_pos],u[num_pos:num_pos+num_neg],mapper,x_low,x_high,y_low,y_high

def inclusion_choice(mean_dist_to_k_ex,max_dist):
    if mean_dist_to_k_ex < max_dist:
        scaled_mean_dist_to_k_ex=mean_dist_to_k_ex/max_dist
    else:
        scaled_mean_dist_to_k_ex=1

    p=[1-scaled_mean_dist_to_k_ex,scaled_mean_dist_to_k_ex]

    #alternative: log scaled probabilites
    # logbase=100
    # p=[min(1,-math.log(scaled_mean_dist,logbase)),
    # 1-min(1,-math.log(scaled_mean_dist,logbase))]

    #include example with prob based on mean dist
    return np.random.choice([True,False],p=p)

def gen_examples(
X_pos_transformed,X_neg_transformed,
kdens_k,
max_dist_pos,max_dist_neg,
num_syn_pos,num_syn_neg,
data_max,data_min,
mapper,
x_low,x_high,
y_low,y_high):

    #list filled with 0-1 scaled examples in R²
    lst=[]

    #Loop for positive examples
    for i in range(num_syn_pos):
        x_synth=gen_synth_example(X_pos_transformed,1,kdens_k,max_dist_pos,x_low,x_high,y_low,y_high)
        lst.append(x_synth)

    #Loop for negative examples
    for i in range(num_syn_neg):
        x_synth=gen_synth_example(X_neg_transformed,0,kdens_k,max_dist_neg,x_low,x_high,y_low,y_high)
        lst.append(x_synth)

    #inverse transformation back into original featurespace
    inv_transformed=mapper.inverse_transform(np.array(lst))
    #rescaling (first column (classlabel) not to be rescaled)
    examples=np.array(inv_transformed[:,1:])*(data_max-data_min)+data_min

    return examples

def gen_synth_example(
X_class,
class_label,
k,
max_dist,
x_low,x_high,
y_low,y_high):
    '''
    generate samples in R² space based on umap transformation of the original feature space
    '''

    num_features=X_class.shape[1]

    #num_ex=X_class.shape[0]

    while True:
        # vector of length(#features) with random elements within min-max of umaos xs and ys
        x_synth=(
        np.array((
        np.random.default_rng().uniform(low=x_low, high=x_high, size=None),
        np.random.default_rng().uniform(low=y_low, high=y_high, size=None)
            ))
        )
        print(".")

        #sorted euclidean distances synthetic example to every original example:
        #maybe change manual calc to some fancy librarys implementation later
        dist_to_class=np.sort(
        np.sqrt(np.sum(np.square(X_class-x_synth),axis=1))
        )

        #mean dist to k nearest points
        mean_dist_to_k_ex=dist_to_class[0:k].mean()

        #decide wether to include synth_example based on prob derived from mean_dist
        if inclusion_choice(mean_dist_to_k_ex,max_dist):
            example=np.concatenate((np.array([class_label]),x_synth))

            return example
            break

if __name__ == '__main__':
    main()
