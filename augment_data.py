import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

#todo:

# data out
# option specify num of pos out num of neg out


def main():
    ### VARS
    path="/home/ramon/kdens_data/heroin_new.csv"
    k=5

    ###
    X_pos,X_neg,data_min,data_max=preprocessing(path)

    ### test
    lst=[]
    for i in range (100):
        x_synth = gen_synth_example(X_pos,X_neg,k,data_min,data_max)
        lst.append(x_synth)

    for ex in lst:
        print(ex,"\n")

    print(np.sum(np.array(lst),axis=0)[0])
    ### end test

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
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)



    #mins and maxes used later on
    data_min=scaler.data_min_
    data_max=scaler.data_max_

    #retrieve positive and negative examples from data via boolean mask
    mask_pos = (data[:, 0] == 1)
    mask_neg = (data[:, 0] == 0)

    # first column is y-label
    X_pos=data[mask_pos,1:]
    X_neg=data[mask_neg,1:]

    return X_pos,X_neg,data_min,data_max


def gen_synth_example(X_pos,X_neg,k,data_min,data_max):

    num_features=X_pos.shape[1]
    num_ex_pos=X_pos.shape[0]
    num_ex_neg=X_neg.shape[0]

    while True:
        x_synth=np.random.rand(num_features)

        #sorted euclidean distances synthetic example to every original example:
        dist_to_pos=np.sort(
        np.sqrt(np.sum(np.square(X_pos-x_synth),axis=1))
        )

        dist_to_neg=np.sort(
        np.sqrt(np.sum(np.square(X_neg-x_synth),axis=1))
        )

        #mean dist to k nearest points
        mean_dist_to_k_pos=dist_to_pos[0:k+1].mean()
        mean_dist_to_k_neg=dist_to_neg[0:k+1].mean()

        #max possible distance between two examples: [0,0,...,0] and [1,1,...,1] = sqrt(num_features)
        max_dist=np.sqrt(num_features)

        #scaling mean dist to 0-1 scale:
        scaled_mean_dist_to_k_pos=mean_dist_to_k_pos/max_dist
        scaled_mean_dist_to_k_neg=mean_dist_to_k_neg/max_dist


        #INDEX OF MAX(TUP_PROB(PROB(CLASS=0),PROB(CLASS=1)) = CLASSLABEL

        #tup_prob=(1-scaled_mean_dist_to_k_neg,1-scaled_mean_dist_to_k_pos)

        #tup_prob alternative non linear relationship dist to prob:
        logbase=10
        tup_prob=(
        min(1,-math.log(scaled_mean_dist_to_k_neg,logbase)),
        min(1,-math.log(scaled_mean_dist_to_k_pos,logbase))
        )

        class_label=tup_prob.index(max(tup_prob))
        #print(tup_prob,class_label)
        if np.random.choice(
        [True,False],
        p=[tup_prob[class_label],1-tup_prob[class_label]]
        ):
            example=np.concatenate((np.array([class_label]),x_synth))
            example_rescaled=example*(data_max-data_min)+data_min

            return example_rescaled
            break

def create_synthetic_data():
    pass

if __name__ == '__main__':
    main()
