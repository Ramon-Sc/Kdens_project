import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

#todo retransform synthetic data to original feature-scale


def main():
    path="/home/ramon/kdens_data/heroin_new.csv"

    X_pos,X_neg=preprocessing(path)


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

    #retrieve positive and negative samples from data via boolean mask
    mask_pos = (data[:, 0] == 1)
    mask_neg = (data[:, 0] == 0)

    X_pos=data[mask_pos,:]
    X_neg=data[mask_neg,:]

    return X_pos,X_neg


def gen_synth_sample():
    pass


def get_density_score(synth_sample,X,k):

    distances=np.apply_along_axis(scipy.spatial.distance.euclidean(),axis=1,args=(x_synth))


    return densityscore

def create_synthetic_data():
    pass

if __name__ == '__main__':
    main()
