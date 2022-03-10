import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

def main(path,k):
    #import CSV file
    data = np.genfromtxt(path, delimiter=',', dtype='U')

    X=data[1:,2:].astype(float)
    y=data[1:,1]

    #hacky removal of leading and trailing qoute-char
    y=np.char.lstrip(y,'"')
    y=np.char.rstrip(y,'"')

    #feature scaling
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    print(X)

    densityscore_array=get_density_score(X,k)

    print(densityscore_array.size)
def get_density_score(X,k):
    distances = pdist(X, metric='euclidean')
    dist_matrix = squareform(distances)                                       #all rows columns 0 to k
    densityscore_array=1/((np.sort(dist_matrix)[:,0:k+1]).mean(axis=1))

    return densityscore_array

if __name__ == '__main__':
    path="/home/ramon/kdens_data/heroin_new.csv"
    main(path,10)
