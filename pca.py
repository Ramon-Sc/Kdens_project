import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import argparse
from matplotlib import colors
import matplotlib.pyplot as plt



def preprocessing(pathorig,pathsynth):
    #import CSV file
    dataorig = np.genfromtxt(pathorig, delimiter=',', dtype='U')
    datasynth = np.genfromtxt(pathsynth, delimiter=',', dtype='U')

    #drop first row (featurelabels) and first column (indexes)
    dataorig=dataorig[1:,1:]
    num_orig=dataorig.shape[0]

    num_synth=datasynth.shape[0]

    data=np.row_stack((dataorig,datasynth))
    #hacky removal of leading and trailing qoute-char, if present
    data=np.char.lstrip(data,'"')
    data=np.char.rstrip(data,'"')

    #cast everything to float
    data.astype(float)

    #minmax feature scaling
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    dataorig=data[:num_orig,:]
    datasynth=data[num_orig:,:]

    #original data
    #retrieve positive and negative examples from data via boolean mask
    mask_pos = (dataorig[:, 0] == 1)
    mask_neg = (dataorig[:, 0] == 0)

    # first column is y-label
    X_pos_orig=dataorig[mask_pos,1:]
    X_neg_orig=dataorig[mask_neg,1:]

    #synth data
    #retrieve positive and negative examples from data via boolean mask
    mask_pos = (datasynth[:, 0] == 1)
    mask_neg = (datasynth[:, 0] == 0)

    X_pos_synth=datasynth[mask_pos,1:]
    print(X_pos_synth.shape)
    X_neg_synth=datasynth[mask_neg,1:]
    print(X_neg_synth.shape)

    return X_pos_orig,X_neg_orig,X_pos_synth,X_neg_synth

def main():


    parser = argparse.ArgumentParser()

    parser.add_argument("-s","--synth",help = "synths in", type=str,)
    parser.add_argument("-o","--orig",help = "orig in", type=str,)
    args=parser.parse_args()
    ### VARS
    #
    #path="/home/ramon/Masterthesis/kdens_data/testdata/gauss.csv"
    #path="/home/ramon/Masterthesis/kdens_data/wdbc_new.csv"
    #path="/home/ramon/Masterthesis/kdens_data/cervical_new.csv"
    #path="/home/ramon/Masterthesis/kdens_data/nafld_new.csv"
    path=args.orig

    dsname=path.split("/")[-1]
    # path_synths="/home/ramon/Masterthesis/Kdens_project/generated_data/densmap_augment_data/generated02_"+dsname
    path_synths=args.synth
    #drop first column = labels here synthdata is only class 1
    #synthdata=np.genfromtxt(path_synths, delimiter=',', dtype='U')[:,1:]
    ###
    X_pos_orig,X_neg_orig,X_pos_synth,X_neg_synth=preprocessing(path,path_synths)

    # X_pos=X_pos[:,:3]
    # print(X_pos.shape)
    # X_neg=X_neg[:,:3]
    # print(data_min.shape)
    # data_min=data_min[:4]
    # print(data_min.shape)
    # data_max=data_max[:4]


    X_neg_synth=X_neg_synth[:100,:]
    X_pos_synth=X_pos_synth[:100,:]


    data=np.concatenate((X_pos_orig,X_neg_orig,X_pos_synth,X_neg_synth), axis=0)

    pca = PCA(n_components=2)
    pc=pca.fit_transform(data)


    #print('shape ',X_neg.shape[0]," ",X_pos.shape[0])
    labels=np.concatenate(
    (
    np.full(X_pos_orig.shape[0],1),
    np.full(X_neg_orig.shape[0],2),
    np.full(X_pos_synth.shape[0],3),
    np.full(X_neg_synth.shape[0],4),
    )
    )

    colormap = colors.ListedColormap(['darkblue','red','blue','orange'])
    #colormap = colors.ListedColormap(['white','white','blue','white'])

    plt.scatter(pc[:,0], pc[:,1],c=labels,s=40,cmap=colormap)
    #plt.title('UMAP umap_k='+str(k_umap)+" "+str(dsname));
    plt.show()





    ### test
if __name__ == '__main__':
    main()
