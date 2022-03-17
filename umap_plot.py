import umap
from augment_data import preprocessing
from augment_data import gen_synth_example
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

def main():
    ### VARS
    #path="/home/ramon/kdens_data/testdata/gauss.csv"
    path="/home/ramon/kdens_data/heroin_new.csv"
    k=25

    ###
    X_pos,X_neg,data_min,data_max=preprocessing(path)


    # X_pos=X_pos[:,:3]
    # print(X_pos.shape)
    # X_neg=X_neg[:,:3]
    # print(data_min.shape)
    # data_min=data_min[:4]
    # print(data_min.shape)
    # data_max=data_max[:4]

    ###
    p=150
    countrounds=0
    n=150
    lst=[]
    while len(lst)<p:
        x_synth = gen_synth_example(X_pos,X_neg,k,data_min,data_max)
        countrounds+=1
        if x_synth[0]==1:
            lst.append(x_synth)

    while len(lst)<p+n:
        x_synth = gen_synth_example(X_pos,X_neg,k,data_min,data_max)
        if x_synth[0]==0:
            lst.append(x_synth)


    synthdata=np.array(lst)
    print(synthdata.shape)
    print(data_min)
    print(data_max)
    print('minarr: \n',np.amin(synthdata,axis=0),'\n')
    print('maxarr: \n',np.amax(synthdata,axis=0),'\n')
    #X_neg=X_neg[0:98,:]
    k_umap=15

    data_max=data_max[1:]
    data_min=data_min[1:]

    print(data_max)
    print(data_min)

    X_pos=X_pos*(data_max-data_min)+data_min
    X_neg=X_neg*(data_max-data_min)+data_min

    data=np.concatenate((X_pos, X_neg,synthdata[:,1:]), axis=0)

    fit = umap.UMAP(n_neighbors=k_umap)
    u = fit.fit_transform(data)


    print(X_neg.shape[0]," ",X_pos.shape[0])
    labels=np.concatenate(
    (
    np.full(X_pos.shape[0],1),
    np.full(X_neg.shape[0],2),
    np.full(p,3),
    np.full(n,4),
    )
    )
    colormap = colors.ListedColormap(['darkblue','red','blue','orange',])

    plt.scatter(u[:,0], u[:,1],c=labels,s=40,cmap=colormap)
    plt.title('UMAP umap_k='+str(k_umap)+" "+"kdens_k="+str(k));
    plt.show()

    ### test
if __name__ == '__main__':
    main()
