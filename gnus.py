import numpy as np
from sklearn.preprocessing import MinMaxScaler
import csv
def gnus(data,std_gaussian_noise,num_candidates):

    mask_pos=(scaled_data[:, 0] == 1)
    mask_neg=(scaled_data[:, 0] == 0)

    rng=np.random.default_rng()

    synth_pos=np.column_stack(
    (
    np.ones(num_candidates),
    rng.choice(X_pos,num_candidates)+rng.normal(0,std,size=(num_candidates,num_features))
    )
    )

    synth_neg=np.column_stack(
    (
    np.zeros(num_candidates),
    rng.choice(X_neg,num_candidates)+rng.normal(0,std,size=(num_candidates,num_features))
    )
    )
    return synth_pos,synth_neg

# def gnus(data,gnus_mean_scale,gnus_std_scale,num_candidates):
#     '''
#     which type of gaussian noise ???
#     '''
#     mean_arr=np.mean(data,axis=0)[1:]
#     std_arr=np.std(data,axis=0)[1:]
#
#     rng=np.random.default_rng()
#     #todo
#     #calc variance of features
#     #x_bar = x_i.mean * 0.001
#     mean=mean_arr*gnus_mean_scale
#     std=std_arr*gnus_mean_scale
#
#     synth_pos=np.column_stack(
#     (
#     np.ones(num_candidates),
#     rng.choice(X_pos,num_candidates)+rng.normal(mean,std,size=(num_candidates,num_features))
#     )
#     )
#
#     synth_neg=np.column_stack(
#     (
#     np.zeros(num_candidates),
#     rng.choice(X_neg,num_candidates)+rng.normal(mean,std,size=(num_candidates,num_features))
#     )
#     )
#     return synth_pos,synth_neg

def main():

####VARS########################################################################
    path_input="/home/ramon/Masterthesis/kdens_data/heroin_new.csv"
    path_csv_outdir="/home/ramon/Masterthesis/Kdens_project/generated_data/GNUS/"

    #GNUS
    gnus_mean_scale=0.001
    gnus_std_scale=0.001

    num_candidates=1000
################################################################################

    X_pos,X_neg,num_pos,num_neg,num_features,scaler,mean_arr,std_arr=preprocessing(path_input)


    #writing synth sxamples to csv file#############################################
    dsname=path_input.split("/")[-1]
    dsname=dsname.split(".")[0]
    print("writing csv file to "+path_csv_outdir+str(dsname)+"_mean"+str(gnus_mean_scale)+"std"+str(gnus_std_scale))
    with open(path_csv_outdir+str(dsname)+"_mean"+str(gnus_mean_scale)+"std"+str(gnus_std_scale),"w+") as f:
        write=csv.writer(f)

        for synth in synth_pos:
            write.writerow(synth)

        for synth in synth_neg:
            write.writerow(synth)

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

    return X_pos,X_neg,num_pos,num_neg,num_features,scaler,mean_arr,std_arr

if __name__ == '__main__':
    main()
