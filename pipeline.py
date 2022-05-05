import argparse
import csv
import numpy as np

#sklearn stuff
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit

#classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


#my modules
from densmap_augment_mindensscaled import kdens_augment
from GNUS import gnus
#0
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

    return data

#1 MCCV Split
def split(data):

    X=data[:,1:]
    y=data[:,0]

    rng=np.random.default_rng()

    #try max 1000 splits
    for i in range(1000):
        #random split
        test_size=rng.random()
        #mccv split
        rs=SchuffleSplit(n_splits=1,test_size=test_size)
        idx_train,idx_test=rs.split(X)

        #prevent train or test set being devoid of samples of one class
        if 1 not in y[idx_train] or if 0 not in y[idx_train]:
            continue

        if 1 not in y[idx_test] or if 0 not in y[idx_test]:
            continue

        if i == 999:
            print("failed to split data in train, test wothout one set having no examples of one class")
            break
        return data[idx_train],data[idx_test]


#2 MinMax scaling
def scaling(data):
    #minmax feature scaling
    #first column (label) scaled as well, uneccessarry and ugly but obviously wouldn't cause any problems since labels stay 0 and 1 after "scaling"
    scaler=MinMaxScaler()
    scaler.fit(data)
    scaled_data=scaler.transform(data)

    return scaled_data,scaler

def model_train(train_X,train_y):
    clf_lr=LogisticRegression(random_state=0).fit(train_X,train_y)
    clf_rdf=RandomForestClassifier(max_depth=2, random_state=0).fit(train_X,train_y)
    clf_svm_lin=SVC(kernel='linear').fit(train_X,train_y)
    clf_svm_rbf=SVC(kernel='rbf').fit(train_X,train_y)
    clf_svm_poly=SVC(kernel='poly').fit(train_X,train_y)

    return clf_lr,clf_rdf,clf_svm_lin,clf_svm_rbf,clf_svm_poly

def model_predict(test_X):
    pred_labels_lr=clf_lr.predict(test_X)
    pred_labels_rdf=clf_rdf.predict(test_X)
    pred_labels_svm_lin=clf_svm_lin.predict(test_X)
    pred_labels_svm_rbf=clf_svm_rbf.predict(test_X)
    pred_labels_svm_poly=clf_svm_poly.predict(test_X)

    return pred_labels_lr,pred_labels_rdf,pred_labels_svm_lin,pred_labels_svm_rbf,pred_labels_svm_poly

def evaluate():
    pass

def PIPE(data):
    'this pipeline is executed for every CV split in num_mccv'

#1 mccv split###################################################################
    train,test=split(data)

#2 minmax scaling (normalization)###############################################
    scaled_data_1,scaler_1=scaling(train)

# calculate class imbalance and number of synths to supplement the minority class with
    mask_pos=(scaled_data_1[:, 0] == 1)
    num_pos=scaled_data_1[mask_pos].shape[0]
    num_neg=num_train-num_pos
    num_synths_needed=num_neg-num_pos

#3 data augmentation kdens,gnus,smote,adasyn,null model####################################

    # A) kdens with param combinations
    for noise_std in lst_noise_std:
        for k in lst_k:

            # using the same k for umap and subsequent kdens since low dim k nn are positioned according to high dim k nn
            # theres probably no point in this function having to "k" arguments but here we are..
            synth_kdens_pos,synth_kdens_neg=kdens_augment(scaled_data_1,k,k,)

            #rescale
            synth_kdens_pos,synth_kdens_pos=map(scaler_1.inverse_transform,[synth_kdens_pos,synth_kdens_neg])

            num_synth_pos=synth_kdens_pos.shape[0]
            #minmax scaling 2
            synth_kdens_data=np.rowstack((synth_kdens_pos,synth_kdens_neg))
            scaled_kdens_data_2,scaler_2=scaling(synth_kdens_data)

            #prepare training data
            scaled_synth_kdens_pos=scaled_kdens_data_2[:num_synth_pos,:]
            scaled_synth_kdens_neg=scaled_kdens_data_2[num_synth_pos:,:]

            #random generator instance for choice of examples to augment or use as synthetic pos/neg train data
            r_gen=np.random.default_rng

            #synth only: num examples per class = num examples majority class of original dataset =num_neg
            train_pos=r_gen.choice(scaled_synth_kdens_pos,num_neg)
            train_neg=r_gen.choice(scaled_synth_kdens_neg,num_neg)



            queue.put(row_synthonly)
            ###todo file out
            queue.put(row_augented)

    # B) gnus with different noise levels
    for noise_std in lst_noise_std:
        synth_gnus_pos,synth_gnus_neg=gnus(scaled_data_1,)

        ###todo file out
        queue.put(row)




    #random choice of examples to augment or use as synthetic pos/neg train data
    r_gen=np.random.default_rng

    #synth only
    num_ex=int(num_train/2)
    pos=rgn.choice(synth_pos,size=num_ex)
    neg=rng.choice(synth_neg,size=num_ex)


    #balanced by augmentation
    #calculate class imbalance
    #assuming positve class is always minority class



#6 minmax scaling
    scaled_data_2,scaler_2=scaling(synth_data)

#7 Model training



#8 prediction


#9 Evaluation
    f1_score=f1_score(test_labels,pred_labels_lr)
    roc_auc_score=roc_auc_score

#10 to csv


def write_to_csv(writer_queue):


if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("-nmccv","--num_mccv",help="number of monte carlo crossvalidation splits", type=int, default=1000,)
    parser.add_argument("-nc","--num_cand",help="number of candidates generated from parent points", type=int, default=1000,)
    parser.add_argument("-uk","--umap_k",help="umap k", type=int, default=30,)
    parser.add_argument("-kdk","--kdens_k",help="kdens_k", type=int, default=30,)
    parser.add_argument("-gnstd","--gnoise_std",help="standard dev gaussian noise", type=int, default=0.01,)
    parser.add_argument("-i","--input",help="input data",type=str,required=True)
    parser.add_argument("-o","--outdir",help="output directory",type=str,required=True)

    args=parser.parse_args()

    path=args.input

    csv_out_name=args.outdir+dsname+"_eval.csv"

    data=preprocessing(path)

    num_mccv=args.num_mccv


    #multiproessing part############################################################

    processes=[]

    write_process=multiprocessing.Process(target=write_to_csv , args=(writer_queue, csv_out_name))

    #start num_mccv processes:
    for _ in range(num_mccv):
        p=multiprocessing.Process(target=PIPE,args=(data))
        p.start()
        processes.append(p)

    #start 1 write process
    write_process.start()

    for process in processes:
        process.join()

    #end write process
    write_process.join()
