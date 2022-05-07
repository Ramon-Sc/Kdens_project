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

#smote, adasyn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

#metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import auc


#my modules
from kdens_mindensscaled import kdens_augment
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

def model_predict(test_X,clf_lr,clf_rdf,clf_svm_lin,clf_svm_rbf,clf_svm_poly):
    pred_labels_lr=clf_lr.predict(test_X)
    pred_labels_rdf=clf_rdf.predict(test_X)
    pred_labels_svm_lin=clf_svm_lin.predict(test_X)
    pred_labels_svm_rbf=clf_svm_rbf.predict(test_X)
    pred_labels_svm_poly=clf_svm_poly.predict(test_X)

    return (pred_labels_lr,pred_labels_rdf,pred_labels_svm_lin,pred_labels_svm_rbf,pred_labels_svm_poly)

def evaluate(pred_labels,test_y):
    roc_auc=roc_auc_score(test_y,pred_labels)
    precision,recall=precision_recall_curve(test_y,pred_labels)
    auc_pr=auc(recall,precision)
    f1_score=f1_score(test_y,pred_labels)
    mccoef=matthews_corrcoef(test_y,pred_labels)

    return np.array((roc_auc,auc_pr,f1_score,mccoef))

def synth_only_splitter(
    writer_queue,
    train_pos_orig,
    train_neg_orig,
    synth_pos,
    synth_neg,
    num_synths_needed,
    iter_mccv,
    code_aug,
    noise_std,
    k
    ):

    num_synth_pos=synth_pos.shape[0]
    #minmax scaling 2
    synth_synth_data=np.rowstack((synth_pos,synth_neg))
    scaled_synth_data_2,scaler_2=scaling(synth_data)

    #prepare training data
    scaled_synth_pos=scaled_synth_data_2[:num_synth_pos,:]
    scaled_synth_neg=scaled_synth_data_2[num_synth_pos:,:]

    for code_synthonly in (0,1):
        #random generator instance for choice of examples to augment or use as synthetic pos/neg train data
        r_gen=np.random.default_rng()

        if not code_synthonly:
        #CLASSIC AUGMENTATION(code=0): num of synths added to minority class = difference(num_minoritiy,num_majority)
            train_pos=np.rowstack((train_pos_orig,r_gen.choice(scaled_synth_pos,num_synths_needed)
            train_neg=train_neg_orig

        else:
        #SYNTH ONLY(code=1): num examples per class = num examples majority class of original dataset =num_neg
            train_pos=r_gen.choice(scaled_synth_pos,num_neg)
            train_neg=r_gen.choice(scaled_synth_neg,num_neg)

        train_data=np.rowstack((train_pos,train_neg))
        train_X=train_data[:,1:]
        train_y=train_data[:,0]

        #model training
        clf_lr,clf_rdf,clf_svm_lin,clf_svm_rbf,clf_svm_poly=model_train(train_X,train_y)

        #model predictions
        pred_labels_tup=model_predict(test_X)

        #model evaluation
        for code_model,pred_labels in enumerate(pred_labels_tup):
            scores=evaluate(pred_labels)
            csv_row=np.append((iter_mccv,code_synthonly,code_aug,code_model,noise_std,k),scores)
            writer_queue.put(csv_row)

def writer(writer_queue,csv_out_name):
    with open (csv_out_name,"w+") as f:
        writer= csv.writer(f)
        row=writer_queue.get()

        writer.writerow(row)

def PIPE(data,iter_mccv,num_candidates,writer_queue):

    '''
    this pipeline is executed for every CV split in num_mccv
    order in csv file:
    iter_mccv(0-999),synth_only(0,1),aug_meth(0-4),model(0-4),noise_std(float),k(int),AUCROC(float),AUCPR(float),F1(float),MMCoef(float)
    '''

    code_kdens=0
    code_gnus=1
    code_smote=2
    code_adasyn=3
    code_null=4
#1 mccv split###################################################################
    train,test=split(data)

    test_X=test[:,1:]
    test_y=test[:,0]

#2 minmax scaling (normalization)###############################################
    scaled_data_1,scaler_1=scaling(train)

# calculate class imbalance and number of synths to supplement the minority class with (in augment)
    mask_pos=(scaled_data_1[:, 0] == 1)
    train_pos_orig=scaled_data_1[mask_pos]
    train_neg_orig=scaled_data_1[np.invert(mask_pos)]
    num_pos=train_pos_orig.shape[0]
    num_neg=num_train-num_pos
    num_synths_needed=num_neg-num_pos

#3 data augmentation kdens,gnus,smote,adasyn,null model####################################

    # A) kdens with param combinations
    for noise_std in lst_noise_std:
        for k in lst_k:

            # using the same k for umap and subsequent kdens since low dim k nn are positioned according to high dim k nn
            # theres probably no point in this function having to "k" arguments but here we are..
            synth_kdens_pos,synth_kdens_neg=kdens_augment(scaled_data_1,num_candidates,k,k,)

            #rescale
            synth_kdens_pos,synth_kdens_neg=map(scaler_1.inverse_transform,[synth_kdens_pos,synth_kdens_neg])


            synth_only_splitter(
            writer_queue,
            train_pos_orig,
            train_neg_orig,
            synth_kdens_pos,
            synth_kdens_neg,
            num_synths_needed,
            iter_mccv,
            code_kdens,
            noise_std,
            k
            )



    # B) gnus with different noise levels
    for noise_std in lst_noise_std:
        synth_gnus_pos,synth_gnus_neg=gnus(scaled_data_1,noise_std,num_candidates)
        #rescale
        synth_gnus_pos,synth_gnus_neg=map(scaler_1.inverse_transform,[synth_gnus_pos,synth_gnus_neg])

        #k=0 since there is no use of a k value in gnus but i didnt want to put N.A. in csv
        k=0

        #rest of pipe stuff and "to csv" eventually
        synth_only_splitter(
        writer_queue,
        train_pos_orig,
        train_neg_orig,
        synth_gnus_pos,
        synth_gnus_neg,
        num_synths_needed,
        iter_mccv,
        code_gnus,
        noise_std,
        k
        )



    #smote and adasyn need X and y to be seperated:
    scaled_data_1_X=scaled_data_1[:,1:]
    scaled_data_1_y=scaled_data_1[:,0]
    #params
    k=5
    noise_std=0
    n_samples_new=1000
    ratio={pos:n_samples_new,neg:n_samples_new}


    # C) SMOTE
    smote=SMOTE(k_neighbors=k,ratio=ratio)
    X_smote,y_smote=smote.fit_sample(scaled_data_1_X,scaled_data_1_y)
    synth_smote=np.column_stack((y_smote,X_smote))

    #retrieve pos and neg
    mask_pos=(synth_smote[:,0] == 1)
    mask_neg=(synth_smote[:,0] == 0)
    syth_smote_pos=synth_smote[mask_pos]
    synth_smote_neg=synth_smote[mask_neg]

    #rescaling
    synth_smote_pos,synth_smote_neg=map(scaler_1.inverse_transform,[synth_smote_pos,synth_smote_neg])

    #rest of pipe stuff and "to csv" eventually
    synth_only_splitter(
    writer_queue,
    train_pos_orig,
    train_neg_orig,
    synth_smote_pos,
    synth_smote_neg,
    num_synths_needed,
    iter_mccv,
    code_smote,
    noise_std,
    k
    )



    # D) ADASYN
    adasyn=ADASYN(n_neighbors=k,ratio=ratio)
    X_adasyn,y_adasyn=adasyn.fit_sample(scaled_data_1_X,scaled_data_1_y)
    synth_adasyn=np.column_stack((y_adasyn,X_adasyn))

    #retrieve pos and neg
    mask_pos=(synth_adasyn[:,0] == 1)
    mask_neg=(synth_adasyn[:,0] == 0)
    synth_adasyn_pos=synth_adasyn[mask_pos]
    synth_adasyn_neg=synth_adasyn[mask_neg]

    synth_only_splitter(
    writer_queue,
    train_pos_orig,
    train_neg_orig,
    synth_adasyn_pos,
    synth_adasyn_neg,
    num_synths_needed,
    iter_mccv,
    code_adasyn,
    noise_std,
    k
    )



if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument("-nmccv","--num_mccv",help="number of monte carlo crossvalidation splits", type=int, default=1000,)
    parser.add_argument("-nc","--num_cand",help="number of candidates generated from parent points", type=int, default=2000,)
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

    num_candidates=args.num_cand
    #multiproessing part############################################################

    processes=[]

    write_process=multiprocessing.Process(target=write_to_csv , args=(writer_queue, csv_out_name))

    #start num_mccv processes:
    for iter_mccv in range(num_mccv):
        p=multiprocessing.Process(target=PIPE,args=(data,iter_mccv,num_candidates,writer_queue))
        p.start()
        processes.append(p)

    #start 1 write process
    write_process.start()

    for process in processes:
        process.join()

    #end write process
    write_process.join()
