# ==============================================================================
#  Copyright (c) 2024. Imen Ben Amor
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from preprocessing.xvectorParser import readVectors
import random
import itertools
from preprocessing.preprocessing import *
from preprocessing.load_train_param import *
#BA-LR V1
from LR_framework.DNA-inspired_LR import *
from LR_framework.llr_prediction import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import os
import argparse

def logging_config(logname):

    logging.basicConfig(level=logging.INFO,filename=os.path.join(PATH_LOGS,logname),filemode='w+', format='%(asctime)s - %(name)s -%(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    return logging.getLogger('logger')

def todelete(df,f):
    todelete=[]
    for c in df[f]:
        if len(df[df[c]!=0])==0:
            todelete.append(c)
    df=df.drop(todelete, axis=1)
    return df
def create_trials(list_utterances):
    tar_device1=[]
    non_device1=[]
    for couple in list(itertools.combinations(list_utterances, 2)):
        if couple[0].split("-")[0]==couple[1].split("-")[0]:
            tar_device1.append(couple)
        else:
            non_device1.append(couple)
    return tar_device1,non_device1
def utterance_spk(nb_utt_spk,utterances,symb="-"):
    utt_spk = {}
    for u in utterances:
        if u.split(symb)[0] not in utt_spk.keys():
            utt_spk[u.split(symb)[0]]=[u]
        else:
            utt_spk[u.split(symb)[0]].append(u)
    return utt_spk
if __name__ == "__main__":
    # Arguments
    PATH_LOGS = "results_v1v2_device"

    parse = argparse.ArgumentParser()
    parse.add_argument("--path",
                       default="/local_disk/clytie/ibenamor/phd_experiments/scripts/NFI_calibration/xvectors.txt",
                       type=str)
    parse.add_argument("--device_session1",
                       default="1-1",
                       type=str)
    parse.add_argument("--device_session2",
                       default="2-1",
                       type=str)
    args = parse.parse_args()
    logging_config(f"logfile_{args.device_session1}-{args.device_session2}")


typ_path="/local_disk/clytie/ibenamor/phd_experiments/data/train_param/typ_clean.txt"
dout_path="/local_disk/clytie/ibenamor/phd_experiments/data/train_param/dout_clean.txt"

utterances, vectors = readVectors(args.path)
logging.info(vectors.shape)
xvectors = np.array(vectors).astype('float64')
BA = ['BA' + str(i) for i in range(xvectors.shape[1])]
df = pd.DataFrame(xvectors, columns=BA)
df=df.round(decimals = 4)
df=todelete(df,BA)

utt_per_spk, loc_list = number_utterances(utterances)
utt_spk=utterance_spk(utt_per_spk,utterances,symb='-')
df["spk_id"]=loc_list
df["utterances"]=utterances

typ_va_test,dropout_test=load_filter_soft(typ_path,dout_path)
BA_test=list(typ_va_test.keys())
utt={}
for (idx,row),u in zip(df[BA_test].iterrows(),df["utterances"]):
    utt[u]=dict(row)

#using the one version of BA-LR_framework framework
nb_attributes_fold={}
EER_fold={}
EER_global_fold={}
cllr_cal_global_fold={}
cllr_cal_fold={}
data_fold={}
data_global_fold={}

for i in range(15):
    logging.info(f"*********************Fold {i}***************************")
    logging.info("select Dev and test speakers")
    #==================================================================
    list_spk_train=random.sample(list(set(loc_list)),150)
    temp_train=[]
    for u in df[df["spk_id"].isin(list_spk_train)]["utterances"]:
        if u.endswith(args.device_session1) or u.endswith(args.device_session2):
            temp_train.append(u)
    list_spk_test=set(loc_list).symmetric_difference(set(list_spk_train))
    temp_test=[]
    for u in df[df["spk_id"].isin(list_spk_test)]["utterances"]:
        if u.endswith(args.device_session1) or u.endswith(args.device_session2):
            temp_test.append(u)
    #==================================================================
    logging.info("create trials")
    tar_test,nontest= create_trials(temp_test)
    tar_train,nontrain= create_trials(temp_train)
    logging.info(f"==========Before Calibration...==================")
    logging.info("Dev EER before calibration")
    non_train = random.sample(nontrain,min(30000,len(nontrain)))
    logging.info(f"=====train={len(tar_train)},{len(non_train)}=====")

    LLR_target,LLR_t,LLR_non,LLR_n,list_eer,list_cllr_min,list_cllr_act,list_Din,partial_LLRstrain=LR_framework(dropout_test,typ_va_test,utt,tar_train,non_train,[0.12])
    scores_train=pd.DataFrame(np.array(partial_LLRstrain))
    scores_train_glob = pd.DataFrame(np.array(LLR_target+LLR_non))
    y_train = [1]*len(tar_train)+[0]*len(non_train)

    logging.info("Test EER before calibration")
    non_test = random.sample(nontest,min(30000,len(nontest)))
    logging.info(f"=====test={len(tar_test)},{len(non_test)}======")
    LLR_target,LLR_t,LLR_non,LLR_n,list_eer,list_cllr_min,list_cllr_act,list_Din,partial_LLRstest=LR_framework(dropout_test,typ_va_test,utt,tar_test,non_test,[0.12])

    cllr_mini, cllr_act, eer, cal_tar, cal_non = Cllr_min(LLR_target, LLR_non)
    logging.info(f"============> $$$$$$ Old Cllrmin/cllract= {cllr_mini}/{cllr_act}$$$$$$")
    eer = eer * 100
    logging.info(f"============> $$$$$$ OLD TEST EER={eer} $$$$$$")

    scores_test=pd.DataFrame(np.array(partial_LLRstest))
    scores_test_glob = pd.DataFrame(np.array(LLR_target+LLR_non))
    y_test = [1]*len(tar_test)+[0]*len(non_test)
    logging.info(f"==========Start Calibration...==================")
    logging.info(f"-------------------Global Calibration...----------------------")
    model = LogisticRegression(class_weight="balanced")
    model.fit(np.array(scores_train_glob).reshape(-1, 1), y_train)
    logging.info(f"train accuracy={model.score(np.array(scores_train_glob).reshape(-1, 1), y_train)}")
    logging.info(f"test accuracy={model.score(np.array(scores_test_glob).reshape(-1, 1), y_test)}")
    weight = model.coef_
    intercept = model.intercept_
    logistic_scores = []
    for v in LLR_target+LLR_non:
        s = v * weight[0] + intercept[0]
        logistic_scores.append(s)
    logistic_scores = [i[0] for i in logistic_scores]
    logging.info(f"len of logistic global scores={len(scores_test_glob)}")
    llr_target = logistic_scores[:len(tar_test)]
    llr_non = logistic_scores[len(tar_test):]
    cllr_mini, cllr_act, eer, cal_tar, cal_non = Cllr_min(llr_target, llr_non)
    logging.info(f"====================> Global Calibrated Cllrmin/Cllract={cllr_mini}/{cllr_act}")
    eer = eer * 100
    logging.info(f"====================> Global Calibrated test EER={eer}")
    cllr_cal_global_fold[i] = cllr_act-cllr_mini
    EER_global_fold[i] = eer

    logging.info(f"-------------------Fusion & Calibration...----------------------")
    logging.info("Standardization...")
    #normalization
    scale= StandardScaler()
    # standardization of dependent variables
    scaled_data = scale.fit_transform(scores_train)
    scaled_train=pd.DataFrame(scaled_data,columns=BA_test)

    scaled_data = scale.transform(scores_test)
    scaled_test=pd.DataFrame(scaled_data,columns=BA_test)

    EER=[]
    cllr_cal = []
    nb_attributes=[]

    for c in [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.01,0.03,0.05,0.07,0.1,0.5,1,1.5,5,10]:
        logging.info(f"C={c}")
        model = LogisticRegression(C= c, penalty= "l1",solver="liblinear",class_weight="balanced")
        model.fit(scaled_train,y_train)
        logging.info(f"Number of attributes:{np.count_nonzero(model.coef_)}")
        y_pred = model.predict(scaled_test)
        logging.info(f"train accuracy={model.score(scaled_train,y_train)}")
        logging.info(f"test accuracy={model.score(scaled_test, y_test)}")
        weights = model.coef_
        intercept = model.intercept_
        logistic_scores=[]
        for (idx, row) in scaled_test.iterrows():
            s=0
            for v,w in zip(row,weights[0]):
                s+=v*w
            s+= intercept[0]
            logistic_scores.append(s)
        logging.info(f"len of logistic scores={len(scaled_test)} Vs. normalized scores={len(logistic_scores)}")
        llr_target=logistic_scores[:len(tar_test)]
        llr_non=logistic_scores[len(tar_test):]
        cllr_mini,cllr_act,eer,cal_tar,cal_non=Cllr_min(llr_target,llr_non)
        logging.info(f"====================> Calibrated Cllrmin/Cllract={cllr_mini}/{cllr_act}")
        eer=eer*100
        logging.info(f"====================> Calibrated test EER={eer}")
        nb_attributes.append(np.count_nonzero(model.coef_))
        EER.append(eer)
        cllr_cal.append((cllr_act-cllr_mini))
    nb_attributes_fold[i]=nb_attributes
    EER_fold[i]=EER
    data_fold[i]={"train":scaled_train,"test":scaled_test}
    data_global_fold[i] = {"train": scores_train_glob, "test": scores_test_glob}
    cllr_cal_fold[i]=cllr_cal

f = open(f"{PATH_LOGS}/data_fold_{args.device_session1}-{args.device_session2}.pkl","wb")
pickle.dump(data_fold,f)
f.close()
f = open(f"{PATH_LOGS}/data_global_fold_{args.device_session1}-{args.device_session2}.pkl","wb")
pickle.dump(data_global_fold,f)
f.close()
f = open(f"{PATH_LOGS}/EER_fold_{args.device_session1}-{args.device_session2}.pkl","wb")
pickle.dump(EER_fold,f)
f.close()
f = open(f"{PATH_LOGS}/EER_fold_global_{args.device_session1}-{args.device_session2}.pkl","wb")
pickle.dump(EER_global_fold,f)
f.close()
f = open(f"{PATH_LOGS}/nb_attributes_fold_{args.device_session1}-{args.device_session2}.pkl","wb")
pickle.dump(nb_attributes_fold,f)
f.close()
f = open(f"{PATH_LOGS}/cllr_cal_fold_{args.device_session1}-{args.device_session2}.pkl","wb")
pickle.dump(cllr_cal_fold,f)
f.close()

f = open(f"{PATH_LOGS}/cllr_cal_global_fold_{args.device_session1}-{args.device_session2}.pkl","wb")
pickle.dump(cllr_cal_global_fold,f)
f.close()