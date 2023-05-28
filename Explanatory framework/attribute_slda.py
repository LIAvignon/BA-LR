# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================

import var_env as env
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import logging
env.logging_config(env.PATH_LOGS + "/logFile_SLDA_BA")

def lda_forward(X,y,slentry=0.05):
    if not isinstance(X,pd.DataFrame):
        raise ValueError("X should be dataFrame")
    if not isinstance(y,pd.Series):
        y = pd.Series(y)
    if X.shape[0] != len(y):
        raise ValueError("Number of samples not same between X and y")
    n = X.shape[0]
    p = X.shape[1]
    K = len(y.unique())
    K_values = y.unique()
    lst_Vk = [np.cov(X[y==k].values.T)*(X[y==k].shape[0]-1) for k in K_values]
    W = sum(lst_Vk)/n
    V = (n-1)/n*np.cov(X.values.T)
    curVars = []
    candidatsVars = X.columns
    curLambda = 1.0
    q = 0
    lambdas = []
    nb_var = []
    i = 0
    while i<p:
        if len(candidatsVars)==0:
            logging.info("End of process")
            logging.info("\n")
            break
        logging.info("Step :",(q+1))
        curRes = pd.DataFrame(np.zeros((len(candidatsVars),3)),index=candidatsVars,columns=["Lambda","F","p-value"])
        for v in candidatsVars:
            tmpVars = curVars + [v]
            W = pd.DataFrame(W,columns=list(X.columns),index=list(X.columns))
            tmpW = W.loc[tmpVars,tmpVars]
            V = pd.DataFrame(V,columns=list(X.columns),index=list(X.columns))
            tmpV = V.loc[tmpVars,tmpVars]
            tmpLambda = np.linalg.det(tmpW)/np.linalg.det(tmpV)
            tmpF = (n-K-q)/(K-1)*(curLambda/tmpLambda-1)
            tmpPValue = stats.f.sf(tmpF,K-1,n-K-q)
            curRes.loc[v,:] = [tmpLambda,tmpF,tmpPValue]
        if curRes.shape[0] > 1:
            curRes = curRes.sort_values("F",ascending=False)
        if curRes.iloc[0,2] < slentry:
            theBest = curRes.index[0]
            logging.info("==> Add variable to the model :",theBest)
            curVars = curVars + [theBest]
            nb_var.append(len(curVars))
            candidatsVars = [v for v in candidatsVars if v != theBest]
            logging.info("\n")
            q = q + 1
            curLambda = curRes.iloc[0,0]
            lambdas.append(curLambda)
        else:
            logging.info("No variable respects selection criteria")
            logging.info("End of process")
            lambdas.append(curLambda)
        i = i + 1
    return {"variables":curVars,"lambda":lambdas,"number_variables":nb_var}

def variables_selection(ba):
    ba0=pd.read_csv(f"data/BA/{ba}_0.csv")
    ba1=pd.read_csv(f"data/BA/{ba}_1.csv")
    ba0=ba0.drop(['Unnamed: 0',"name"],axis=1)
    ba1=ba1.drop(['Unnamed: 0',"name"],axis=1)
    X=pd.concat([ba0,ba1])
    S = StandardScaler()
    X=pd.DataFrame(S.fit_transform(X),columns=ba0.columns)
    y=[0]*len(ba0)+[1]*len(ba1)
    lda_dict=lda_forward(X,y,slentry=0.01)
    logging.info(f"number of selected variables for {ba} is {len(lda_dict['lambda'])}")
    plt.figure(figsize=(5,5))
    plt.scatter([i for i in range(len(lda_dict["number_variables"]))],lda_dict["lambda"])
    plt.xlabel("Number of variables")
    plt.ylabel("Lambda values")
    plt.show()
    plt.savefig(f"data/BA/var_selection_{ba}.png")
    return lda_dict,X,y,ba0,ba1


BA=[f"BA{i}" for i in range(256)]
for ba in BA:
    if os.path.isfile(f"data/BA/{ba}_0.csv"):
        logging.info(f"====={ba}=====")
        lda_dict,X,y,ba0,ba1=variables_selection(ba)
        selected_variables = []
        for i, v in zip(lda_dict["lambda"], lda_dict["variables"]):
            selected_variables.append((v, i))
        df_var = pd.DataFrame(selected_variables, columns=["variables", "lambda"])
        df_var.to_csv(f"data/BA/var_selected_{ba}.csv")