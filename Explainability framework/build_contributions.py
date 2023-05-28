# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================

import pandas as pd


def contributions(df_0, X_train):
    # list of variables in each group
    var_groups = {}
    for c in X_train.columns:
        if c.startswith("log"):
            key = c[:3]
        else:
            key = c[:2]
        if key not in var_groups.keys():
            var_groups[key] = [c]
        else:
            var_groups[key].append(c)

    # Individual mean shap value of each variable
    meanshap_val_var = {}
    for var in df_0.columns:
        meanshap_val_var[var] = sum(abs(df_0[var])) / df_0.shape[0]

    # cumul of variables in a group
    mean_shap_group = {}
    for key in var_groups.keys():
        s = 0
        for var in var_groups[key]:
            s += meanshap_val_var[var]
            mean_shap_group[key] = s
    return var_groups, mean_shap_group, meanshap_val_var
def build_dataframe(var_groups,contribution,contribution_var):
    list_all_subkeys=[]
    list_key=[]
    list_val_key=[]
    list_subkey=[]
    new_contribution={}
    new_contribution_var={}
    for k in var_groups.keys():
        list_key.extend([k]*len(var_groups[k]))
        new_contribution[k]=contribution[k]/sum(contribution.values())
        list_val_key.extend([new_contribution[k]]*len(var_groups[k]))
        list_subkey.extend(var_groups[k])
        templist=[]
        for i in var_groups[k]:
            if contribution[k]!=0:
                new_contribution_var[i]=contribution_var[i]
            else:
                new_contribution_var[i] = 0
            templist.append(new_contribution_var[i])
        list_all_subkeys.extend(templist)
    df_plot=pd.DataFrame(list_key,columns=["Family"])
    df_plot["Members"]=list_subkey
    df_plot["Cont_Member"]=list_all_subkeys
    df_plot["Cont_Family"]=list_val_key
    return df_plot,new_contribution,new_contribution_var