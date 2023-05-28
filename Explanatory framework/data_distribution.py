# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================

from matplotlib import pyplot as plt
import pandas as pd
import logging

def partition_gender_plot(dict_ba,ba,chaine):
    stats_ba=pd.DataFrame(dict_ba).T
    stats_ba["ba_state"]=stats_ba.index
    temp1=stats_ba[["nb_M","nb_F","ba_state"]]
    temp2=stats_ba[["nb_utt_M","nb_utt_F","ba_state"]]
    figure, axis = plt.subplots(1, 2)
    figure.suptitle(f"Partition of gender for {ba}")
    plots1=temp2.plot(kind='bar', figsize=(9,4), ylabel='nb_utt_M',ax=axis[0])
    plots2=temp1.plot(kind='bar', figsize=(9,4), ylabel='nb_M',ax=axis[1])
    for bar1,bar2 in zip(plots1.patches,plots2.patches):
        plots1.annotate(format(bar1.get_height(), '.2f'),(bar1.get_x() + bar1.get_width() / 2,bar1.get_height()), ha='center', va='center', size=6, xytext=(0, 5),textcoords='offset points')
        plots2.annotate(format(bar2.get_height(), '.2f'),(bar2.get_x() + bar2.get_width() / 2,bar2.get_height()), ha='center', va='center',size=6, xytext=(0, 5),textcoords='offset points')
    stats_ba.to_csv(f"data/BA/parition_gender_{ba}_{chaine}.csv")
    plt.tight_layout()
    plt.savefig(f"data/BA/partition_gender_{ba}_{chaine}.png")
    plt.show()


def loc_gendre(X,meta_vox2):
    locs=[]
    gendre=[]
    dict_spk={}
    for idx,row in X.iterrows():
        locs.append(X["name"][idx].split("/")[-3])
        gendre.append(meta_vox2[meta_vox2["VoxCeleb2 ID "]==X["name"][idx].split("/")[-3]+" "]["Gender "].values[0].rstrip())
    utt_man=gendre.count("m")
    utt_female=gendre.count("f")
    for s,g in zip(locs,gendre):
        dict_spk[s]=g
    nb_m=list(dict_spk.values()).count("m")
    nb_f=list(dict_spk.values()).count("f")
    return nb_m,nb_f,utt_man,utt_female,list(dict_spk.keys())

def loc_gendre_vox1(X,meta_vox1):
    locs=[]
    gendre=[]
    dict_spk={}
    for idx,row in X.iterrows():
        locs.append(X["name"][idx].split("/")[-3])
        gendre.append(meta_vox1[meta_vox1["VoxCeleb1 ID"]==X["name"][idx].split("/")[-3]]["Gender"].values[0])
    logging.info(f'Number of utterances of man ={gendre.count("m")}')
    logging.info(f'Number of utterances of female ={gendre.count("f")}')
    utt_man=gendre.count("m")
    utt_female=gendre.count("f")
    for s,g in zip(locs,gendre):
        dict_spk[s]=g
    logging.info(f'Number of men ={list(dict_spk.values()).count("m")}')
    logging.info(f'Number of female ={list(dict_spk.values()).count("f")}')
    nb_m=list(dict_spk.values()).count("m")
    nb_f=list(dict_spk.values()).count("f")
    return nb_m,nb_f,utt_man,utt_female
