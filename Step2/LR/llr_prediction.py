


# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================

from lr import LR_00,LR_10,LR_01,LR_11,Cllr_min
import numpy as np
import matplotlib.pyplot as plt

def LLR(pair,utt,dropout,typ_va,prob_dropin):
    LLR= []
    for (u1, u2) in pair:
        lr = 1
        for i in utt[u1]:
            if utt[u1][i] == 1 and utt[u2][i] == 1:
                lr *= LR_11(typ_va[i], dropout[i], prob_dropin)
            elif utt[u1][i] != utt[u2][i]:
                lr *= (LR_01(typ_va[i], prob_dropin) + LR_10(typ_va[i], dropout[i])) / 2
            else:
                lr *= LR_00(typ_va[i], dropout[i], prob_dropin)
        LLR.append(np.log(lr))
    return LLR
def LR_framework(dropout,typ_va,utt,target,non,lprob_dropin):
    list_cllr_min=[]
    list_cllr_act=[]
    list_eer=[]
    list_Din=[]
    for prob_dropin in lprob_dropin:
        list_Din.append(prob_dropin)
        LLR_target=LLR(target,utt,dropout,typ_va,prob_dropin)
        LLR_non=LLR(non,utt,dropout,typ_va,prob_dropin)
        cllr_mini,cllr_act,eer,cal_tar,cal_non=Cllr_min(LLR_target, LLR_non)
        eer=eer*100
        list_eer.append(eer)
        list_cllr_min.append(cllr_mini)
        list_cllr_act.append(cllr_act)
        plt.subplots(1, 1, figsize=(10, 5))
        plt.hist(LLR_target, alpha=0.5, bins="auto", label=f"{len(LLR_target)} target")
        plt.hist(LLR_non, alpha=0.5, bins="auto", label=f"{len(LLR_non)} non")
        plt.title(f"relu test C={prob_dropin},Cllr (min/act):({cllr_mini}, {cllr_act}),eer= {eer} ")
        plt.xlabel("LLR scores")
        plt.legend()
        plt.savefig("data/"+f"{prob_dropin}"+".png")
    return LLR_target,LLR_non,list_eer,list_cllr_min,list_cllr_act,list_Din
