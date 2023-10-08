# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================

import numpy as np
from Step2.LR.performance import cllr, min_cllr
#=============================== LR Framework ===================================================
def LR_00(x, prob_dropout, prob_dropin):
    return  1/ (x*((1-prob_dropin)+prob_dropout))
def LR_01(x,  prob_dropin):
    return prob_dropin / (prob_dropin*x + (1 - prob_dropin))

def LR_10(x, prob_dropout):
    return prob_dropout / x

def LR_11(x, prob_dropout, prob_dropin):
    return 1/ (x*(x * prob_dropin + (1 - prob_dropout)))
#================================ Calibration ===============================================

def Cllr_min(LLR_target, LLR_non):
    """
    :param LLR_target:
    :param LLR_non:
    :return: Cllrmin cllract, eer and calibrated scores
    """
    compute_eer = True
    tar = np.array(LLR_target)
    non = np.array(LLR_non)
    cllr_act = cllr(tar, non)
    if compute_eer:
        cllr_min,eer,cal_tar,cal_non= min_cllr(tar, non, compute_eer=True)
    else:
        cllr_min,cal_tar,cal_non= min_cllr(tar, non)

    return cllr_min, cllr_act,eer,cal_tar,cal_non