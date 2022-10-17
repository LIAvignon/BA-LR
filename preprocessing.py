# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================

import numpy as np
import pandas as pd
import var_env as env
import itertools
from itertools import product

def utterance_dictionary(df_binary):
    """
    :param df_binary: dataframe of BA-vectors
    :return: dictionary of {utt_i:BA_vector}
    """
    utt = {}
    for idx, row in df_binary.iterrows():
        BA_dict = {}
        for c, r in zip(row.index, row):
            BA_dict[c] = r
        utt["utt" + str(idx)] = BA_dict
    return utt

def binarize_data(xvectors):
    xvectors[xvectors != 0] = 1
    return xvectors

def nb_spk_per_BA(BA,profil):
    """
    :param BA: list of BAs
    :param profil: BA exist in at least one utterance of the speaker
    :return: nb of speakers having the BA at least once
    """
    spk_BA={}
    for BA in BA:
        nb = 0
        for p in profil.keys():
            if profil[p][BA] == 1:
                nb += 1
        spk_BA[BA] = nb
    return spk_BA

def number_utterances(utt):
    '''
    :param utt: list of utterances
    :return: dictionary of speakers and corresponding number of utterances
    '''
    speaker_dict = {}
    loc_list = []
    for u in utt:
        first_element_after_split = u.split("-")[0]
        if speaker_dict.get(first_element_after_split) is not None:
            speaker_dict[first_element_after_split] += 1
        else:
            speaker_dict[first_element_after_split] = 1
        loc_list.append(first_element_after_split)

    return speaker_dict, loc_list
def utterance_spk(nb_utt_spk):
    """
    :param nb_utt_spk:
    :return: dictionary of {spk:utt_i}
    """
    utt_spk = {}
    j = 0
    for spk in nb_utt_spk.keys():
        nb = nb_utt_spk[spk]
        utt_spk[spk] = ["utt" + str(i) for i in range(j, j + nb)]
        j += nb
    return utt_spk

def todelete(xvectors):
    """
    Delete BAs that are null for all utterances
    :param xvectors:
    :return: returned vectors and index of deleted
    """
    res=np.all(xvectors[..., :] == 0, axis=0)
    idx=[]
    for i in range(len(res)):
        if res[i]:
            idx.append(i)
    v = np.delete(xvectors, idx, axis=1)
    return v, idx

def profil_spk(xvectors, utt_per_spk, BA):
    """
    Profile of speaker S_{j}: P_{S_{j}} = {BA_{i}^{S_{j}} =1, if BA_{i}
    :param xvectors:
    :param utt_per_spk:
    :param BA:
    :return:dcitionary
    """
    profil = {}
    j = 0
    for spk in list(utt_per_spk.keys()):
        BA_dict = {}
        df_spk = xvectors[j:utt_per_spk[spk] + j]
        nb = 0
        for c, ba in zip(df_spk.T, BA):
            if 1 in c:
                nb = 1
                BA_dict[ba] += 1
            else:
                BA_dict[ba] = 0
        profil[spk] = BA_dict
        j += utt_per_spk[spk]
        # print(j)
    return profil