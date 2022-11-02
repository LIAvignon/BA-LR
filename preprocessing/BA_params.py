# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================
# Librairies
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import gc
import numpy as np
from preprocessing.xvectorParser import readVectors
import var_env as env
import itertools
import random
from itertools import product
import scipy.stats as stats
import argparse
import logging


def number_utterances(utt):
    '''
    This function calculates for each speaker the number of utterances
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
def todelete(xvectors):
    """
    This function deletes zero columns for all rows in array "xvectors"
    :param xvectors: array of binary xvectors
    :return: filtered array, index of deleted column
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
    This function calculate the profile for each speaker
    :param xvectors: array of xvectors
    :param utt_per_spk: dict spk: nb_utterances
    :param BA: list of BAs
    :return:
    """
    profil = {}
    j = 0
    for spk in list(utt_per_spk.keys()):
        BA_dict = {}
        df_spk = xvectors[j:utt_per_spk[spk] + j]
        for c, ba in zip(df_spk.T, BA):
            if 1 in c:
                BA_dict[ba] = 1
            else:
                BA_dict[ba] = 0
        profil[spk] = BA_dict
        j += utt_per_spk[spk]
        # print(j)
    return profil
def compute_typicality(b, couples, profil):
    """
    This function calculates typicality
    :param b: BAi
    :param couples: combination of all speakers in couples
    :param profil: dictionary of speakers profiles
    :return: dictionary of BAi:typ_value
    """
    nb = 0
    for (spk1, spk2) in couples:
        if spk1 != spk2 and profil[spk1][b] == 1 and profil[spk2][b] == 1:
            nb += 1
    # stat_BA[b] = nb
    typ_BA = nb / len(couples)
    return typ_BA
def utterance_spk(nb_utt_spk):
    """
    This function provides a dictionary of the utterance for spki
    :param nb_utt_spk: dictionary of spk:nbr of utterances
    :return: spk1:["utt0","utt1"],spk2:["utt3","utt4"]
    """
    utt_spk = {}
    j = 0
    for spk in nb_utt_spk.keys():
        nb = nb_utt_spk[spk]
        utt_spk[spk] = ["utt" + str(i) for i in range(j, j + nb)]
        j += nb
    return utt_spk

def utterance_dictionary(binary_vectors, utterances, BA):
    """
    This function gives the binary vector (using BAs) for each utterance
    :param binary_vectors: array of all binary vectors files
    :param utterances: list of utterances ids
    :param BA: list of BAs
    :return: {"id001-9fddfetl-001":{"BA0":1,"BA2":0, "BA3":1..},...}
    """
    utt = {}
    for (u, row) in zip(utterances, binary_vectors):
        utt[u] = {b: i for i, b in zip(row, BA)}
    return utt

def typicality_and_dropout(profil, couples,  utt_spk, BA, vectors,typ_path,dout_path):
    """
    This function calculate the typicality and Dropout for all BAs
    :param profil: dictionary of speakers profiles
    :param couples: combination of all speakers in couples
    :param utt_spk: dictionary spk: list of utterances"index"
    :param BA:
    :param vectors: Train data binary array
    :param typ_path: path of typicality file
    :param dout_path: path of dropout file
    :return: 2 files
    """
    with open(typ_path, "w+") as file1:
        with open(dout_path, "w+") as file2:
            last_percent = -1
            for index, b in enumerate(BA):
                typ_BA = compute_typicality(b, couples, profil)
                dropout= compute_dropout(b, profil, utt_spk, vectors, index)

                file1.write("%s : %f " % (b, typ_BA))
                file1.write("\n")

                file2.write("%s:%f" % (b, dropout))
                file2.write("\n")

                percent = round((index / len(BA)) * 100, 0)
                if percent % 10 == 0 and last_percent != percent:
                    logging.info(f"{percent}%")
                    last_percent = percent

        file2.close()
    file1.close()


def stringToList(string):
    listRes = list(string.split(" "))
    return listRes

def readVectors(filePath):
    vectors = []
    utt = []
    with open(filePath, "r") as f:
        line_idx = 0
        last_printed_percent = -1
        number_of_lines = 5105875
        for line in f:
            line_idx += 1
            elems = line.split("  ")
            vec = []
            utt.append(elems[0])
            for elem in stringToList(elems[1][2:-2].rstrip()):
                value_to_append = 1 if (round(float(elem),4) != 0) else 0
                vec.append(value_to_append)
            vectors.append(vec)
            percent = round(line_idx / number_of_lines * 100, 0)
            if percent % 10 == 0 and percent != last_printed_percent:
                print(f"{percent}%")
                last_printed_percent = percent
    return utt, np.array(vectors)


if __name__ == "__main__":
    # Arguments
    env.logging_config(env.PATH_LOGS + "/logFile")
    parse = argparse.ArgumentParser()
    parse.add_argument("--path",default="data/xvectors.txt",type=str)
    parse.add_argument("--typ_path",default="data/typ.txt",type=str)
    parse.add_argument("--dout_path",default="data/dout.txt",type=str)
    args = parse.parse_args()
    logging.info("read xvectors")
    utterances, binary = readVectors(args.path)
    logging.info("finish reading xvectors")
    logging.info("xvectors array ready")
    utt_per_spk, loc_list = number_utterances(utterances)
    logging.info("delete zero columns...")
    binary_vectors, idx = todelete(binary)
    logging.info(f"number of deleted columns: {len(idx)}")
    BA = ['BA' + str(i) for i in range(binary.shape[1]) if np.array([i]) not in idx]
    #liberate memory
    del binary
    del loc_list
    del idx
    gc.collect()
    logging.info("utterance_spk...")
    utt_spk = utterance_spk(utt_per_spk)
    logging.info("profil_spk...")
    profil = profil_spk(binary_vectors, utt_per_spk, BA)
    # speakers couples
    logging.info("computing combinations...")
    couples = list(itertools.combinations(utt_per_spk.keys(), 2))
    typicality_and_dropout(profil, couples,  utt_spk, BA, binary_vectors,args.typ_path,args.dout_path)