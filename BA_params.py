# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================

# Librairies
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import gc
import pickle
import numpy as np
import var_env as env
import itertools
import random
from itertools import product
import time
import re
import argparse
import logging
from preprocessing import *



def compute_dropout(b, profil, utt_spk, matrix_utterances, index_of_b):
    BA_spk = 0
    nb_BA_spk = {}
    spk_has_BA = 0
    dropout_per_spk={}
    for spk in utt_spk.keys():
        nb_BA = 0
        nb_present_BA = 0
        if profil[spk][b] != 0:
            spk_has_BA += 1
            for u in utt_spk[spk]:
                index_utt = int(u[3:])
                if matrix_utterances[index_utt][index_of_b] == 0:
                    nb_BA += 1
                else:
                    nb_present_BA += 1
        nb_BA_spk[spk] = nb_present_BA
        BA_spk += nb_BA / len(utt_spk[spk])
        dropout_per_spk[spk]= nb_BA / len(utt_spk[spk])
    logging.info(f"number of speakers having {b} = {spk_has_BA}")
    out = BA_spk / spk_has_BA
    return out
def compute_typicality(b, couples, profil):
    nb = 0
    for (spk1, spk2) in couples:
        if spk1 != spk2 and profil[spk1][b] == 1 and profil[spk2][b] == 1:
            nb += 1
    # stat_BA[b] = nb
    typ_BA = nb / len(couples)
    return typ_BA

def typicality_and_dropout(profil, couples,  utt_spk, BA, vectors,typ_path,dout_path):
    with open(typ_path, "w+") as file1:
        with open(dout_path, "w+") as file2:
            last_percent = -1
            for index, b in enumerate(BA):
                typ_BA = compute_typicality(b, couples, profil)
                dropout = compute_dropout(b, profil, utt_spk, vectors, index)

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