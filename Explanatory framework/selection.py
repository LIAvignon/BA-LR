# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================

import pickle
import pandas as pd
import logging
import var_env as env
import argparse
if __name__ == "__main__":
    # Arguments
    env.logging_config(env.PATH_LOGS + "/logFile_BAparams")
    parse = argparse.ArgumentParser()
    parse.add_argument("--path_params",default="data/open_smile_params.csv",type=str)
    parse.add_argument("--path_utt_list", default="data/BA_uttlist_0_1.pickle", type=str)
    parse.add_argument("--path_BA", default="data/BA/",type=str)
    args = parse.parse_args()
    df = pd.read_csv(args.path_params)
    with open(args.path_utt_list,"rb") as f:
        BA_list_utt=pickle.load(f)
    for ba in list(BA_list_utt.keys()):
        ba_0=df.loc[df['name'].isin(BA_list_utt[ba]["0"])]
        ba_1=df.loc[df['name'].isin(BA_list_utt[ba]["1"])]
        logging.info(f"{ba}: number of 0={len(BA_list_utt[ba]['0'])} and number of utterances selected={len(ba_0)}")
        logging.info(f"{ba}: number of 1={len(BA_list_utt[ba]['1'])} and number of utterances selected={len(ba_1)}")
        ba_0.to_csv(args.path_BA+f"{ba}_0.csv")
        ba_1.to_csv(args.path_BA+f"{ba}_1.csv")