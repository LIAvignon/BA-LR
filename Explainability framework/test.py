# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================

from preprocessing.data_distribution import partition_gender_plot,loc_gendre_vox1
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from sklearn.metrics import confusion_matrix


def accuracy(model,X,y):
    y_pred = model.predict(X)
    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
    accuracy =  (TP+TN) /(TP+FP+TN+FN)
    return accuracy
def test_vox1(ba, trained_model, features_vox1, df_binary):
    ba_1_vox1 = list(df_binary[df_binary[ba] == 1]["name"].values)
    features_0_vox1 = []
    features_1_vox1 = []
    for idx, row in features_vox1.iterrows():
        if row["name"] in ba_1_vox1:
            features_1_vox1.append(row)
        else:
            features_0_vox1.append(row)
    dict_ba = {}
    features_0_vox1 = pd.DataFrame(features_0_vox1)
    features_1_vox1 = pd.DataFrame(features_1_vox1)
    nb_m0, nb_f0, utt_man0, utt_female0 = loc_gendre_vox1(features_0_vox1, meta_vox1)
    nb_m1, nb_f1, utt_man1, utt_female1 = loc_gendre_vox1(features_1_vox1, meta_vox1)
    dict_ba[f"0_vox1"] = {"nb_M": nb_m0, "nb_F": nb_f0, "nb_utt_F": utt_female0, "nb_utt_M": utt_man0,"nb_M_train": mloc_train}
    dict_ba[f"1_vox1"] = {"nb_M": nb_m1, "nb_F": nb_f1, "nb_utt_F": utt_female1, "nb_utt_M": utt_man1,"nb_F_train": floc_train}
    partition_gender_plot(dict_ba, ba, f"partition_vox1")
    # remove name column
    features_0_vox1 = features_0_vox1.iloc[:, 1:]
    features_1_vox1 = features_1_vox1.iloc[:, 1:]
    # balance 0 et 1
    features_0_vox1 = features_0_vox1[:len(features_1_vox1)]
    # prepare X and Y for model
    y_vox1 = [0] * len(features_0_vox1) + [1] * len(features_1_vox1)
    X_vox1 = pd.concat([features_0_vox1, features_1_vox1])
    S = StandardScaler()
    X_vox1_scaled = pd.DataFrame(S.fit_transform(X_vox1), columns=X_vox1.columns)
    # apply trained model
    logging.info(f"The accuracy of {ba} trained model on voxceleb1={accuracy(trained_model, X_vox1_scaled, y_vox1)}")
    return accuracy(trained_model, X_vox1_scaled, y_vox1)
