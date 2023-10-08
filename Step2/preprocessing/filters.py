# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================
#BA lost: BA doesn't exist in Train so they are not considered in BA-vectors in test phase
def filter_BA_not_train(df_binary,BA_train):
    not_in_train=[]
    for c in df_binary.columns:
        if c not in BA_train:
            if c!="id_spk":
                not_in_train.append(c)
    df_binary=df_binary.drop(not_in_train,axis=1)
    return df_binary

def filter_BA_test(df_binary,VA_train):
    VA_test=[]
    for c in df_binary.columns:
        if c in VA_train and c!="id_spk":
            VA_test.append(c)
    return VA_test


def filter_train_param(BA_test,typ_va,dropout):
    typ_va_test={}
    dropout_test={}
    for v in BA_test:
        typ_va_test[v]=typ_va[v]
        dropout_test[v]=dropout[v]
    return typ_va_test,dropout_test