# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================


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





