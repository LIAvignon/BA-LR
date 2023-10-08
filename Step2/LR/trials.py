# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================

def load_trials():
    with open("data/test/target_trials.txt","r") as f:
        text=f.readlines()
        target=[]
        for couple in text:
            target.append(eval(couple.strip()))
    with open("data/test/non_trials.txt","r") as f:
        text=f.readlines()
        non=[]
        for couple in text:
            non.append(eval(couple.strip()))
    return non, target