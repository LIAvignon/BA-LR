# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================

def load_filter_soft(typ_path,dout_path):
    #train typicality from file
    with open (typ_path,"r") as f:
        text=f.readlines()
        f.close()
    typ={}
    for t in text:
        typ[t.split(":")[0].strip()]=float(t.split(":")[1].strip())

    #train dropout from file
    with open (dout_path,"r") as f:
        text=f.readlines()
        f.close()
    dropout={}
    for t in text:
        dropout[t.split(":")[0].strip()]=float((t.split(":")[1]).strip())
    return typ,dropout