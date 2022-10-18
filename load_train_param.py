# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================

def load_filter_soft(min_typ,typ_path,dout_path):
    #train typicality from file
    with open (typ_path,"r") as f:
        text=f.readlines()
        f.close()
    typ={}
    for t in text:
        typ[t.split(":")[0].strip()]=float(t.split(":")[1].strip())
    #remove typ=0
    filtered_typ={}
    deleted_va={}
    for va in typ.keys():
        if typ[va]>min_typ:
            filtered_typ[va]=typ[va]
        else:
            deleted_va[va]=typ[va]
    #train dropout from file
    with open (dout_path,"r") as f:
        text=f.readlines()
        f.close()
    dropout={}
    for t in text:
        if t.split(":")[0].strip() in filtered_typ.keys():
            dropout[t.split(":")[0].strip()]=float((t.split(":")[1]).strip())
    return filtered_typ,dropout,deleted_va