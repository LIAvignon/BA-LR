# ==============================================================================
#  Copyright (c) 2024. Imen Ben Amor
# ==============================================================================

def LR_00(x, dout, din):
    No_din=1-din
    return (1+dout**2)/(x*(2*dout*No_din+No_din**2+dout**2))
def LR_01(x, dout, din):
    No_dout=1-dout
    No_din=1-din
    return (No_din*din*x+dout*No_dout)/(x*(1+din*x*dout+No_din*din*x+dout*No_dout))
def LR_10(x, dout, din):
    No_dout=1-dout
    No_din=1-din
    return (No_din*din*x+dout*No_dout)/(x*(1+din*x*dout+No_din*din*x+dout*No_dout))
def LR_11(x, dout,din):
    No_dout=1-dout
    return (1+(din*x)**2) / (x*(2*din*x*No_dout+(din*x)**2+No_dout**2))
