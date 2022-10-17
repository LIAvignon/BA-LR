# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================

from itertools import combinations
import numpy as np

def compute_entropy(field,df):
    values = df[field].value_counts()
    n = values.sum()
    probs = [v / n for v in values]
    H = -sum([p * np.log(p) for p in probs if p > 0.0])
    return H

def compute_mi(x, y,df):
    print((x,y))
    def compute_pmi(xy_c, x_c, y_c, xy_n, x_n, y_n):
        a = xy_c / xy_n
        b = x_c / x_n
        c = y_c / y_n
        d = a / (b * c)

        if b == 0 or c == 0 or d == 0:
            return 0

        return a * np.log(d)

    def compute_joint_entropy(probs):
        return sum([p * np.log(p) for p in probs])

    xy_00 = df[(df[x]==0) & (df[y]==0)].shape[0]
    xy_01 = df[(df[x]==0) & (df[y]==1)].shape[0]
    xy_10 = df[(df[x]==1) & (df[y]==0)].shape[0]
    xy_11 = df[(df[x]==1) & (df[y]==1)].shape[0]
    xy_n = xy_00 + xy_01 + xy_10 + xy_11

    x_0 = df[df[x]==0].shape[0]
    x_1 = df[df[x]==1].shape[0]
    x_n = x_0 + x_1

    y_0 = df[df[y]==0].shape[0]
    y_1 = df[df[y]==1].shape[0]
    y_n = y_0 + y_1

    values = [
        compute_pmi(xy_00, x_0, y_0, xy_n, x_n, y_n),
        compute_pmi(xy_01, x_0, y_1, xy_n, x_n, y_n),
        compute_pmi(xy_10, x_1, y_0, xy_n, x_n, y_n),
        compute_pmi(xy_11, x_1, y_1, xy_n, x_n, y_n)
    ]

    MI = sum(values)
    H_X = compute_entropy(x,df)
    H_Y = compute_entropy(y,df)

    values = [
        (xy_00/xy_n, x_0/x_n, y_0/y_n),
        (xy_01/xy_n, x_0/x_n, y_1/y_n),
        (xy_10/xy_n, x_1/x_n, y_0/y_n),
        (xy_11/xy_n, x_1/x_n, y_1/y_n)
    ]

    I_XY = sum([p_xy * np.log(p_x * p_y) for p_xy, p_x, p_y in values])
    H_XY = compute_joint_entropy([xy_00/xy_n, xy_01/xy_n, xy_10/xy_n, xy_11/xy_n])

    C_XY = MI / H_Y
    C_YX = MI / H_X
    R = MI / (H_X + H_Y)
    U = 2 * R
    IQR = (I_XY / H_XY) - 1
    P = MI / np.sqrt(H_X * H_Y)

    return {
        'X': x,
        'Y': y,
        'H(X)': H_X,
        'H(Y)': H_Y,
        'H(X,Y)': H_XY,
        'I(X;Y)': MI,
        'R(X;Y)': R,
        'U(X;Y)': U,
        'C(X;Y)': C_XY,
        'C(Y;X)': C_YX,
        'IQR(X,Y)': IQR,
        'P(X;Y)': P

    }