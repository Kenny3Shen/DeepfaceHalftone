import math

import numpy as np


def find_closet_color(in_color):
    in_color = max(0, min(in_color, 255))
    target_col_num = 2
    return int(math.floor(in_color / (256.0 / target_col_num)) * (255.0 / (target_col_num - 1)))


def dithering(src):
    ed_res = np.copy(src)
    if (len(np.shape(src)) != 2):
        print("require gray image")
        return None
    r, c = np.shape(src)
    for i in range(r):
        for j in range(c):
            # error diffusion Floyd_Steinberg
            old_c = ed_res[i][j]
            new_c = find_closet_color(old_c)
            ed_res[i][j] = new_c
            error = old_c - new_c
            if (j != c - 1):
                ed_res[i][j + 1] = int(ed_res[i][j + 1] + error * 7.0 / 16.0)
            if (i != r - 1 and j != 0):
                ed_res[i + 1][j - 1] = int(ed_res[i + 1][j - 1] + error * 3.0 / 16.0)
            if (i != r - 1):
                ed_res[i + 1][j] = int(ed_res[i + 1][j] + error * 5.0 / 16.0)
            if (i != r - 1 and j != c - 1):
                ed_res[i + 1][j + 1] = int(ed_res[i + 1][j + 1] + error * 1.0 / 16.0)
    return ed_res