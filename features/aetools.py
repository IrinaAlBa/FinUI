import numpy as np
import pandas as pd
import itertools
import cv2


def get_balance_measure(h, w, df):

    mid_x, mid_y = w / 2, h / 2

    tmp = df.loc[df.c_x < mid_x]
    w_l = ((mid_x - tmp.c_x) * tmp.a).sum()

    tmp = df.loc[df.c_x > mid_x]
    w_r = ((tmp.c_x - mid_x) * tmp.a).sum()

    tmp = df.loc[df.c_y < mid_y]
    w_t = ((mid_y - tmp.c_y) * tmp.a).sum()

    tmp = df.loc[df.c_y > mid_y]
    w_b = ((tmp.c_y - mid_y) * tmp.a).sum()

    bm_v = abs(w_l - w_r) / max(w_l, w_r)
    bm_h = abs(w_t - w_b) / max(w_t, w_b)

    return 1 - (bm_v + bm_h) / 2


def get_equilibrium_measure(h, w, df):

    mid_x, mid_y = w/ 2, h / 2

    em_x = (2 / w) * np.average(df.c_x - mid_x, weights=df.a)
    em_y = (2 / h) * np.average(df.c_y - mid_y, weights=df.a)

    return 1 - (abs(em_x) + abs(em_y)) / 2


def get_symmetry_measure(h, w, df):

    mid_x, mid_y = w / 2, h / 2
    
    try:
        tmp = {
            'UL': df.loc[(df.c_x < mid_x) & (df.c_y < mid_y)],
            'UR': df.loc[(df.c_x > mid_x) & (df.c_y < mid_y)],
            'LL': df.loc[(df.c_x < mid_x) & (df.c_y > mid_y)],
            'LR': df.loc[(df.c_x > mid_x) & (df.c_y > mid_y)]
        }

        x_, y_, w_, h_, th_, r_ = {}, {}, {}, {}, {}, {}
        for q in tmp:
            x_[q] = 1.0 if tmp[q].empty else (2 / w) * np.abs(tmp[q].c_x - mid_x).mean()
            y_[q] = 1.0 if tmp[q].empty else (2 / h) * np.abs(tmp[q].c_y - mid_y).mean()
            w_[q] = 0.0 if tmp[q].empty else (2 / w) * tmp[q].w.mean()
            h_[q] = 0.0 if tmp[q].empty else (2 / h) * tmp[q].h.mean()
            th_[q] = mid_y / mid_x if tmp[q].empty else (np.abs(tmp[q].c_y - mid_y) / np.abs(tmp[q].c_x - mid_x)).mean()
            r_[q] = np.sqrt((w**2 + h**2)/(w*h)) if tmp[q].empty else (2 / np.sqrt(w * h)) * np.sqrt((tmp[q].c_x - mid_x) ** 2 + (tmp[q].c_y - mid_y) ** 2).mean()

        sym_v = abs(x_['UL'] - x_['UR']) / max(x_['UL'], x_['UR'])
        sym_v += abs(x_['LL'] - x_['LR']) / max(x_['LL'], x_['LR'])
        sym_v += abs(y_['UL'] - y_['UR']) / max(y_['UL'], y_['UR'])
        sym_v += abs(y_['LL'] - y_['LR']) / max(y_['LL'], y_['LR'])
        sym_v += abs(h_['UL'] - h_['UR']) / max(h_['UL'], h_['UR'])
        sym_v += abs(h_['LL'] - h_['LR']) / max(h_['LL'], h_['LR'])
        sym_v += abs(w_['UL'] - w_['UR']) / max(w_['UL'], w_['UR'])
        sym_v += abs(w_['LL'] - w_['LR']) / max(w_['LL'], w_['LR'])
        sym_v += abs(th_['UL'] - th_['UR']) / max(th_['UL'], th_['UR'])
        sym_v += abs(th_['LL'] - th_['LR']) / max(th_['LL'], th_['LR'])
        sym_v += abs(r_['UL'] - r_['UR']) / max(r_['UL'], r_['UR'])
        sym_v += abs(r_['LL'] - r_['LR']) / max(r_['LL'], r_['LR'])
        sym_v /= 12

        sym_h = abs(x_['UL'] - x_['LL']) / max(x_['UL'], x_['LL'])
        sym_h += abs(x_['UR'] - x_['LR']) / max(x_['UR'], x_['LR'])
        sym_h += abs(y_['UL'] - y_['LL']) / max(y_['UL'], y_['LL'])
        sym_h += abs(y_['UR'] - y_['LR']) / max(y_['UR'], y_['LR'])
        sym_h += abs(h_['UL'] - h_['LL']) / max(h_['UL'], h_['LL'])
        sym_h += abs(h_['UR'] - h_['LR']) / max(h_['UR'], h_['LR'])
        sym_h += abs(w_['UL'] - w_['LL']) / max(w_['UL'], w_['LL'])
        sym_h += abs(w_['UR'] - w_['LR']) / max(w_['UR'], w_['LR'])
        sym_h += abs(th_['UL'] - th_['LL']) / max(th_['UL'], th_['LL'])
        sym_h += abs(th_['UR'] - th_['LR']) / max(th_['UR'], th_['LR'])
        sym_h += abs(r_['UL'] - r_['LL']) / max(r_['UL'], r_['LL'])
        sym_h += abs(r_['UR'] - r_['LR']) / max(r_['UR'], r_['LR'])
        sym_h /= 12

        sym_r = abs(x_['UL'] - x_['LR']) / max(x_['UL'], x_['LR'])
        sym_r += abs(x_['UR'] - x_['LL']) / max(x_['UR'], x_['LL'])
        sym_r += abs(y_['UL'] - y_['LR']) / max(y_['UL'], y_['LR'])
        sym_r += abs(y_['UR'] - y_['LL']) / max(y_['UR'], y_['LL'])
        sym_r += abs(h_['UL'] - h_['LR']) / max(h_['UL'], h_['LR'])
        sym_r += abs(h_['UR'] - h_['LL']) / max(h_['UR'], h_['LL'])
        sym_r += abs(w_['UL'] - w_['LR']) / max(w_['UL'], w_['LR'])
        sym_r += abs(w_['UR'] - w_['LL']) / max(w_['UR'], w_['LL'])
        sym_r += abs(th_['UL'] - th_['LR']) / max(th_['UL'], th_['LR'])
        sym_r += abs(th_['UR'] - th_['LL']) / max(th_['UR'], th_['LL'])
        sym_r += abs(r_['UL'] - r_['LR']) / max(r_['UL'], r_['LR'])
        sym_r += abs(r_['UR'] - r_['LL']) / max(r_['UR'], r_['LL'])
        sym_r /= 12
    except:
        return 0.

    return 1 - (sym_v + sym_h + sym_r) / 3


def get_sequence_measure(h, w, df):
    mid_x, mid_y = w/2, h/2
    tmp = {
        'UL': df.loc[(df.c_x<mid_x) & (df.c_y<mid_y)],
        'UR': df.loc[(df.c_x>mid_x) & (df.c_y<mid_y)],
        'LL': df.loc[(df.c_x<mid_x) & (df.c_y>mid_y)],
        'LR': df.loc[(df.c_x>mid_x) & (df.c_y>mid_y)]
    }
    w = pd.DataFrame(
        {'area': [tmp[q].a.sum() for q in tmp], 'q': np.arange(4,0,-1)}
    ).sort_values(by='area', ascending=False).reset_index()
    w['v'] = np.arange(4, 0, -1)

    return 1 - np.abs(w.v - w.q).sum()/8


def get_cohesion_measure(h, w, df):

    c = (df.h / df.w).mean() / (h/w)

    return c if c<=1 else 1/c


def get_unity_measure(df):

    n_h = len(np.round(df.h/df.h.min()).unique())
    n_w = len(np.round(df.w/df.w.min()).unique())

    return 1 - (n_h + n_w - 2) / (2 * len(df))


def get_proportion_measure(h, w, df):

    r = min(h, w)/max(h, w)

    nice = np.array([1., 1./np.sqrt(2), 2./(np.sqrt(5)+1), 1./np.sqrt(3), 0.5])

    pm_lo = 1 - 2*min(abs(r - nice))

    pm_o = np.array(
        [1-2*min(abs(r-nice)) for r in df[['h','w']].min(axis=1)/df[['h','w']].max(axis=1)]
    ).mean()

    return (pm_lo + pm_o) / 2


def get_simplicity_measure(df):

    nh = len((df.c_x/df.w.min()).astype(int).unique())
    nv = len((df.c_y/df.h.min()).astype(int).unique())

    return 3 / (nh + nv + len(df))


def get_density_measure(h, w, df):

    return max(0, 1 - 2 * abs(0.5 - df.a.sum()/(h*w)))


def get_regularity_measure(df):
    n = len(df)
    nh = len((df.c_x / df.w.min()).astype(int).unique())
    nv = len((df.c_y / df.h.min()).astype(int).unique())
    a = 1 - (nv + nh) / (2 * n)

    nsh = len(np.unique(np.diff(np.sort((df.c_x / df.w.min()).astype(int).unique()))))
    nsv = len(np.unique(np.diff(np.sort((df.c_y / df.h.min()).astype(int).unique()))))
    s = 1 - (nsv + nsh) / (2 * (n - 1))

    return (a + s) / 2


def get_economy_measure(df):

    n_h = len(np.round(df.h/df.h.min()).unique())
    n_w = len(np.round(df.w/df.w.min()).unique())

    return 2 / (n_h + n_w)


def get_homogeneity_measure(h, w, df):
    n = len(df)
    mid_x, mid_y = w/2, h/2

    nUL = len(df.loc[(df.c_x<mid_x) & (df.c_y<mid_y)])
    nUR = len(df.loc[(df.c_x>mid_x) & (df.c_y<mid_y)])
    nLL = len(df.loc[(df.c_x<mid_x) & (df.c_y>mid_y)])
    nLR = len(df.loc[(df.c_x>mid_x) & (df.c_y>mid_y)])

    score  = np.math.factorial(int(n/4)) / np.math.factorial(nUL)
    score *= np.math.factorial(int(n/4)) / np.math.factorial(nUR)
    score *= np.math.factorial(int(n/4)) / np.math.factorial(nLL)
    score *= np.math.factorial(int(n/4)) / np.math.factorial(nLR)

    return score


def get_rhythm_measure(h, w, df):

    mid_x, mid_y = w / 2, h / 2
    
    try:
        tmp = {
            'UL': df.loc[(df.c_x < mid_x) & (df.c_y < mid_y)],
            'UR': df.loc[(df.c_x > mid_x) & (df.c_y < mid_y)],
            'LL': df.loc[(df.c_x < mid_x) & (df.c_y > mid_y)],
            'LR': df.loc[(df.c_x > mid_x) & (df.c_y > mid_y)]
        }

        x_, y_, a_ = {}, {}, {}
        for q in tmp:
            x_[q] = 1.0 if tmp[q].empty else (2 / w) * np.abs(tmp[q].c_x - mid_x).mean()
            y_[q] = 1.0 if tmp[q].empty else (2 / h) * np.abs(tmp[q].c_y - mid_y).mean()
            a_[q] = 0.0 if tmp[q].empty else (2 / w) * (2 / h) * tmp[q].a.mean()

        rhm_x, rhm_y, rhm_a = 0., 0., 0.
        for keys in list(itertools.combinations(tmp.keys(), 2)):
            rhm_x += abs(x_[keys[0]] - x_[keys[1]]) / max(x_[keys[0]], x_[keys[1]])
            rhm_y += abs(y_[keys[0]] - y_[keys[1]]) / max(y_[keys[0]], y_[keys[1]])
            rhm_a += abs(a_[keys[0]] - a_[keys[1]]) / max(a_[keys[0]], a_[keys[1]])
        rhm_x /= 6
        rhm_y /= 6
        rhm_a /= 6
    except:
        return 0.

    return 1 - (rhm_x + rhm_y + rhm_a) / 3


def get_balance_measure_w(h, w, df):
    mid_x, mid_y = w / 2, h / 2

    tmp = df.loc[df.c_x < mid_x]
    w_l = ((mid_x - tmp.c_x) * tmp.a * tmp.b).sum()

    tmp = df.loc[df.c_x > mid_x]
    w_r = ((tmp.c_x - mid_x) * tmp.a * tmp.b).sum()

    tmp = df.loc[df.c_y < mid_y]
    w_t = ((mid_y - tmp.c_y) * tmp.a * tmp.b).sum()

    tmp = df.loc[df.c_y > mid_y]
    w_b = ((tmp.c_y - mid_y) * tmp.a * tmp.b).sum()

    bm_v = abs(w_l - w_r) / max(w_l, w_r)
    bm_h = abs(w_t - w_b) / max(w_t, w_b)

    return 1 - (bm_v + bm_h) / 2


def get_equilibrium_measure_w(h, w, df):

    mid_x, mid_y = w/2, h/2

    em_x = (2/w)*np.average(df.c_x-mid_x, weights=df.a * df.b)
    em_y = (2/h)*np.average(df.c_y-mid_y, weights=df.a * df.b)

    return 1 - (abs(em_x) + abs(em_y)) / 2


def get_cohesion_measure_w(h, w, df):

    c = np.average(df.h/df.w, weights=df.a)/(h/w)

    return c if c<=1 else 1/c


def get_features(img, boxes):
    img_h, img_w = img.shape[:2]
    c_x, c_y, w, h, a, b = [], [], [], [], [], []

    for idx, box in boxes.iterrows():

        c_x.append((box.xmin + box.xmax) / 2)
        c_y.append((box.ymin + box.ymax) / 2)
        w.append(box.xmax - box.xmin)
        h.append(box.ymax - box.ymin)
        a.append(
            (box.xmax - box.xmin) * (box.ymax - box.ymin)
        )
        c = cv2.cvtColor(img[box.ymin:box.ymax, box.xmin:box.xmax], cv2.COLOR_RGB2GRAY).mean()
        b.append(255 - c)

    df = pd.DataFrame({'c_x': c_x, 'c_y': c_y, 'a': a, 'w': w, 'h': h, 'b': b})

    return {
        'balance': get_balance_measure(img_h, img_w, df),
        'equilibrium': get_equilibrium_measure(img_h, img_w, df),
        'symmetry': get_symmetry_measure(img_h, img_w, df),
        'sequence': get_sequence_measure(img_h, img_w, df),
        'cohesion': get_cohesion_measure(img_h, img_w, df),
        'unity': get_unity_measure(df),
        'proportion': get_proportion_measure(img_h, img_w, df),
        'simplicity': get_simplicity_measure(df),
        'density': get_density_measure(img_h, img_w, df),
        'regularity': get_regularity_measure(df),
        'economy': get_economy_measure(df),
        'homogeneity': get_homogeneity_measure(img_h, img_w, df),
        'rhythm': get_rhythm_measure(img_h, img_w, df),
        'w balance': get_balance_measure_w(img_h, img_w, df),
        'w equilibrium': get_equilibrium_measure_w(img_h, img_w, df),
        'w_cohesion': get_cohesion_measure_w(img_h, img_w, df)
    }

