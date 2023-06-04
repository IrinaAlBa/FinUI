import numpy as np


def get_loc_layout_features(img_h, img_w, loc, objects, txt_objects):
    objects['cx'] = (objects.xmin + objects.xmax)/2
    objects['cy'] = (objects.ymin + objects.ymax)/2
    txt_objects['cx'] = (txt_objects.xmin + txt_objects.xmax)/2
    txt_objects['cy'] = (txt_objects.ymin + txt_objects.ymax)/2

    if loc == 0:
        xmin, ymin = 0, 0
        xmax, ymax = img_w, img_h
        w, h = img_w, img_h
    elif (loc > 0) & (loc < 10):
        loc -= 1
        (i, j) = 1 + np.trunc(loc / 3).astype(int), 1 + loc % 3
        xmin = int((i - 1) * img_w / 3)
        xmax = int(i * img_w / 3)
        ymin = int((j - 1) * img_h / 3)
        ymax = int(j * img_h / 3)
        w, h = img_w / 3, img_h / 3
    else:
        raise ValueError('loc is an int from 0 to 9')

    o = objects.loc[
        (objects.cx > xmin) & (objects.cx <= xmax) & (objects.cy > ymin) & (objects.cy <= ymax)
        ]
    t = txt_objects.loc[
        (txt_objects.cx > xmin) & (txt_objects.cx <= xmax) & (txt_objects.cy > ymin) & (txt_objects.cy <= ymax)
        ]

    # all objects
    f = [len(o) + len(t)]
    cols = ['n_all']
    
    if (len(o) + len(t)) > 0:
        w_ = np.append(o.xmax - o.xmin, t.xmax - t.xmin)
        h_ = np.append(o.ymax - o.ymin, t.ymax - t.ymin)
        f += [np.average(w_) / w, np.std(w_) / w, np.average(h_) / h, np.std(h_) / h]
    else:
        f += [0., 0., 0., 0.]
    cols += ['w_all', 'sw_all', 'h_all', 'sh_all']

    # text objects
    f.append(len(t))
    cols.append('n_txt')
    
    if len(t)>0:
        w_ = t.xmax - t.xmin
        h_ = t.ymax - t.ymin
        f += [np.average(w_) / w, np.std(w_) / w, np.average(h_) / h, np.std(h_) / h]
    else:
        f += [0., 0., 0., 0.]
    cols += ['w_txt', 'sw_txt', 'h_txt', 'sh_txt']

    # other objects
    for c in ['image', 'form', 'chart', 'table', 'block']:
        if c == 'image':
            o_ = o.loc[o['class'].isin(['image', 'icon', 'logo'])]
        elif c == 'form':
            continue
            # o_ = o.loc[o['class'].isin(['form', 'button'])]
        else:
            o_ = o.loc[o['class'] == c]

        f += [len(o_)]
        cols.append(f'n_{c}')

        if len(o_) > 0:
            w_ = o_.xmax - o_.xmin
            h_ = o_.ymax - o_.ymin
            f += [np.average(w_) / w, np.std(w_) / w, np.average(h_) / h, np.std(h_) / h]
        else:
            f += [0., 0., 0., 0.]
        cols += [f'w_{c}', f'sw_{c}', f'h_{c}', f'sh_{c}']

    return f, cols


def get_layout_features(img_h, img_w, objects, txt_objects):
    f, cols = [], []
    
    for loc in range(10):
        f_loc, cols_loc = get_loc_layout_features(img_h, img_w, loc, objects, txt_objects)
        cols_loc = [f'{c}_{loc}' for c in cols_loc]
        f += f_loc
        cols += cols_loc

    return f, cols
