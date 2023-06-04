import re
import numpy as np
import pandas as pd
import cv2
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from . import imtools
import warnings

warnings.filterwarnings('ignore')


def filter_outliers(df, c='box_h'):

    z = np.abs(scipy.stats.zscore(df[c]))

    return df.loc[z < 3]


def get_density(df, coe=1.):

    df = filter_outliers(df)

    return (coe ** 2) * df.n_char / (df.box_h * df.box_w)


def get_word_count(df):

    words = ' '.join(
        df.loc[df.n_let > 3]['text'].apply(
            lambda x: re.sub('[^A-Za-z]+', ' ', str(x).lower())
        ).to_list()
    )
    words = " ".join(words.split())

    return len([w for w in words.split(' ') if len(w) > 3])


def get_km_score(data, center):
    km = KMeans(n_clusters=center)
    model = km.fit_predict(data)
    score = davies_bouldin_score(data, model)

    return score


def get_box_ratios(df):

    df = filter_outliers(df)
    data = df.box_h.values.reshape(-1, 1)

    scores = []
    centers = list(range(3, 8))
    for center in centers:
        scores.append(get_km_score(data, center))

    nc = centers[scores.index(min(scores))]
    km = KMeans(n_clusters=nc).fit(data)
    centers = km.cluster_centers_

    x = np.sort(centers.flatten())

    return [x[i - 1] / x[i] for i in range(1, len(x))]


def get_features(img, df):

    img_h, img_w = img.shape[:2]
    coe = max(img_h, img_w) / 1200
    
    df['box_w'] = df.xmax - df.xmin
    df['box_h'] = df.ymax - df.ymin
    df['box_x'] = df.xmin
    df['box_y'] = df.ymin
    df = df.rename(columns={'txt': 'text'})
    df = df.loc[(df.xmin >= 0) & (df.ymin >= 0) & (df.xmax <= img_w) & (df.ymax <= img_h) & (df.box_w > 5) & (df.box_h > 5)]
    
    sharpness, contrast = [], []
    for idx, row in df.iterrows():
        sub_img = cv2.cvtColor(img[row.ymin:row.ymax, row.xmin:row.xmax], cv2.COLOR_RGB2GRAY)
        sharpness.append(imtools.get_sharpness(sub_img))
        contrast.append(imtools.get_contrast(sub_img))
            
    df['sharpness'] = sharpness
    df['contrast'] = contrast

    df['n_char'] = df['text'].str.len()
    df['n_let'] = df['text'].apply(lambda x: len(re.sub('[^A-Za-z]+', '', str(x))))
    df['n_num'] = df['text'].apply(lambda x: len(re.sub('[^0-9]', '', str(x))))

    df = filter_outliers(df)
    density = get_density(df, coe)
    box_ratios = get_box_ratios(df)

    return {
        'n_char': df.n_char.sum(),
        'n_box': len(df),
        'n_word': get_word_count(df),
        'num_ratio': df.n_num.sum() / df.n_char.sum(),
        'contrast': df.contrast.mean(),
        'contrast_std': df.contrast.std(),
        'sharpness': df.sharpness.mean(),
        'sharpness_std': df.sharpness.std(),
        'coverage': (df.box_h * df.box_w).sum() / (img_h * img_w),
        'h_box': df.box_h.mean() / coe,
        'h_box_std': df.box_h.std() / coe,
        'density': np.average(density),
        'density_std': np.std(density),
        'r_box': np.average(box_ratios),
        'r_box_std': np.std(box_ratios)
    }
