import cv2
import numpy as np
import pandas as pd
import pyemd
import random
from . import colordict
import scipy.stats, scipy.spatial


def split_block(bw, n_block):

    sp1d = np.array_split(bw, n_block, axis=0)
    sp2d = [np.array_split(x, n_block, axis=1) for x in sp1d]

    return [x for sublist in sp2d for x in sublist]


def smallest_sublist(a_list, target):

    window_sum, left = 0, 0
    length = float('inf')
    final_left = 0
    final_right = len(a_list)

    for right in range(len(a_list)):
        window_sum += a_list[right]
        while window_sum > target and left <= right:
            if right - left + 1 < length:
                length = right - left + 1
                final_left = left
                final_right = right
            window_sum -= a_list[left]
            left = left + 1

    return [length, final_left, final_right]


def get_brightness(gray_img):

    return cv2.mean(gray_img)[0]


def get_contrast(gray_img):

    return cv2.meanStdDev(gray_img)[1][0][0] / get_brightness(gray_img)


def get_complexity(gray_img):

    return np.std(cv2.Laplacian(gray_img, cv2.CV_64F))


def get_sharpness(gray_img):

    h, w = gray_img.shape[:2]

    f = np.fft.fft2(gray_img / 255)
    f_shift = np.fft.fftshift(f)

    mags = np.abs(f_shift)
    n_mag = np.count_nonzero(mags > 5)

    return n_mag / (w * h)


def get_entropy(gray_img):

    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()

    return -np.sum(hist_norm * np.log2(hist_norm + 1e-7))


def get_color_variety(img):

    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    std_dev = np.std(lab_img, axis=(0, 1))

    return np.sqrt(np.sum(np.square(std_dev)))


def get_colorful(img):

    (R, G, B) = cv2.split(img.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    std_root = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    mean_root = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    return std_root + (0.3 * mean_root)


def get_colorful_emd(img):

    h, w = img.shape[:2]
    n = 4
    distribution1 = np.empty(n ** 3)
    distribution1.fill(1 / (n ** 3))
    rgb4 = img // (256 / n)

    centers, distribution2 = [], []

    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                rgb_center = [(128 / n + x * 256 / n) for x in [i, j, k]]
                rgb_center = np.array([[rgb_center]]).astype('uint8')
                luv_center = cv2.cvtColor(rgb_center, cv2.COLOR_RGB2Luv)
                centers.append(luv_center[0, 0, :].astype(float))

                n_pixels = np.sum(np.all(rgb4 == np.array([i, j, k]), axis=2))
                distribution2.append(n_pixels)
    distribution2 = np.array(distribution2).astype('float64')
    distribution2 = distribution2 / (w * h)

    n_total = n ** 3
    distance_matrix = np.zeros([n_total, n_total])

    for i in range(0, n_total):
        for j in range(0, n_total):
            distance_matrix[i, j] = scipy.spatial.distance.euclidean(centers[i], centers[j])

    em_dist = pyemd.emd(distribution1, distribution2, distance_matrix)

    return 128 - em_dist


def get_color_percentage(img):

    h, w = img.shape[:2]
    rgb = img.astype('int64')
    rgb8 = rgb // 8
    r8, g8, b8 = cv2.split(rgb8)
    one = r8 * (32 * 32) + g8 * 32 + b8
    color_dict = colordict.color_dict()

    tf = color_dict[one]
    unique, counts = [list(x) for x in np.unique(tf, return_counts=True)]
    color_percents = [0] * 11
    for i in range(0, 11):
        if i in unique:
            k = unique.index(i)
            color_percents[i] = counts[k] / (h * w)

    black, blue, brown, gray, green, orange, pink, purple, red, white, yellow = color_percents[:]

    all_colors = np.array([black, blue, brown, gray, green, orange, pink, purple, red, white, yellow])
    non_bw_colors = np.array([blue, brown, green, orange, pink, purple, red, yellow])
    if np.sum(non_bw_colors) > 0:
        non_bw_colors = non_bw_colors / np.sum(non_bw_colors)
        shannon_e = scipy.stats.entropy(non_bw_colors)
    else:
        shannon_e = np.nan
    if white > black:
        pixel_density = 1 - np.sum(white) / np.sum(all_colors)
    else:
        pixel_density = 1 - np.sum(black) / np.sum(all_colors)

    return shannon_e, pixel_density


def get_palette_ratio(img, n_colors=10):

    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    idx = (palette.sum(axis=1) < 700) & (palette.sum(axis=1) > 50)
    filtered = palette[idx]
    counts = counts[idx] / np.sum(counts[idx])
    idx = (counts >= 0.01)
    filtered = filtered[idx]

    if len(idx) > 2:
        int_col = filtered.astype(int)[:, 0] * 2 ** 16 + filtered.astype(int)[:, 1] * 2 ** 8 + filtered.astype(int)[:, 2]
        int_col.sort()
        r = [int_col[i - 1] / int_col[i] for i in range(1, len(int_col))]

        return np.average(r), np.std(r)

    else:
        return np.nan, np.nan


def get_complexity_edge(gray_img, otsu_ratio=0.5, gaussian_blur_kernel=(5, 5), n_random=1000):

    gray = cv2.GaussianBlur(gray_img, gaussian_blur_kernel, 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = cv2.Canny(gray, threshold1=(ret * otsu_ratio), threshold2=ret)

    edge_threshold = 0
    e_points = np.argwhere(edge > edge_threshold)

    h, w = edge.shape
    dia = (h ** 2 + w ** 2) ** 0.5
    e_total = len(e_points)
    e_density = e_total / (h * w)
    if e_total > 0:
        random.seed(42)
        if isinstance(n_random, int):
            e_points = random.sample(list(e_points), min(n_random, e_total))
        dists = scipy.spatial.distance.pdist(e_points, 'euclidean')
        e_dist = np.mean(dists) / dia
    else:
        e_dist = np.nan

    return e_density, e_dist


def get_saliency(img, saliency_type=0):

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create() if saliency_type == 0 else cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(img)

    return (saliency_map * 255).astype("uint8")


def get_complexity_saliency(gray_img):
    bw = get_saliency(gray_img, 0)
    h, w = bw.shape
    t_saliency = np.sum(bw)

    return t_saliency / (255 * w * h)


def get_complexity_saliency_consistency(gray_img, top_percent=0.6, n_block=5):

    bw1 = get_saliency(gray_img, 0)
    bw2 = get_saliency(gray_img, 1)

    blocks1 = split_block(bw1, n_block)
    sals1 = np.array([np.sum(x) for x in blocks1])
    blocks2 = split_block(bw2, n_block)
    sals2 = np.array([np.sum(x) for x in blocks2])

    ntop = int(n_block * n_block * top_percent)

    sorted_index1 = np.argsort(sals1)
    sorted_index2 = np.argsort(sals2)

    topindex1 = sorted_index1[-ntop:]
    topindex2 = sorted_index2[-ntop:]

    shareindex = np.intersect1d(topindex1, topindex2)

    return len(shareindex) / (n_block * n_block)


def get_ruleofthirds_centroid(gray_img):

    h, w = gray_img.shape
    dia = (w ** 2 + h ** 2) ** 0.5
    weight_all = np.sum(gray_img)

    com_x = np.sum([np.sum(gray_img[:, i]) * (i + 1) for i in range(0, w)]) / weight_all
    com_x_s = com_x / w

    com_y = np.sum([np.sum(gray_img[j, :]) * (j + 1) for j in range(0, h)]) / weight_all
    com_y_s = com_y / h

    dcm_x = abs(com_x_s - 1 / 2)
    dcm_y = abs(com_y_s - 1 / 2)

    return com_x_s, com_y_s, dcm_x, dcm_y


def get_features(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edge_density, edge_distance = get_complexity_edge(gray_img)
    shannon_e, pixel_density = get_color_percentage(img)
    com_x_s, com_y_s, dcm_x, dcm_y = get_ruleofthirds_centroid(gray_img)
    palette_ratio, palette_std = get_palette_ratio(img)

    return {
        'brightness': get_brightness(gray_img) / 256,
        'contrast': get_contrast(gray_img),
        'complexity': get_complexity(gray_img),
        'sharpness': get_sharpness(gray_img),
        'endge_density': edge_density,
        'edge_distance': edge_distance,
        'saliency': get_complexity_saliency(gray_img),
        'saliency_consistency': get_complexity_saliency_consistency(gray_img),
        'entropy': get_entropy(gray_img),
        'com_x_s': com_x_s,
        'com_y_s': com_y_s,
        'dcm_x': dcm_x,
        'dcm_y': dcm_y,
        'color_variety': get_color_variety(img),
        'colorful': get_colorful(img),
        'colorful_emd': get_colorful_emd(img),
        'color_entropy': shannon_e,
        'pixel_density': pixel_density,
        'palette_ratio': palette_ratio,
        'palette_std': palette_std
    }
