import numpy as np
import scipy.linalg as linalg
from collections import deque
from tqdm import tqdm
from clustering import clustFunc
import logging


def get_neighbours(img, x, y, n, dims=3):
    """
    This function outputs nearby pixels for a given x,y and matrix img
    The window size is nxn
    n must be odd
    """
    assert n % 2 == 1
    if dims == 3:
        out = np.zeros((n, n, img.shape[2]))
    else:
        out = np.zeros((n, n))

    for j in range(y - n // 2, y + (n // 2) + 1):
        if j < 0:
            continue
        if j >= img.shape[0]:
            break
        for i in range(x - n // 2, x + n // 2 + 1):
            if i < 0:
                continue
            if i >= img.shape[1]:
                break
            if dims == 3:
                out[(n // 2) - (y - j), n // 2 - (x - i), :] = img[j, i, :]
            else:
                out[(n // 2) - (y - j), n // 2 - (x - i)] = img[j, i]
    return out


def create_gaussian(shape, std=8):
    """
    This is used to create weights which is distributed as bivariate gaussian
    The standard dev of 8 has been mentioned in the paper.
    """

    tmpx = (shape[0] - 1) / 2
    tmpy = (shape[1] - 1) / 2
    x_idx = np.arange(-tmpx, tmpx + 1)
    y_idx = np.arange(-tmpy, tmpy + 1)
    y_idx = y_idx.reshape(shape[1], 1)

    # bivariate gaussian
    gauss = np.exp(-(x_idx * x_idx + y_idx * y_idx) / (2 * std * std))

    # normalizing gaussian weights
    if np.sum(gauss) != 0:
        gauss /= np.sum(gauss)
    gauss /= np.max(gauss)
    return gauss


def solve_eqn9(inv_fcov, inv_bcov, alpha, std_c, curr_fbar, curr_bbar, C, fmax, bmax):
    """
    This function solves the 6x6 matrix mentioned as equation 9 in the paper
    If the det=0, then just return prev F(fmax) and prev B(bmax)
    """
    I = np.eye(3)

    # creating lhs of 6x6 matrix
    lhs = np.zeros((6, 6))
    lhs[0:3, 0:3] = inv_fcov + I * alpha * alpha * (1 / (std_c * std_c))
    lhs[0:3, 3:6] = I * alpha * (1 - alpha) * (1 / (std_c * std_c))
    lhs[3:6, 0:3] = I * alpha * (1 - alpha) * (1 / (std_c * std_c))
    lhs[3:6, 3:6] = inv_bcov + I * (1 - alpha) * (1 - alpha) * (1 / (std_c * std_c))

    # creating rhs
    rhs = np.zeros((6))
    tmp = np.dot(inv_fcov, curr_fbar) + C * alpha * (1 / (std_c * std_c))
    rhs[0:3] = np.dot(inv_fcov, curr_fbar) + C * alpha * (1 / (std_c * std_c))
    rhs[3:6] = np.dot(inv_bcov, curr_bbar) + C * (1 - alpha) * (1 / (std_c * std_c))

    # if singular just returns prev values
    if linalg.det(lhs) == 0:
        return fmax, bmax, alpha

    # else computes its inverse and multiplied by rhs to get F and B
    X = np.dot(linalg.inv(lhs), rhs)
    curr_F = np.maximum(0, np.minimum(1, X[0:3]))
    curr_B = np.maximum(0, np.minimum(1, X[3:6]))
    return curr_F, curr_B, alpha


def bayesian_matting(image, trimap1, trimap2, window_size):
    cnt_skipped = 0
    foreground = (trimap1 == 255) & (trimap2 == 255)  # both must be 255
    background = (trimap1 == 0) & (trimap2 == 0)  # both must be 0
    unknown = (trimap1 == 128) | (trimap2 == 128)  # all other marked unknown
    combined_trimap = foreground * 255 + unknown * 128

    # normalizing image matrix and alpha matte
    image = np.divide(image, 255)
    alpha_out = np.zeros((combined_trimap.shape))
    alpha_out[foreground] = 1
    alpha_out[unknown] = np.nan

    # masking the input image by extending the 2D mask to RGB axis
    foreground_col_pixels = image * np.repeat(foreground[:, :, np.newaxis], 3, axis=2)
    background_col_pixels = image * np.repeat(background[:, :, np.newaxis], 3, axis=2)
    # finding out x and y values that are unknown
    y_unknown, x_unknown = np.nonzero(unknown)

    # creating a queue becuase those pixels that were had lot of unknown pixels nearby
    # needs to be skipped and added to the end
    y_unknown = deque(y_unknown)
    x_unknown = deque(x_unknown)

    gauss_weights = create_gaussian((window_size, window_size), 8)

    pbar = tqdm(total=len(y_unknown), desc="Estimating unknown pixels")

    # looping for each unknown pixel
    while len(y_unknown) != 0:
        if cnt_skipped == len(y_unknown):
            break

        # taking out 1 pixel
        curr_x = x_unknown.popleft()
        curr_y = y_unknown.popleft()

        # getting neighbours of alpha, F and B
        alp = get_neighbours(alpha_out, curr_x, curr_y, window_size, 2)

        fg_neig = get_neighbours(foreground_col_pixels, curr_x, curr_y, window_size)
        fg_neig = fg_neig.reshape((fg_neig.shape[0] ** 2, 3))
        fg_weights = ((alp * alp) * gauss_weights).flatten()

        bg_neig = get_neighbours(background_col_pixels, curr_x, curr_y, window_size)
        bg_neig = bg_neig.reshape((bg_neig.shape[0] ** 2, 3))
        bg_weights = (((1 - alp) ** 2) * gauss_weights).flatten()

        # removing nan values from the created weight matrix
        fini = np.isfinite(fg_weights)
        fg_neig = fg_neig[fini, :]
        fg_weights = fg_weights[fini]

        fini = np.isfinite(bg_weights)
        bg_neig = bg_neig[fini, :]
        bg_weights = bg_weights[fini]

        # if there are less than a 15 known values in neighbourhood of current pixel, skipping that
        if (len(bg_weights) < 15 or len(fg_weights) < 15 or np.sum(fg_weights) == 0 or np.sum(bg_weights) == 0):
            x_unknown.append(curr_x)
            y_unknown.append(curr_y)
            cnt_skipped +=1
            # logging.warn(f"Point Skipped. Total = {cnt_skipped}")
            continue

        pbar.update(1)
        cnt_skipped = 0

        # creating clusters and finding mean and covariance for each of the cluster
        fbar_clusters, fcov_clusters = clustFunc(fg_neig, fg_weights)
        bbar_clusters, bcov_clusters = clustFunc(bg_neig, bg_weights)

        # current pixel values
        C = image[curr_y, curr_x]

        # tunable parameters. was taken 0.01 at most of the reference sites.
        std_c = 0.01
        tol = 1e-5

        # to keep track of max values of alpha, F and B at max log likelihood
        alpha_max = 0
        log_max = -np.inf
        fmax = np.zeros(3)
        bmax = np.zeros(3)

        # iterating to each combination of f cluster and b cluster
        for fcl_no in range(fbar_clusters.shape[0]):
            for bcl_no in range(bbar_clusters.shape[0]):
                curr_fbar = fbar_clusters[fcl_no, :]
                curr_fcov = fcov_clusters[fcl_no, :, :]

                # finding the inverse of covariance matrix.
                inv_fcov = linalg.inv(curr_fcov, check_finite=False)

                curr_bbar = bbar_clusters[bcl_no, :]
                curr_bcov = bcov_clusters[bcl_no, :, :]
                inv_bcov = linalg.inv(curr_bcov, check_finite=False)

                # solving matrix
                iter = 1
                maxiter = 100
                I = np.eye(3)
                alpha = np.nanmean(alp)

                while iter <= maxiter:
                    # equation 9 keeping alpha constant. optimizing F and B
                    curr_F, curr_B, alpha = solve_eqn9(
                        inv_fcov,
                        inv_bcov,
                        alpha,
                        std_c,
                        curr_fbar,
                        curr_bbar,
                        C,
                        fmax,
                        bmax,
                    )

                    # equation 10. keeping F and B constant. optimizing alpha
                    alpha = np.dot((C - curr_B), (curr_F - curr_B)) / (
                        linalg.norm((curr_F - curr_B)) ** 2
                    )
                    alpha = np.minimum(1, np.maximum(0, alpha))

                    # finding the log likelihood loss
                    vec = C - alpha * curr_F - (1 - alpha) * curr_B
                    log_c = -linalg.norm(vec) ** 2 / (std_c * std_c)
                    log_f = -(1 / 2) * np.dot(
                        np.dot((curr_F - curr_fbar).T, inv_fcov), (curr_F - curr_fbar)
                    )
                    log_b = -(1 / 2) * np.dot(
                        np.dot((curr_B - curr_bbar).T, inv_bcov), (curr_B - curr_bbar)
                    )

                    curr_log = log_c + log_f + log_b

                    if curr_log > log_max:
                        log_max = curr_log
                        alpha_max = alpha
                        fmax = curr_F
                        bmax = curr_B

                    if iter != 1 and abs(prev_log - curr_log) < tol:
                        break
                    prev_log = curr_log
                    iter += 1
        alpha_out[curr_y, curr_x] = alpha_max
        foreground_col_pixels[curr_y, curr_x, :] = fmax
        background_col_pixels[curr_y, curr_x, :] = bmax

    alpha_out[np.isnan(alpha_out)] = 0.5
    return combined_trimap, alpha_out
