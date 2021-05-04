from os import access
import numpy as np
from numpy.random import default_rng
from model import Permutation
from matplotlib import pyplot as plt
from tqdm import tqdm

# def decode(ciphertext: str, has_breakpoint: bool) -> str:

#     plaintext = ciphertext  # Replace with your code

#     return plaintext


def decode(y: np.array, P: np.array, M: np.array, x: np.array = None, has_breakpoint: bool = False, max_sample_size: int = 3000, max_iter_count: int = 5000, alphabet_size: int = 28, p_val_store: list = None, accept_store: list = None, accuracy_store: list = None, initial_mapping: np.array = None) -> str:

    if initial_mapping is not None:
        f_0 = Permutation(initial_mapping=initial_mapping)
    else:
        f_0 = Permutation(random_init=True)
    f_best = None
    accepted_count = 0
    iter_count = 0
    rng = default_rng()

    # while accepted_count < max_sample_size or iter_count > max_iter_count:
    for iter_count in tqdm(range(max_iter_count)):

        if accepted_count > max_sample_size:
            break

        swap_idxs = rng.choice(alphabet_size, 2, replace=False)
        f_prime = Permutation(f_0)
        f_prime.swap_idxs(swap_idxs)

        p_tilde_f_val = f_0.p_tilde(y, M, P)
        p_tilde_f_prime_val = f_prime.p_tilde(y, M, P)

        # accept if p_tilde_f_val or p_tilde_f_prime_val are zero
        accept = False

        if p_tilde_f_val > -np.inf:
            p_tilde_ratio_log = p_tilde_f_prime_val - p_tilde_f_val
            p_tilde_ratio_log = min(0, p_tilde_ratio_log)
            a_f_fp = 2.0**(p_tilde_ratio_log)
            r = np.random.rand()
            if r <= a_f_fp:
                accept = True
        else:
            accept = True

        if accept:
            f_0 = f_prime
            accepted_count += 1
            if not f_best:
                f_best = f_0
            elif f_best.p_tilde(y, M, P) < f_0.p_tilde(y, M, P):
                f_best = Permutation(f_0)

                if p_val_store is not None:
                    p_val_store.append((iter_count, f_best.p_tilde(y, M, P)))

        if (x is not None) and (accuracy_store is not None):
            x_decode = f_best.translate(y)
            acc_rate = (x == x_decode).mean()
            accuracy_store.append(acc_rate)

        if accept_store is not None:
            accept_store.append(accept)
        iter_count += 1

    return f_best.translate(y)
