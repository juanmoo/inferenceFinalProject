import os
from time import time
import numpy as np
# import multiprocessing

# Initialization
rng = np.random.default_rng()

# Constants
dir_path = os.path.dirname(os.path.abspath(__file__))

# External Files
is_submission = True
path_prefix = '' if is_submission else '..'
A = np.genfromtxt(os.path.join(path_prefix, 'data/alphabet.csv'),
                  delimiter=',', dtype=str)
A_map = {c: j for j, c in enumerate(A)}
log_P = np.log2(np.genfromtxt(
    os.path.join(path_prefix, 'data/letter_probabilities.csv'), delimiter=','))
log_M = np.log2(np.genfromtxt(os.path.join(path_prefix,
                                           'data/letter_transition_matrix.csv'), delimiter=','))
freq_eng = np.argsort(log_P)[::-1]


# Ciphertext samples
ciphertext_mean_length = 10000
ciphertext_std_length = 2000

# Decode configs
max_iters = 10000
bp_std = 20


##############


def decode(ciphertext: str, has_breakpoint: bool, logger=None) -> str:

    plaintext: str = None

    # start_time = time()
    if has_breakpoint:

        # num_reps = 1
        num_reps = 200
        arg_list = [(ciphertext, logger) for _ in range(num_reps)]
        runs = [decode_breakpoint(a) for a in arg_list]
        # with multiprocessing.get_context('spawn').Pool(4) as p:
        # runs = p.map(decode_breakpoint, arg_list)
        plaintext, _, logger_ret = max(runs, key=lambda e: e[1])

    else:
        # num_reps = 1
        num_reps = 100
        arg_list = [(ciphertext, logger) for _ in range(num_reps)]
        # with multiprocessing.get_context('spawn').Pool(4) as p:
        # runs = p.map(decode_no_breakpoint, arg_list)
        runs = [decode_no_breakpoint(a) for a in arg_list]
        plaintext, _, logger_ret = max(runs, key=lambda e: e[1])

    # ellapsed_time = time() - start_time

    # if logger:
    #     logger.update(logger_ret)
    #     logger['time'] = ellapsed_time

    return plaintext


def compute_log_likelihood(seq, f):

    if len(seq) > 0:
        val = log_P[f[seq[0]]]
        val += log_M[f[seq[1:]], f[seq[:-1]]].sum()
    else:
        val = 0

    return val


def compute_log_likelihood_swap(seq, f, prev_ll, swap):
    if len(seq) > 0:
        val = prev_ll

        # count instances of i and j
        i, j = swap
        unique, counts = np.unique(seq, return_counts=True)
        freqs = dict(zip(unique, counts))

        val -= freqs[i] * f[i]

    else:
        return 0


'''
BREAKPOINT
'''


def mh_step_bp(f1, f2, b, seq, f1_ll=None, f2_ll=None):

    # Sample F1
    seq1 = seq[:b]
    swap_idxs = rng.choice(len(A), 2, replace=False)
    f1_prime = np.array(f1, copy=True)
    f1_prime[swap_idxs] = f1_prime[swap_idxs[::-1]]

    f1_ll = compute_log_likelihood(seq1, f1)
    f1p_ll = compute_log_likelihood(seq1, f1_prime)

    accept = False

    if f1_ll <= -np.inf:
        accept = True
    elif f1p_ll > f1_ll:
        accept = True
    else:
        ratio_log = f1p_ll - f1_ll
        accept_prob = 2.0**ratio_log
        accept = np.random.rand() < accept_prob

    if accept:
        f1 = f1_prime
        ll1 = f1p_ll
    else:
        ll1 = f1_ll

    # Sample F1
    seq2 = seq[b:]
    swap_idxs = rng.choice(len(A), 2, replace=False)
    f2_prime = np.array(f2, copy=True)
    f2_prime[swap_idxs] = f2_prime[swap_idxs[::-1]]

    f2_ll = compute_log_likelihood(seq2, f2)
    f2p_ll = compute_log_likelihood(seq2, f2_prime)

    accept = False

    if f2_ll <= -np.inf:
        accept = True
    elif f2p_ll > f2_ll:
        accept = True
    else:
        ratio_log = f2p_ll - f2_ll
        accept_prob = 2.0**ratio_log
        accept = np.random.rand() < accept_prob

    if accept:
        f2 = f2_prime
        ll2 = f2p_ll
    else:
        ll2 = f2_ll

    # Sample b
    b_prime = int(np.random.normal(b, bp_std))
    b_prime = np.clip(b_prime, 0, len(seq))

    b_ll = ll1 + ll2
    b_prime_ll = compute_log_likelihood(
        seq[:b_prime], f1) + compute_log_likelihood(seq[b_prime:], f2)

    accept = False

    if b_ll <= -np.inf:
        accept = True
    elif b_prime_ll > b_ll:
        accept = True
    else:
        ratio_log = b_prime_ll - b_ll
        accept_prob = 2.0**ratio_log
        accept = np.random.rand() < accept_prob

    if accept:
        b = b_prime
        b_ll = b_prime_ll
    else:
        b_ll = b_ll

    return (f1, f2, b, ll1, ll2, b_ll)


def decode_breakpoint(args) -> str:

    # print('Decode with breakpoint')

    ciphertext, logger = args

    if logger:
        logger = logger.copy()

    # Initialization
    N = len(ciphertext)
    Y = np.array([A_map[char] for char in ciphertext])

    # f1 and f2 are random permutations
    f1 = np.arange(len(A))
    np.random.shuffle(f1)
    f2 = np.arange(len(A))
    np.random.shuffle(f2)

    # breakpoint initialized to middle of ciphertext
    b = N//2

    # best vals
    best_vals = None
    best_ll = -np.inf

    ll_q = []
    window = 500
    threshold = 0.003
    # threshold = 0.001

    for iter_idx in range(max_iters):

        # mh_step_bp(f1, f2, b, seq, f1_ll=None, f2_ll=None)
        vals = mh_step_bp(f1, f2, b, Y)
        f1, f2, b, f1_ll, f2_ll, b_ll = vals

        if best_ll < b_ll:
            best_vals = vals
            best_ll = b_ll

        if logger is not None:
            if 'total_ll' in logger:
                # print('logging!')
                logger['total_ll'].append(best_ll)

        ll_q.append(best_ll)

        if (best_ll > -np.inf) and (iter_idx > window):
            prev = np.clip(ll_q[iter_idx - window], -1e99, 1e99)
            curr = np.clip(best_ll, -1e99, 1e99)
            p_diff = -(curr - prev)/prev

            if p_diff <= threshold:
                break

    # decode Y using best f1, f2, and b
    if best_vals:
        best_f1, best_f2, best_b, _, _, best_ll = best_vals
        x_decode = [best_f1[y_i]
                    for y_i in Y[:best_b]] + [best_f2[y_i] for y_i in Y[best_b:]]
        plaintext = ''.join([A[i] for i in x_decode])
    else:
        plaintext = ciphertext
        best_ll = -np.inf

    return plaintext, best_ll, logger


'''
NO BREAKPOINT
'''


def mh_step_nbp(f1, seq, f1_ll=None):

    swap_idxs = rng.choice(len(A), 2, replace=False)
    f1_prime = np.array(f1, copy=True)
    f1_prime[swap_idxs] = f1_prime[swap_idxs[::-1]]

    f1_ll = compute_log_likelihood(seq, f1)
    f1p_ll = compute_log_likelihood(seq, f1_prime)

    accept = False

    if f1_ll <= -np.inf:
        accept = True
    elif f1p_ll > f1_ll:
        accept = True
    else:
        ratio_log = f1p_ll - f1_ll
        accept_prob = 2.0**ratio_log
        accept = np.random.rand() < accept_prob

    if accept:
        f1 = f1_prime
        ll1 = f1p_ll
    else:
        ll1 = f1_ll

    return f1, ll1


def decode_no_breakpoint(args) -> str:

    # print('Decode no with breakpoint')

    ciphertext, logger = args
    if logger:
        logger = logger.copy()

    # Initialization
    N = len(ciphertext)
    Y = np.array([A_map[char] for char in ciphertext])

    # f is random perm
    f = np.arange(len(A))
    # np.random.shuffle(f)

    # f matches empirical freq order to english order
    unique, counts = np.unique(Y, return_counts=True)
    freq_dict = {symb: count for (symb, count) in zip(unique, counts)}
    emp_freq = np.array(
        sorted(range(len(A)), key=lambda char: -1 * freq_dict.get(char, 0)))
    # freq_ct = unique[np.argsort(counts)[::-1]]
    f[freq_eng] = emp_freq

    # best vals
    best_vals = None
    best_ll = -np.inf

    ll_q = []
    window = 500
    threshold = 0.003
    # threshold = 0.001

    should_terminate = False
    for iter_idx in range(max_iters):

        vals = mh_step_nbp(f, Y)
        f, ll = vals

        if logger is not None:
            if 'total_ll' in logger:
                logger['total_ll'].append(ll)

        if ll > best_ll:
            best_vals = vals
            best_ll = ll

        if logger is not None:
            if 'total_ll' in logger:
                logger['total_ll'].append(best_ll)

        ll_q.append(best_ll)

        if (best_ll > -np.inf) and (iter_idx > window):
            prev = np.clip(ll_q[iter_idx - window], -1e99, 1e99)
            curr = np.clip(best_ll, -1e99, 1e99)
            p_diff = -(curr - prev)/prev

            if p_diff <= threshold:
                break

    if best_vals:
        best_f, ll = best_vals
        x_decode = [best_f[y_i]for y_i in Y]
        plaintext = ''.join([A[i] for i in x_decode])
    else:
        plaintext = ciphertext, -np.inf, logger

    return plaintext, ll, logger
