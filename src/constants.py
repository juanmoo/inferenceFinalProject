# '''
# Constants File
# '''

# import numpy as np
# import os


# dir_path = os.path.dirname(os.path.abspath(__file__))


# # Alphabet
# A = np.genfromtxt(os.path.join(dir_path, '../data/alphabet.csv'),
#                   delimiter=',', dtype=str)
# A_map = {c: j for j, c in enumerate(A)}

# log_P = np.log2(np.genfromtxt(
#     os.path.join(dir_path, '../data/letter_probabilities.csv'), delimiter=','))

# log_M = np.log2(np.genfromtxt(os.path.join(
#     dir_path, '../data/letter_transition_matrix.csv'), delimiter=','))
# freq_eng = np.argsort(log_P)[::-1]


# # Ciphertext samples
# ciphertext_mean_length = 10000
# ciphertext_std_length = 2000

# # Decode configs
# max_iters = 10000

# bp_std = 20
