import numpy as np


class Permutation:
    def __init__(self, other=None, random_init=True, initial_mapping=None, alphabet_size=28):

        if isinstance(other, Permutation):
            self.inv_map = np.array(other.inv_map)
            self.p_tilde_val = other.p_tilde_val
            self.map = None

        elif initial_mapping is not None:
            self.map = initial_mapping
            self.inv_map = np.arange(alphabet_size)
            for j, v in enumerate(initial_mapping):
                self.inv_map[v] = j
            self.p_tilde_val = None

        elif random_init:
            self.map = None
            self.inv_map = np.arange(alphabet_size)
            np.random.shuffle(self.inv_map)
            self.p_tilde_val = None

        else:
            raise Exception(
                'Initialization is not random and no other initialization method was passed.')

    def p_tilde(self, y, M, P):

        if self.p_tilde_val:
            return self.p_tilde_val

        else:
            # Compute from scratch
            val = P[self.inv_map[y[0]]]
            prods = [M[self.inv_map[b], self.inv_map[a]]
                     for (a, b) in zip(y[:-1], y[1:])]
            val += np.sum(prods)

            return val

    def p_tilde_precomp(self, y, swap, M, P):
        # TODO use precomputed p_tilde_value and swap to compute p_tilde_val efficiently
        return self.p_tilde(y, M, P)

    def swap_idxs(self, swap):
        self.inv_map[swap] = self.inv_map[swap[::-1]]
        self.p_tilde_val = None
        self.map = None

    def translate(self, y):
        # if not self.map:
        #     self.map = np.array(self.inv_map)
        #     for j, v in enumerate(self.inv_map):
        #         self.map[v] = j
        return np.array([self.inv_map[v] for v in y])

    def __repr__(self) -> str:
        return str(self.inv_map)
