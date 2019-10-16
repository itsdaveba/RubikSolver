import numpy as np


class Rubik():
    def __init__(self, scramble=None):
        self.reset()
        self.apply_scramble(scramble)

    def is_solved(self):
        is_orientation = np.array_equal(self.orientation,
                                        np.zeros(20, dtype=int))
        is_permutation = np.array_equal(self.permutation,
                                        np.arange(20))
        return is_orientation and is_permutation

    def reset(self):
        self.orientation = np.zeros(20, dtype=int)
        self.permutation = np.arange(20)

    def apply_scramble(self, scramble):
        self.scramble = scramble
        if self.scramble is not None:
            for move in self.scramble.split():
                self.apply_move(move)

    def apply_move(self, move):
        if move == "U":
            self._rearrange([2, 0, 3, 1, 9, 11, 8, 10], 0)
        if move == "U2":
            self._rearrange([3, 2, 1, 0, 11, 10, 9, 8], 0)
        if move == "U'":
            self._rearrange([1, 3, 0, 2, 10, 8, 11, 9], 0)
        if move == "L":
            self._rearrange([3, 7, 1, 5, 15, 10, 18, 13], 2)
        if move == "L2":
            self._rearrange([7, 5, 3, 1, 18, 15, 13, 10], 0)
        if move == "L'":
            self._rearrange([5, 1, 7, 3, 13, 18, 10, 15], 2)
        if move == "F":
            self._rearrange([1, 5, 0, 4, 13, 8, 16, 12], 1)
        if move == "F2":
            self._rearrange([5, 4, 1, 0, 16, 13, 12, 8], 0)
        if move == "F'":
            self._rearrange([4, 0, 5, 1, 12, 16, 8, 13], 1)
        if move == "R":
            self._rearrange([4, 0, 6, 2, 12, 17, 9, 14], -2)
        if move == "R2":
            self._rearrange([6, 4, 2, 0, 17, 14, 12, 9], 0)
        if move == "R'":
            self._rearrange([2, 6, 0, 4, 14, 9, 17, 12], -2)
        if move == "B":
            self._rearrange([6, 2, 7, 3, 14, 19, 11, 15], -1)
        if move == "B2":
            self._rearrange([7, 6, 3, 2, 19, 15, 14, 11], 0)
        if move == "B'":
            self._rearrange([3, 7, 2, 6, 15, 11, 19, 14], -1)
        if move == "D":
            self._rearrange([5, 7, 4, 6, 18, 16, 19, 17], 0)
        if move == "D2":
            self._rearrange([7, 6, 5, 4, 19, 18, 17, 16], 0)
        if move == "D'":
            self._rearrange([6, 4, 7, 5, 17, 19, 16, 18], 0)

    def _rearrange(self, order, modifier):
        tmp1 = [self.orientation[i] for i in order]
        if modifier != 0:
            tmp1 = np.concatenate((
                [(t + (1 if i == 0 or i == 3 else -1) * np.sign(modifier)) % 3
                 for t, i in zip(tmp1[:4], range(4))],
                [(t + (1 if modifier % 2 == 1 else 0)) % 2
                 for t, i in zip(tmp1[4:], range(4))]))
        tmp2 = [self.permutation[i] for i in order]
        j = 0
        order.sort()
        for i in order:
            self.orientation[i] = tmp1[j]
            self.permutation[i] = tmp2[j]
            j += 1
