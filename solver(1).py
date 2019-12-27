import numpy as np
from numpy import random
from numpy import math
import pickle


class Rubik():
    def __init__(self, scramble=None):
        self.reset()
        self.apply_scramble(scramble)
        self.load_tables()

    def reset(self):
        self.position = dict(orientation=np.zeros(20, dtype=np.int8),
                             permutation=np.arange(20, dtype=np.int8))
        self.solution = []

    def apply_scramble(self, scramble):
        if scramble is not None:
            for move in scramble.split():
                self.position = self.result(self.position, move)

    def result(self, position, move):
        if move == "U":
            return self._rearrange(position, [2, 0, 3, 1, 9, 11, 8, 10], 0)
        elif move == "U2":
            return self._rearrange(position, [3, 2, 1, 0, 11, 10, 9, 8], 0)
        elif move == "U'":
            return self._rearrange(position, [1, 3, 0, 2, 10, 8, 11, 9], 0)
        elif move == "L":
            return self._rearrange(position, [3, 7, 1, 5, 19, 17, 10, 14], 2)
        elif move == "L2":
            return self._rearrange(position, [7, 5, 3, 1, 14, 10, 19, 17], 0)
        elif move == "L'":
            return self._rearrange(position, [5, 1, 7, 3, 17, 19, 14, 10], 2)
        elif move == "F":
            return self._rearrange(position, [1, 5, 0, 4, 17, 16, 8, 12], 1)
        elif move == "F2":
            return self._rearrange(position, [5, 4, 1, 0, 12, 8, 17, 16], 0)
        elif move == "F'":
            return self._rearrange(position, [4, 0, 5, 1, 16, 17, 12, 8], 1)
        elif move == "R":
            return self._rearrange(position, [4, 0, 6, 2, 16, 18, 13, 9], -2)
        elif move == "R2":
            return self._rearrange(position, [6, 4, 2, 0, 13, 9, 18, 16], 0)
        elif move == "R'":
            return self._rearrange(position, [2, 6, 0, 4, 18, 16, 9, 13], -2)
        elif move == "B":
            return self._rearrange(position, [6, 2, 7, 3, 18, 19, 15, 11], -1)
        elif move == "B2":
            return self._rearrange(position, [7, 6, 3, 2, 15, 11, 19, 18], 0)
        elif move == "B'":
            return self._rearrange(position, [3, 7, 2, 6, 19, 18, 11, 15], -1)
        elif move == "D":
            return self._rearrange(position, [5, 7, 4, 6, 14, 12, 15, 13], 0)
        elif move == "D2":
            return self._rearrange(position, [7, 6, 5, 4, 15, 14, 13, 12], 0)
        elif move == "D'":
            return self._rearrange(position, [6, 4, 7, 5, 13, 15, 12, 14], 0)

    def _rearrange(self, position, order, modifier):
        orientation = [position["orientation"][i] for i in order]
        if modifier != 0:
            orientation = np.concatenate((
                [(t + (1 if i == 0 or i == 3 else -1) * np.sign(modifier)) % 3
                 for t, i in zip(orientation[:4], range(4))],
                [(t + (1 if modifier % 2 == 1 else 0)) % 2
                 for t in orientation[4:]]))
        permutation = [position["permutation"][i] for i in order]
        j = 0
        order.sort()
        new_position = dict(orientation=np.copy(position["orientation"]),
                            permutation=np.copy(position["permutation"]))
        for i in order:
            new_position["orientation"][i] = orientation[j]
            new_position["permutation"][i] = permutation[j]
            j += 1
        return new_position

    def load_tables(self):
        try:
            f = open('corner_orientation_transition_table.pckl', 'rb')
            self.corner_orientation_transition_table = pickle.load(f)
            f.close()
        except IOError:
            self.corner_orientation_transition_table =\
                self.get_corner_orientation_transition_table()
            f = open('corner_orientation_transition_table.pckl', 'wb')
            pickle.dump(self.corner_orientation_transition_table, f)
            f.close()

        try:
            f = open('corner_permutation_transition_table.pckl', 'rb')
            self.corner_permutation_transition_table = pickle.load(f)
            f.close()
        except IOError:
            self.corner_permutation_transition_table =\
                self.get_corner_permutation_transition_table()
            f = open('corner_permutation_transition_table.pckl', 'wb')
            pickle.dump(self.corner_permutation_transition_table, f)
            f.close()

        try:
            f = open('edge_orientation_transition_table.pckl', 'rb')
            self.edge_orientation_transition_table = pickle.load(f)
            f.close()
        except IOError:
            self.edge_orientation_transition_table =\
                self.get_edge_orientation_transition_table()
            f = open('edge_orientation_transition_table.pckl', 'wb')
            pickle.dump(self.edge_orientation_transition_table, f)
            f.close()

        try:
            f = open('edge_combination_transition_table.pckl', 'rb')
            self.edge_combination_transition_table = pickle.load(f)
            f.close()
        except IOError:
            self.edge_combination_transition_table =\
                self.get_edge_combination_transition_table()
            f = open('edge_combination_transition_table.pckl', 'wb')
            pickle.dump(self.edge_combination_transition_table, f)
            f.close()

        try:
            f = open('corner_permutation_phase2_transition_table.pckl', 'rb')
            self.corner_permutation_phase2_transition_table = pickle.load(f)
            f.close()
        except IOError:
            self.corner_permutation_phase2_transition_table =\
                self.get_corner_permutation_phase2_transition_table()
            f = open('corner_permutation_phase2_transition_table.pckl', 'wb')
            pickle.dump(self.corner_permutation_phase2_transition_table, f)
            f.close()

        try:
            f = open('edge1_permutation_phase2_transition_table.pckl', 'rb')
            self.edge1_permutation_phase2_transition_table = pickle.load(f)
            f.close()
        except IOError:
            self.edge1_permutation_phase2_transition_table =\
                self.get_edge1_permutation_phase2_transition_table()
            f = open('edge1_permutation_phase2_transition_table.pckl', 'wb')
            pickle.dump(self.edge1_permutation_phase2_transition_table, f)
            f.close()

        try:
            f = open('edge2_permutation_phase2_transition_table.pckl', 'rb')
            self.edge2_permutation_phase2_transition_table = pickle.load(f)
            f.close()
        except IOError:
            self.edge2_permutation_phase2_transition_table =\
                self.get_edge2_permutation_phase2_transition_table()
            f = open('edge2_permutation_phase2_transition_table.pckl', 'wb')
            pickle.dump(self.edge2_permutation_phase2_transition_table, f)
            f.close()

        try:
            f = open('corner_orientation_table.pckl', 'rb')
            self.corner_orientation_table = pickle.load(f)
            f.close()
        except IOError:
            self.corner_orientation_table = self.get_corner_orientation_table()
            f = open('corner_orientation_table.pckl', 'wb')
            pickle.dump(self.corner_orientation_table, f)
            f.close()

        try:
            f = open('corner_permutation_table.pckl', 'rb')
            self.corner_permutation_table = pickle.load(f)
            f.close()
        except IOError:
            self.corner_permutation_table = self.get_corner_permutation_table()
            f = open('corner_permutation_table.pckl', 'wb')
            pickle.dump(self.corner_permutation_table, f)
            f.close()

        try:
            f = open('edge_orientation_table.pckl', 'rb')
            self.edge_orientation_table = pickle.load(f)
            f.close()
        except IOError:
            self.edge_orientation_table = self.get_edge_orientation_table()
            f = open('edge_orientation_table.pckl', 'wb')
            pickle.dump(self.edge_orientation_table, f)
            f.close()

        try:
            f = open('edge_combination_table.pckl', 'rb')
            self.edge_combination_table = pickle.load(f)
            f.close()
        except IOError:
            self.edge_combination_table = self.get_edge_combination_table()
            f = open('edge_combination_table.pckl', 'wb')
            pickle.dump(self.edge_combination_table, f)
            f.close()

        try:
            f = open('phase1_tables.pckl', 'rb')
            self.phase1_tables = pickle.load(f)
            f.close()
        except IOError:
            self.phase1_tables = self.get_phase1_tables()
            f = open('phase1_tables.pckl', 'wb')
            pickle.dump(self.phase1_tables, f)
            f.close()

        try:
            f = open('phase2_tables.pckl', 'rb')
            self.phase2_tables = pickle.load(f)
            f.close()
        except IOError:
            self.phase2_tables = self.get_phase2_tables()
            f = open('phase2_tables.pckl', 'wb')
            pickle.dump(self.phase2_tables, f)
            f.close()

    def get_corner_orientation_transition_table(self):
        print('Creating corner orientation transition table...')
        table = np.zeros((2187, 18), dtype=np.uint16)
        for p in range(2187):
            for q, move in zip(range(18), self.available_moves(None)):
                table[p][q], _, _, _ = self.position_to_index(
                    self.result(self.index_to_position(p, 0, 0, 0), move))
        return table

    def get_corner_permutation_transition_table(self):
        print('Creating corner permutation transition table...')
        table = np.zeros((40320, 18), dtype=np.uint16)
        for p in range(40320):
            for q, move in zip(range(18), self.available_moves(None)):
                _, table[p][q], _, _ = self.position_to_index(
                    self.result(self.index_to_position(0, p, 0, 0), move))
        return table

    def get_edge_orientation_transition_table(self):
        print('Creating edge orientation transition table...')
        table = np.zeros((2048, 18), dtype=np.uint16)
        for p in range(2048):
            for q, move in zip(range(18), self.available_moves(None)):
                _, _, table[p][q], _ = self.position_to_index(
                    self.result(self.index_to_position(0, 0, p, 0), move))
        return table

    def get_edge_combination_transition_table(self):
        print('Creating edge combination transition table...')
        table = np.zeros((495, 18), dtype=np.uint16)
        for p in range(495):
            for q, move in zip(range(18), self.available_moves(None)):
                table[p][q] = self.edge_combination_to_index(
                    self.result(self.index_to_edge_combination(p), move))
        return table

    def get_corner_orientation_table(self):
        print('Creating corner orientation table...')
        table = np.zeros(2187, dtype=np.int8)
        table -= 1
        table[0] = 0
        length = 0
        while True:
            counter = 0
            for p in range(2187):
                if table[p] == length:
                    for i in range(18):
                        q = self.corner_orientation_transition_table[p][i]
                        if table[q] == -1:
                            counter += 1
                            table[q] = length + 1
            length += 1
            print(str(counter) + " positions at distance " + str(length))
            if counter == 0:
                break
        return table

    def get_corner_permutation_table(self):
        print('Creating corner permutation table...')
        table = np.zeros(40320, dtype=np.int8)
        table -= 1
        table[0] = 0
        length = 0
        while True:
            counter = 0
            for p in range(40320):
                if table[p] == length:
                    for i in range(18):
                        q = self.corner_permutation_transition_table[p][i]
                        if table[q] == -1:
                            counter += 1
                            table[q] = length + 1
            length += 1
            print(str(counter) + " positions at distance " + str(length))
            if counter == 0:
                break
        return table

    def get_edge_orientation_table(self):
        print('Creating edge orientation table...')
        table = np.zeros(2048, dtype=np.int8)
        table -= 1
        table[0] = 0
        length = 0
        while True:
            counter = 0
            for p in range(2048):
                if table[p] == length:
                    for i in range(18):
                        q = self.edge_combination_transition_table[p][i]
                        if table[q] == -1:
                            counter += 1
                            table[q] = length + 1
            length += 1
            print(str(counter) + " positions at distance " + str(length))
            if counter == 0:
                break
        return table

    def get_edge_combination_table(self):
        print('Creating edge combination table...')
        table = np.zeros(495, dtype=np.int8)
        table -= 1
        table[0] = 0
        length = 0
        while True:
            counter = 0
            for p in range(495):
                if table[p] == length:
                    for i in range(18):
                        q = self.edge_combination_transition_table[p][i]
                        if table[q] == -1:
                            counter += 1
                            table[q] = length + 1
            length += 1
            print(str(counter) + " positions at distance " + str(length))
            if counter == 0:
                break
        return table

    def get_phase1_tables(self):
        print('Creating phase 1 tables...')
        print('Table1')
        table1 = np.zeros((2187, 2048), dtype=np.int8)
        table1 -= 1
        table1[0][0] = 0
        length = 0
        while True:
            counter = 0
            for p in range(2187):
                for q in range(2048):
                    if table1[p][q] == length:
                        for i in range(18):
                            s = self.corner_orientation_transition_table[p][i]
                            t = self.edge_orientation_transition_table[q][i]
                            if table1[s][t] == -1:
                                counter += 1
                                table1[s][t] = length + 1
            length += 1
            print(str(counter) + " positions at distance " + str(length))
            if counter == 0:
                break

        print('Table2')
        table2 = np.zeros((2187, 495), dtype=np.int8)
        table2 -= 1
        table2[0][0] = 0
        length = 0
        while True:
            counter = 0
            for p in range(2187):
                for r in range(495):
                    if table2[p][r] == length:
                        for i in range(18):
                            s = self.corner_orientation_transition_table[p][i]
                            u = self.edge_combination_transition_table[r][i]
                            if table2[s][u] == -1:
                                counter += 1
                                table2[s][u] = length + 1
            length += 1
            print(str(counter) + " positions at distance " + str(length))
            if counter == 0:
                break

        print('Table3')
        table3 = np.zeros((2048, 495), dtype=np.int8)
        table3 -= 1
        table3[0][0] = 0
        length = 0
        while True:
            counter = 0
            for q in range(2048):
                for r in range(495):
                    if table3[q][r] == length:
                        for i in range(18):
                            t = self.edge_orientation_transition_table[q][i]
                            u = self.edge_combination_transition_table[r][i]
                            if table3[t][u] == -1:
                                counter += 1
                                table3[t][u] = length + 1
            length += 1
            print(str(counter) + " positions at distance " + str(length))
            if counter == 0:
                break
        return table1, table2, table3

    def get_corner_permutation_phase2_transition_table(self):
        print('Creating corner permutation phase 2 transition table...')
        table = np.zeros((40320, 10), dtype=np.uint16)
        for p in range(40320):
            for q, move in zip(range(10), self.available_moves_phase2(None)):
                _, table[p][q], _, _ = self.position_to_index(
                    self.result(self.index_to_position(0, p, 0, 0), move))
        return table

    def get_edge1_permutation_phase2_transition_table(self):
        print('Creating edge1 permutation phase 2 transition table...')
        table = np.zeros((40320, 10), dtype=np.uint16)
        for p in range(40320):
            for q, move in zip(range(10), self.available_moves_phase2(None)):
                table[p][q] = self.edge1_permutation_to_index(
                    self.result(self.index_to_edge1_permutation(p), move))
        return table

    def get_edge2_permutation_phase2_transition_table(self):
        print('Creating edge2 permutation phase 2 transition table...')
        table = np.zeros((40320, 10), dtype=np.uint16)
        for p in range(40320):
            for q, move in zip(range(10), self.available_moves_phase2(None)):
                table[p][q] = self.edge2_permutation_to_index(
                    self.result(self.index_to_edge2_permutation(p), move))
        return table

    def get_phase2_tables(self):
        print('Creating phase 2 tables...')
        print('Table1')
        table1 = np.zeros((40320, 24), dtype=np.int8)
        table1 -= 1
        table1[0][0] = 0
        length = 0
        while True:
            counter = 0
            for p in range(40320):
                for r in range(24):
                    if table1[p][r] == length:
                        for i in range(10):
                            s = self.corner_permutation_phase2_transition_table[p][i]
                            u = self.edge2_permutation_phase2_transition_table[r][i]
                            if table1[s][u] == -1:
                                counter += 1
                                table1[s][u] = length + 1
            length += 1
            print(str(counter) + " positions at distance " + str(length))
            if counter == 0:
                break

        print('Table2')
        table2 = np.zeros((40320, 24), dtype=np.int8)
        table2 -= 1
        table2[0][0] = 0
        length = 0
        while True:
            counter = 0
            for q in range(40320):
                for r in range(24):
                    if table2[q][r] == length:
                        for i in range(10):
                            t = self.edge1_permutation_phase2_transition_table[q][i]
                            u = self.edge2_permutation_phase2_transition_table[r][i]
                            if table2[t][u] == -1:
                                counter += 1
                                table2[t][u] = length + 1
            length += 1
            print(str(counter) + " positions at distance " + str(length))
            if counter == 0:
                break
        return table1, table2

    def position_to_index(self, position):
        corner_orientation_index = 0
        corner_permutation_index = 0
        edge_orientation_index = 0
        edge_permutation_index = 0

        for i in range(7):
            corner_orientation_index *= 3
            corner_orientation_index += position["orientation"][i]

            corner_permutation_index *= 8-i
            for j in range(i+1, 8):
                if position["permutation"][i] > position["permutation"][j]:
                    corner_permutation_index += 1

        for i in range(8, 19):
            edge_orientation_index *= 2
            edge_orientation_index += position["orientation"][i]

            if i != 18:
                edge_permutation_index *= 20-i
                for j in range(i+1, 20):
                    if position["permutation"][i] > position["permutation"][j]:
                        edge_permutation_index += 1

        return corner_orientation_index, corner_permutation_index,\
            edge_orientation_index, edge_permutation_index

    def index_to_position(self, corner_orientation_index,
                          corner_permutation_index, edge_orientation_index,
                          edge_permutation_index):
        position = dict(orientation=np.zeros(20, dtype=np.int8),
                        permutation=np.zeros(20, dtype=np.int8))

        last_orientation = 0
        position["permutation"][7] = 0
        for i in range(6, -1, -1):
            position["orientation"][i] = corner_orientation_index % 3
            last_orientation -= position["orientation"][i]
            if last_orientation < 0:
                last_orientation += 3
            corner_orientation_index //= 3

            position["permutation"][i] = corner_permutation_index % (8-i)
            corner_permutation_index //= (8-i)
            for j in range(i+1, 8):
                if position["permutation"][j] >= position["permutation"][i]:
                    position["permutation"][j] += 1
        position["orientation"][7] = last_orientation

        last_orientation = 0
        position["permutation"][19] = 9
        position["permutation"][18] = 8
        parity = self._parity_permutation(position["permutation"][0:8])
        for i in range(18, 7, -1):
            position["orientation"][i] = edge_orientation_index % 2
            last_orientation -= position["orientation"][i]
            if last_orientation < 0:
                last_orientation += 2
            edge_orientation_index //= 2

            if i != 18:
                position["permutation"][i] = 8 +\
                    edge_permutation_index % (20-i)
                parity += position["permutation"][i] - 1
                edge_permutation_index //= (20-i)
                for j in range(i+1, 20):
                    if position["permutation"][j] >=\
                            position["permutation"][i]:
                        position["permutation"][j] += 1
        position["orientation"][19] = last_orientation

        if parity % 2 == 1:
            position["permutation"][18], position["permutation"][19] =\
                position["permutation"][19], position["permutation"][18]

        return position

    def _parity_permutation(self, permutation):
        inversions = 0
        for i in range(len(permutation)-1):
            for j in range(i+1, len(permutation)):
                if permutation[i] > permutation[j]:
                    inversions += 1
        return inversions % 2

    def edge_combination_to_index(self, position):
        edge_combination_index = 0
        r = 8
        for i in range(11, -1, -1):
            if position["permutation"][i+8] < 16:
                edge_combination_index += self._comb(i, r)
                r -= 1
        return edge_combination_index

    def index_to_edge_combination(self, edge_combination_index):
        position = dict(orientation=np.zeros(20, dtype=np.int8),
                        permutation=np.array([16] * 20), dtype=np.int8)
        r = 8
        for i in range(11, -1, -1):
            if edge_combination_index >= self._comb(i, r):
                edge_combination_index -= self._comb(i, r)
                position["permutation"][i+8] = 8
                r -= 1
        return position

    def _comb(self, i, r):
        if (i-r) < 0:
            return 0
        return math.factorial(i) // (math.factorial(r) * math.factorial(i-r))

    def edge1_permutation_to_index(self, position):
        edge1_permutation_index = 0
        for i in range(8, 15):
            edge1_permutation_index *= 16-i
            for j in range(i+1, 16):
                if position["permutation"][i] > position["permutation"][j]:
                    edge1_permutation_index += 1
        return edge1_permutation_index

    def index_to_edge1_permutation(self, edge1_permutation_index):
        position = dict(orientation=np.zeros(20, dtype=np.int8),
                        permutation=np.zeros(20, dtype=np.int8))
        position["permutation"][15] = 8
        for i in range(14, 7, -1):
            position["permutation"][i] = 8 +\
                edge1_permutation_index % (16-i)
            edge1_permutation_index //= (16-i)
            for j in range(i+1, 16):
                if position["permutation"][j] >= position["permutation"][i]:
                    position["permutation"][j] += 1
        return position

    def edge2_permutation_to_index(self, position):
        edge2_permutation_index = 0
        for i in range(16, 19):
            edge2_permutation_index *= 20-i
            for j in range(i+1, 20):
                if position["permutation"][i] > position["permutation"][j]:
                    edge2_permutation_index += 1
        return edge2_permutation_index

    def index_to_edge2_permutation(self, edge2_permutation_index):
        position = dict(orientation=np.zeros(20, dtype=np.int8),
                        permutation=np.zeros(20, dtype=np.int8))
        position["permutation"][19] = 16
        for i in range(18, 15, -1):
            position["permutation"][i] = 16 +\
                edge2_permutation_index % (20-i)
            edge2_permutation_index //= (20-i)
            for j in range(i+1, 20):
                if position["permutation"][j] >=\
                        position["permutation"][i]:
                    position["permutation"][j] += 1
        return position

    def available_moves(self, last_move):
        if last_move is None:
            return "U U2 U' L L2 L' F F2 F' R R2 R' B B2 B' D D2 D'".split()
        elif last_move[0] == 'U':
            return "L L2 L' F F2 F' R R2 R' B B2 B' D D2 D'".split()
        elif last_move[0] == 'L':
            return "U U2 U' F F2 F' R R2 R' B B2 B' D D2 D'".split()
        elif last_move[0] == 'F':
            return "U U2 U' L L2 L' R R2 R' B B2 B' D D2 D'".split()
        elif last_move[0] == 'R':
            return "U U2 U' F F2 F' B B2 B' D D2 D'".split()
        elif last_move[0] == 'B':
            return "U U2 U' L L2 L' R R2 R' D D2 D'".split()
        elif last_move[0] == 'D':
            return "L L2 L' F F2 F' R R2 R' B B2 B'".split()

    def available_moves_phase2(self, last_move):
        if last_move is None:
            return "U U2 U' L2 F2 R2 B2 D D2 D'".split()
        elif last_move[0] == 'U':
            return "L2 F2 R2 B2 D D2 D'".split()
        elif last_move[0] == 'L':
            return "U U2 U' F2 R2 B2 D D2 D'".split()
        elif last_move[0] == 'F':
            return "U U2 U' L2 R2 B2 D D2 D'".split()
        elif last_move[0] == 'R':
            return "U U2 U' F2 B2 D D2 D'".split()
        elif last_move[0] == 'B':
            return "U U2 U' L2 R2 D D2 D'".split()
        elif last_move[0] == 'D':
            return "L2 F2 R2 B2".split()

    def generate_scramble(self, length):
        scramble = [None]
        for i in range(length):
            scramble.append(random.choice(self.available_moves(scramble[i])))
        return ' '.join(scramble[1:])

    def random_scramble(self):
        co = random.randint(0, 2187)
        cp = random.randint(0, 40320)
        eo = random.randint(0, 2048)
        ep = random.randint(0, 239500800)
        self.position = self.index_to_position(co, cp, eo, ep)

    def IDA(self, max_length=20):
        for depth in range(max_length+1):
            if self.tree_search(self.position, depth):
                return True
        return False

    def tree_search(self, position, depth, last_move=None):
        if depth == 0:
            if self.is_solved(position):
                return True
        elif depth > 0:
            co, cp, eo, _ = self.position_to_index(position)
            ec = self.edge_combination_to_index(position)
            if self.corner_orientation_table[co] <= depth and\
                    self.corner_permutation_table[cp] <= depth and\
                    self.edge_orientation_table[eo] <= depth and\
                    self.edge_combination_table[ec] <= depth:
                for move in self.available_moves(last_move):
                    if self.tree_search(self.result(position, move),
                                        depth-1, move):
                        self.solution.append(move)
                        return True
        return False

    def is_solved(self, position=None):
        is_orientation = np.array_equal(self.position["orientation"]
                                        if position is None
                                        else position["orientation"],
                                        np.zeros(20, dtype=np.int8))
        is_permutation = np.array_equal(self.position["permutation"]
                                        if position is None
                                        else position["permutation"],
                                        np.arange(20, dtype=np.int8))
        return is_orientation and is_permutation

    def Kociemba(self):
        for depth in range(13):
            if self.phase1_search(self.position, depth):
                return True
        return False

    def phase1_search(self, position, depth, last_move=None):
        if depth == 0:
            if self._subgoal(position) and self.quarter_turn_RLFB(last_move):
                if self.phase2_start(position):
                    return True
        elif depth > 0:
            co, _, eo, _ = self.position_to_index(position)
            ec = self.edge_combination_to_index(position)
            if self.phase1_tables[0][co][eo] <= depth and\
                    self.phase1_tables[1][co][ec] <= depth and\
                    self.phase1_tables[2][eo][ec] <= depth:
                for move in self.available_moves(last_move):
                    if self.phase1_search(self.result(position, move),
                                          depth-1, move):
                        self.solution.append(move)
                        return True
        return False

    def phase2_start(self, position):
        for depth in range(19):
            if self.phase2_search(position, depth):
                self.solution.append(".")
                return True
        return False

    def phase2_search(self, position, depth, last_move=None):
        if depth == 0:
            if self.is_solved(position):
                return True
        elif depth > 0:
            _, cp, _, _ = self.position_to_index(position)
            e1p = self.edge1_permutation_to_index(position)
            e2p = self.edge2_permutation_to_index(position)
            if self.phase2_tables[0][cp][e2p] <= depth and\
                    self.phase2_tables[1][e1p][e2p] <= depth:
                for move in self.available_moves_phase2(last_move):
                    if self.phase2_search(self.result(position, move),
                                          depth-1, move):
                        self.solution.append(move)
                        return True
        return False

    def _subgoal(self, position):
        is_orientation = np.array_equal(position["orientation"],
                                        np.zeros(20, dtype=np.int8))
        is_combination = True if self.edge_combination_to_index(position) == 0\
            else False
        return is_orientation and is_combination

    def quarter_turn_RLFB(self, move):
        if move is None:
            return True
        elif move == "R":
            return True
        elif move == "R'":
            return True
        elif move == "L":
            return True
        elif move == "L'":
            return True
        elif move == "F":
            return True
        elif move == "F'":
            return True
        elif move == "B":
            return True
        elif move == "B'":
            return True
        return False


rubik = Rubik()
scramble = rubik.generate_scramble(30)
print("Scramble:", scramble)
rubik.apply_scramble(scramble)
rubik.Kociemba()
rubik.solution.reverse()
print("Solution:", ' '.join(rubik.solution))
