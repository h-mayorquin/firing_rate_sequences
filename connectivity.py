import numpy as np


def designed_matrix_sequences(N, sequences, self_excitation=1, transition=1, inhbition=1):
    w = np.ones((N, N)) * -inhbition

    # Self-excitation
    for sequence in sequences:
        for element in sequence:
            w[element, element] = self_excitation

    # sequence
    for sequence in sequences:
        for index in range(len(sequence) - 1):
            w[sequence[index + 1], sequence[index]] = transition

    return w