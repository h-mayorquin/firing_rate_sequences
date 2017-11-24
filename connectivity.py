import numpy as np


def designed_matrix_sequences(N, sequences, self_excitation=1, transition=1, inhibition=1):
    w = np.ones((N, N)) * -inhibition

    # Self-excitation
    for sequence in sequences:
        for element in sequence:
            w[element, element] = self_excitation

    # sequence
    for sequence in sequences:
        for index in range(len(sequence) - 1):
            w[sequence[index + 1], sequence[index]] = transition

    return w


def designed_matrix_sequences_local(N, sequences, self_excitation=1, transition=1, inhibition=1):
    w = np.zeros((N, N))

    # Self-excitation
    for sequence in sequences:
        for element in sequence:
            w[element, element] = self_excitation

    # sequence
    for sequence in sequences:
        for index in range(len(sequence) - 1):
            w[sequence[index + 1], sequence[index]] = transition
            w[sequence[index], sequence[index + 1]] = -inhibition

    return w