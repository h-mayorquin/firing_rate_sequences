import numpy as np


def get_recall_duration_for_pattern(x_history, pattern, dt):
    x_indexes = np.where(x_history[:, pattern] > 0.5)[0]
    if len(x_indexes) > 0.0:
        start = x_indexes[0] * dt
        end = x_indexes[-1] * dt
        duration = end - start
    else:
        duration = np.nan
    return duration


def get_recall_duration_sequence(x_history, dt):

    durations = []
    for pattern in range(x_history.shape[1]):
        duration = get_recall_duration_for_pattern(x_history, pattern, dt)
        durations.append(duration)

    return np.nanmedian(durations[1:-1]), np.nanstd(durations[1:-1]), durations


def time_charge(tau_z, A, threshold):
    """
    This is how long it takes for the first pattern to charge if you want the sequence mechanism to fire correclty
    :param tau_z: tau_z
    :param A:  self-excitation
    :param threshold:  threshold of the non-linear function
    :return: the time it takes for the neuron to charge enough (lower limit)
    """

    return tau_z * np.log(A / (A - threshold))


def time_t1(tau_z, T, I, threshold):

    return tau_z * np.log((T + I) / (T - threshold))


def time_t2(tau_z, A, T, I, threshold):

    t1 = time_t1(tau_z, T, I, threshold)
    B = np.exp(-t1 / tau_z)
    return tau_z * np.log((A * B + I * B - I) / (A - (threshold + I)))


def time_t2_complicated(tau_z, A, T, I, threshold):

    t1 = time_t1(tau_z, T, I, threshold)
    B = np.exp(-t1 / tau_z)
    D = 1.0 - B

    nominator = B * D * (T + I) + I * D - A * B

    return tau_z * np.log(nominator / ((threshold + I) - A))


def time_t1_local(tau_z, T, threshold):

    return tau_z * np.log(T / (T - threshold))


def time_t2_local(tau_z, A, T, I, threshold):

    t1 = time_t1_local(tau_z, T, threshold)
    B = np.exp(-t1 / tau_z)

    return tau_z * np.log((A * B - I) / (A - (threshold + I)))


def create_sequence_chain(number_of_sequences, half_width, units_to_overload):
    chain = []
    number = 0
    for _ in range(number_of_sequences):

        sequence = []

        # The first half
        i = 0
        while i < half_width:
            if number in units_to_overload:
                number += 1

            else:
                sequence.append(number)
                number += 1
                i += 1

        # The overload units in the middle
        sequence += units_to_overload

        # The second half
        i = 0
        while i < half_width:
            if number in units_to_overload:
                number += 1
            else:
                sequence.append(number)
                number += 1
                i += 1

        chain.append(sequence)

    return chain