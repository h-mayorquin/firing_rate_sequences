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


def get_recall_duration_sequence(x_history):

    durations = []
    for pattern in range(x_history.shape[1]):
        duration = get_recall_duration_for_pattern(x_history, pattern)
        durations.append(duration)

    return np.nanmedian(durations[1:-1]), np.nanstd(durations[1:-1]), durations