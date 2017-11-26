import numpy as np


def phi(x, G):
    return 1.0/(1 + np.exp(-G * x))


def run_network_recall(N, w, G, threshold, tau_m, tau_z,  T, dt, I_cue, T_cue, sigma=0):

    x = np.zeros(N)
    current = np.zeros(N)
    z = np.zeros(N)
    x[I_cue] = 1.0  # Initial condition
    z[I_cue] = 0.0

    x_history = []
    z_history = []
    current_history = []

    steps = int(T / dt)
    steps_cue = int(T_cue / dt)
    for i in range(steps):
        x_history.append(np.copy(x))
        z_history.append(np.copy(z))
        current_history.append(np.copy(current))
        current = np.dot(w, z) + sigma * np.random.randn(N)
        x += (dt/tau_m) * (phi(G, current - threshold) - x)
        if i < steps_cue:
            x[I_cue] = 1
        z += (dt / tau_z) * (x - z)

    x_history = np.array(x_history)
    z_history = np.array(z_history)
    current_history = np.array(current_history)

    dic = {}
    dic['x'] = x_history
    dic['z'] = z_history
    dic['current'] = current_history

    return dic


def train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z, tau_z_post, tau_w,
                  epochs=1, max_w=1.0, min_w=None, save_w_history=False, pre_rule=True):

    w = np.zeros((N, N))
    w_history = [w]

    inter_sequence_steps = int(inter_sequence_time / dt)

    x_total = np.array([]).reshape(0, N)
    for epoch in range(epochs):
        for sequence in sequences:
            n_sequence = len(sequence)
            training_steps = int(training_time / dt)

            for element in sequence:
                x = np.zeros((training_steps, N))
                for time in range(training_steps):
                    x[time, element] = 1.0
                # Concatenate for the total history
                x_total = np.concatenate((x_total, x), axis=0)

            # Inter-sequence steps
            x = np.zeros((inter_sequence_steps, N))
            x_total = np.concatenate((x_total, x), axis=0)

    # Train the Z-filters and w
    z = np.zeros(N)
    z_post = np.zeros(N)

    z_history = np.zeros_like(x_total)
    z_post_history = np.zeros_like(x_total)

    for index, x_example in enumerate(x_total):
        z += (dt / tau_z) * (x_example - z)
        z_post += (dt / tau_z_post) * (x_example - z_post)
        z_history[index, :] = z
        z_post_history[index, :] = z_post

        normal = np.outer(z_post, z)
        # This is the pre-synaptic rule (check this statement)
        if pre_rule:
            negative = np.outer(1 - z_post, z)
        else:
            negative = np.outer(1 - z, z_post)
        if min_w is None:
            w += (dt / tau_w) * ((max_w - w) * normal - negative)
        else:
            w += (dt / tau_w) * ((max_w - w) * normal + (min_w - w) * negative)

        if save_w_history:
            w_history.append(np.copy(w))


    dic = {}
    dic['w'] = w
    dic['x'] = x_example
    dic['z'] = z_history
    dic['z_post'] = z_post_history

    if save_w_history:
        dic['w_history'] = np.array(w_history)

    return dic


def run_network_recall_limit(N, w, G, threshold, tau_m, tau_z,  T, dt, I_cue, T_cue, sigma=0):

    x = np.zeros(N)
    current = np.zeros(N)
    z = np.zeros(N)
    x[I_cue] = 1.0  # Initial condition
    z[I_cue] = 0.0

    x_history = []
    z_history = []
    current_history = []

    steps = int(T / dt)
    steps_cue = int(T_cue / dt)
    for i in range(steps):
        x_history.append(np.copy(x))
        z_history.append(np.copy(z))
        current_history.append(np.copy(current))
        current = np.dot(w, z) + sigma * np.random.randn(N)
        x = np.heaviside(current - threshold, 1.0)
        if i < steps_cue:
            x[I_cue] = 1
        z += (dt / tau_z) * (x - z)

    x_history = np.array(x_history)
    z_history = np.array(z_history)
    current_history = np.array(current_history)

    dic = {}
    dic['x'] = x_history
    dic['z'] = z_history
    dic['current'] = current_history

    return dic

def run_network_recall_limit_end(N, w, G, threshold, tau_m, tau_z,  T, dt, I_cue, I_end, T_cue, sigma=0):

    x = np.zeros(N)
    current = np.zeros(N)
    z = np.zeros(N)
    x[I_cue] = 1.0  # Initial condition
    z[I_cue] = 0.0

    x_history = []
    z_history = []
    current_history = []

    steps = int(T / dt)
    steps_cue = int(T_cue / dt)
    for i in range(steps):
        x_history.append(np.copy(x))
        z_history.append(np.copy(z))
        current_history.append(np.copy(current))
        current = np.dot(w, z) + sigma * np.random.randn(N)
        x = np.heaviside(current - threshold, 1.0)
        if i < steps_cue:
            x[I_cue] = 1
        z += (dt / tau_z) * (x - z)

        if x[I_end] > 0.5:
            break

    x_history = np.array(x_history)
    z_history = np.array(z_history)
    current_history = np.array(current_history)

    dic = {}
    dic['x'] = x_history
    dic['z'] = z_history
    dic['current'] = current_history

    return dic

