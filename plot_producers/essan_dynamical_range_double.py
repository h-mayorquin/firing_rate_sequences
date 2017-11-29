import sys
sys.path.append('../')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["figure.figsize"] = [16, 9]
sns.set(font_scale=3.0)
sns.set_style(style='white')

from network import run_network_recall, train_network, run_network_recall_limit
from connectivity import designed_matrix_sequences, designed_matrix_sequences_local
from analysis import get_recall_duration_for_pattern, get_recall_duration_sequence
from analysis import time_t1, time_t2, time_t1_local, time_t2_local, time_t2_complicated

N = 10
sequences = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
self_excitation = 3.0
transition = 0.5
inhibition = 50.0

threshold = 0.5
tau_z = 0.050

G = 100.0
tau_m = 0.010
T = 20.0
I_cue = 0
T_cue = 0.100
dt = 0.001

pattern = 2
transition_vector_2 = np.arange(0.55, 1.0, 0.05)
transition_vector_1 = np.logspace(-14, -0.3, 20) + 0.5
transition_vector = np.hstack((transition_vector_1, transition_vector_2))
transition_vector.sort()

recall_times = []
for transition in transition_vector:
    w = designed_matrix_sequences(N, sequences, self_excitation=self_excitation, transition=transition,
                              inhibition=inhibition)

    dic = run_network_recall_limit(N, w, G, threshold, tau_m, tau_z,  T, dt, I_cue, T_cue)
    x_history = dic['x']
    duration = get_recall_duration_for_pattern(x_history, pattern, dt)
    recall_times.append(duration)


# The figure
time_t1_recall = time_t1(tau_z, T=transition_vector, I=inhibition, threshold=threshold)

captions = True

fig = plt.figure()
gs = gridspec.GridSpec(2, 2)

ax = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])

if captions:
    size = 30
    fig.text(0.05, 0.95, 'a)', size=size)
    fig.text(0.55, 0.95, 'b)', size=size)
    fig.text(0.55, 0.50, 'c)', size=size)

plt.tight_layout()

ax.plot(transition_vector, recall_times, '*', markersize=15, label='recall time')
ax.plot(transition_vector, time_t1_recall, label='theoretical')

ax.axhline(0, ls='--', color='black')
ax.axvline(threshold, ls=':', color='grey', label=r'threshold $\theta$')

ax.set_title('Dynamical Range')
ax.set_xlabel(r'$Exc_T$')
ax.set_ylabel('Recall time (s)')

ax.legend()

# Here the examples
sequences = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
cmap = matplotlib.cm.binary


# The first example
T = 8.25
transition = 1.0

w = designed_matrix_sequences(N, sequences, self_excitation=self_excitation, transition=transition,
                              inhibition=inhibition)
dic = run_network_recall_limit(N, w, G, threshold, tau_m, tau_z, T, dt, I_cue, T_cue)
x_history = dic['x']


im1 = ax1.imshow(x_history.T, aspect='auto', cmap=cmap)

ax1.set_xlabel('Time')
ax1.set_ylabel('Unit')
ax1.set_title(r'$Exc_T$ = ' + str(transition))
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])

transition = 0.51
w = designed_matrix_sequences(N, sequences, self_excitation=self_excitation, transition=transition,
                              inhibition=inhibition)
dic = run_network_recall_limit(N, w, G, threshold, tau_m, tau_z, T, dt, I_cue, T_cue)
x_history = dic['x']

# The second example
im2 = ax2.imshow(x_history.T, aspect='auto', cmap=cmap)

ax2.set_xlabel('Time')
ax2.set_ylabel('Unit')
ax2.set_title(r'$Exc_T$ = ' + str(transition))
ax2.xaxis.set_ticklabels([])
ax2.yaxis.set_ticklabels([])


fig.savefig('./plot_producers/dynamical_range.eps', frameon=False, dpi=110, bbox_inches='tight')
