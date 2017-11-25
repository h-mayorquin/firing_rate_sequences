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

from network import run_network_recall, train_network, run_network_recall_limit

legends_instead_of_post = True


fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
if not legends_instead_of_post:
    ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

plt.tight_layout()

# Some general parameters
N = 10
inter_sequence_time = 1.000
dt = 0.001
sequence1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sequences = [sequence1]
epochs = 5
pattern = 3
pattern_from = 2
pre_rule = True


# Training time
training_times_vector = np.arange(0.050, 1.050, 0.050)
w_self = []
w_transition = []
w_inhition = []

tau_z = 0.050
tau_z_post = 0.005
tau_w = 0.100
max_w = 10.0
min_w = -3.0

training_time = 0.100

for training_time in training_times_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=epochs, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=True)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])


ax1.plot(training_times_vector, w_transition, 'o:', color='blue', linewidth=4, markersize=13, label='Exc')
ax1.plot(training_times_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
ax1.plot(training_times_vector, w_self, '*--', color='green', linewidth=2, markersize=15, label='Self')

ax1.axhline(0, ls='--', color='black')

ax1.set_xlabel('training time')
ax1.set_ylabel('weight value')
# ax1.legend(loc=4)

# Tau w
tau_w_vector = np.arange(0.020, 0.520, 0.020)
w_self = []
w_transition = []
w_inhition = []
pattern = 3
pattern_from = 2

N = 10
tau_z = 0.050
tau_z_post = 0.005
tau_w = 0.100
max_w = 10.0
min_w = -3.0

training_time = 0.100
inter_sequence_time = 1.000
dt = 0.001
sequence1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sequences = [sequence1]

for tau_w in tau_w_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])

ax2.plot(tau_w_vector, w_transition, 'o:', color='blue', linewidth=4, markersize=13, label='Exc')
ax2.plot(tau_w_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
ax2.plot(tau_w_vector, w_self, '*--', color='green', linewidth=2, markersize=15, label='Self')

ax2.axhline(0, ls='--', color='black')

ax2.set_xlabel(r'$\tau_w$')
# ax2.legend(loc=4)

# tau_z

tau_z_vector = np.arange(0.050, 0.550, 0.050)
w_self = []
w_transition = []
w_inhition = []
pattern = 3
pattern_from = 2

N = 10
tau_z = 0.050
tau_z_post = 0.005
tau_w = 0.100
max_w = 80.0
min_w = -50.0

training_time = 0.350
inter_sequence_time = 1.000
dt = 0.001
sequence1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sequences = [sequence1]

for tau_z in tau_z_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=False)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])


ax3.plot(tau_z_vector, w_transition, 'o:', color='blue', linewidth=4, markersize=13, label='Exc')
ax3.plot(tau_z_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
ax3.plot(tau_z_vector, w_self, '*--', color='green', linewidth=2, markersize=15, label='Self')

ax3.axhline(0, ls='--', color='black')

ax3.set_xlabel(r'$\tau_z$')
ax3.set_ylabel('weight value')
# ax3.legend(loc=4);

# tau_z_ post
if not legends_instead_of_post:
    tau_z_vector = np.arange(0.005, 0.105, 0.005)
    w_self = []
    w_transition = []
    w_inhition = []
    pattern = 3
    pattern_from = 2

    N = 10
    tau_z = 0.300
    tau_z_post = 0.005
    tau_w = 0.100
    max_w = 100.0
    min_w = -3.0

    training_time = 0.100
    inter_sequence_time = 1.000
    dt = 0.001
    sequence1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sequences = [sequence1]

    for tau_z_post in tau_z_vector:

        dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                            tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=True)

        w = dic['w']
        w_self.append(w[pattern, pattern])
        w_transition.append(w[pattern, pattern_from])
        w_inhition.append(w[pattern_from, pattern])


    ax4.plot(tau_z_vector, w_transition, 'o-', color='blue', linewidth=4, markersize=13, label='Exc')
    ax4.plot(tau_z_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
    ax4.plot(tau_z_vector, w_self, '*-', color='green', linewidth=2, markersize=15, label='Self')

    ax4.axhline(0, ls='--', color='black')

    ax4.set_xlabel(r'$\tau_{z_{post}}$')
    # ax4.set_ylabel('weight value')
    #ax4.legend(loc=4)

# Max  w
max_w_vector = np.arange(1, 56.0 , 5.0)
w_self = []
w_transition = []
w_inhition = []
pattern = 3
pattern_from = 2

N = 10
tau_z = 0.050
tau_z_post = 0.005
tau_w = 0.100
max_w = 30.0
min_w = -3.0

training_time = 0.100
inter_sequence_time = 1.000
dt = 0.001
sequence1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sequences = [sequence1]

for max_w in max_w_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=True)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])

ax5.plot(max_w_vector, w_transition, 'o-', color='blue', linewidth=4, markersize=13, label='Exc')
ax5.plot(max_w_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
ax5.plot(max_w_vector, w_self, '*-', color='green', linewidth=2, markersize=15, label='Self')

ax5.axhline(0, ls='--', color='black')

ax5.set_xlabel(r'$w_{max}$')
ax5.set_ylabel('weight value')
# ax5.legend(loc=4)


# min w
min_w_vector = -np.arange(1, 56.0 , 5.0)
w_self = []
w_transition = []
w_inhition = []
pattern = 3
pattern_from = 2

N = 10
tau_z = 0.050
tau_z_post = 0.005
tau_w = 0.100
max_w = 30.0
min_w = -3.0

training_time = 0.100
inter_sequence_time = 1.000
dt = 0.001
sequence1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sequences = [sequence1]

for min_w in min_w_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=True)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])


ax6.plot(min_w_vector, w_transition, 'o-', color='blue', linewidth=4, markersize=13, label='Exc')
ax6.plot(min_w_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
ax6.plot(min_w_vector, w_self, '*-', color='green', linewidth=2, markersize=15, label='Self')

ax6.axhline(0, ls='--', color='black')

ax6.set_xlabel(r'$w_{min}$')
# ax6.set_ylabel('weight value')
# ax6.legend(loc=4)

# The legends
if legends_instead_of_post:
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc=(0.70, 0.40), fancybox=True, frameon=True, fontsize=38, ncol=1)
# Save the figure

fig.savefig('./plot_producers/training_rule_quantities.eps', frameon=False, dpi=110, bbox_inches='tight')


