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
sns.set_style(style='white')

from network import train_network, run_network_recall_limit

legends_instead_of_post = True
add_max_and_min_markers = True
captions = True
pre_rule = False
invert = True


N = 10
inter_sequence_time = 1.000
dt = 0.001
sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sequences = [sequence]

if pre_rule:
    tau_z_post_general = 0.005
    tau_w_general = 0.100
    training_time_general = 0.500
    tau_z_general = 0.050
    max_w_general = 20.0
    min_w_general = -10.0
    epochs_general = 4

else:
    tau_z_post_general = 0.005
    tau_w_general = 0.050
    training_time_general = 0.500
    tau_z_general = 0.100
    max_w_general = 5.0
    min_w_general = -10.0
    epochs_general = 4

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
if not legends_instead_of_post:
    ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

if captions:
    size = 30
    fig.text(0.05, 0.97, 'a)', size=size)
    fig.text(0.50, 0.97, 'b)', size=size)
    fig.text(0.05, 0.65, 'c)', size=size)
    fig.text(0.05, 0.34, 'd)', size=size)
    fig.text(0.50, 0.32, 'f)', size=size)


plt.tight_layout()

# Some general parameters
epochs = epochs_general

pattern = 3
pattern_from = 2

# Training time
training_times_vector = np.arange(0.050, 1.050, 0.050)
w_self = []
w_transition = []
w_inhition = []

tau_z = tau_z_general
tau_z_post = tau_z_post_general
tau_w = tau_w_general
max_w = max_w_general
min_w = min_w_general
epocs = epochs_general


training_time = training_time_general

for training_time in training_times_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=epochs, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=pre_rule)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])


if invert:
    w_inhition = [-w for w in w_inhition]
ax1.plot(training_times_vector, w_transition, 'o:', color='blue', linewidth=4, markersize=13, label=r'$Exc_T$')
ax1.plot(training_times_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label=r'$Inh$')
ax1.plot(training_times_vector, w_self, '*--', color='green', linewidth=2, markersize=15, label=r'$Exc_{self}$')

if add_max_and_min_markers:
    ax1.plot(training_times_vector, np.ones(len(w_self)) * max_w, '--', color='gray', label=r'$w_{max}$')
    ax1.plot(training_times_vector, -np.ones(len(w_self)) * min_w, ls=':', color='gray', label=r'$w_{min}$' )

ax1.axhline(0, ls='--', color='black', label='0')
ax1.axvline(training_time_general, ls='-', color='black')

ax1.set_xlabel('training time')
ax1.set_ylabel(r'$w$')
# ax1.legend(loc=4)

# Tau w
tau_w_vector = np.arange(0.010, 0.220, 0.010)
w_self = []
w_transition = []
w_inhition = []

tau_z = tau_z_general
tau_z_post = tau_z_post_general
tau_w = tau_w_general
max_w = max_w_general
min_w = min_w_general
epochs = epochs_general

for tau_w in tau_w_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=pre_rule)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])


if invert:
    w_inhition = [-w for w in w_inhition]
ax2.plot(tau_w_vector, w_transition, 'o:', color='blue', linewidth=4, markersize=13, label='Exc')
ax2.plot(tau_w_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
ax2.plot(tau_w_vector, w_self, '*--', color='green', linewidth=2, markersize=15, label='Self')

if add_max_and_min_markers:
    ax2.plot(tau_w_vector, np.ones(len(w_self)) * max_w, '--', color='gray', label=r'$w_{max}$')
    ax2.plot(tau_w_vector, -np.ones(len(w_self)) * min_w, ':', color='gray', label=r'$w_{min}$')

ax2.axhline(0, ls='--', color='black', label='0')
ax2.axvline(tau_w_general, ls='-', color='black')

ax2.set_xlabel(r'$\tau_w$')
ax2.set_ylabel(r'$w$')
# ax2.legend(loc=4)

# tau_z

tau_z_vector = np.arange(0.050, 0.550, 0.050)
w_self = []
w_transition = []
w_inhition = []
pattern = 3
pattern_from = 2


tau_z = tau_z_general
tau_z_post = tau_z_post_general
tau_w = tau_w_general
max_w = max_w_general
min_w = min_w_general
epochs = epochs_general


for tau_z in tau_z_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=pre_rule)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])

if invert:
    w_inhition = [-w for w in w_inhition]
ax3.plot(tau_z_vector, w_transition, 'o:', color='blue', linewidth=4, markersize=13, label='Exc')
ax3.plot(tau_z_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
ax3.plot(tau_z_vector, w_self, '*--', color='green', linewidth=2, markersize=15, label='Self')

if add_max_and_min_markers:
    ax3.plot(tau_z_vector, np.ones(len(w_self)) * max_w, '--', color='gray', label=r'$w_{max}$')
    ax3.plot(tau_z_vector, -np.ones(len(w_self)) * min_w, ls=':', color='gray', label=r'$w_{min}$')

ax3.axhline(0, ls='--', color='black', label='0')
ax3.axvline(tau_z_general, ls='-', color='black')

ax3.set_xlabel(r'$\tau_z$')
ax3.set_ylabel(r'$w$')

# tau_z_ post
if not legends_instead_of_post:
    tau_z_vector = np.arange(0.005, 0.105, 0.005)
    w_self = []
    w_transition = []
    w_inhition = []
    pattern = 3
    pattern_from = 2

    tau_z = tau_z_general
    tau_z_post = tau_z_post_general
    tau_w = tau_w_general
    max_w = max_w_general
    min_w = min_w_general
    epochs = epochs_general

    for tau_z_post in tau_z_vector:

        dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                            tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=pre_rule)

        w = dic['w']
        w_self.append(w[pattern, pattern])
        w_transition.append(w[pattern, pattern_from])
        w_inhition.append(w[pattern_from, pattern])

    if invert:
        w_inhition = [-w for w in w_inhition]
    ax4.plot(tau_z_vector, w_transition, 'o-', color='blue', linewidth=4, markersize=13, label='Exc')
    ax4.plot(tau_z_vector, w_inhition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
    ax4.plot(tau_z_vector, w_self, '*-', color='green', linewidth=2, markersize=15, label='Self')

    ax4.axhline(0, ls='--', color='black', label='0')

    ax4.set_xlabel(r'$\tau_{z_{post}}$')
    # ax4.set_ylabel('weight value')
    #ax4.legend(loc=4)

# Max  w
max_w_vector = np.arange(1, 20.0, 2.0)
w_self = []
w_transition = []
w_inhition = []
pattern = 3
pattern_from = 2

tau_z = tau_z_general
tau_z_post = tau_z_post_general
tau_w = tau_w_general
max_w = max_w_general
min_w = min_w_general
epochs = epochs_general

for max_w in max_w_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=pre_rule)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])

if invert:
    w_inhibition = [-w for w in w_inhition]

ax5.plot(max_w_vector, w_transition, 'o-', color='blue', linewidth=4, markersize=13, label='Exc')
ax5.plot(max_w_vector, w_inhibition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
ax5.plot(max_w_vector, w_self, '*-', color='green', linewidth=2, markersize=15, label='Self')

if add_max_and_min_markers:
    ax5.plot(max_w_vector, max_w_vector, '--', color='gray', label=r'$w_{max}$')
    ax5.plot(max_w_vector, -np.ones(len(w_self)) * min_w, ls=':', color='gray', label=r'$w_{min}$')

ax5.axhline(0, ls='--', color='black', label='0')
ax5.axvline(max_w_general, ls='-', color='black')

ax5.set_xlabel(r'$w_{max}$')
ax5.set_ylabel(r'$w$')

# min w
min_w_vector = -np.arange(1, 20.0 , 2.0)
w_self = []
w_transition = []
w_inhition = []
pattern = 3
pattern_from = 2

tau_z = tau_z_general
tau_z_post = tau_z_post_general
tau_w = tau_w_general
max_w = max_w_general
min_w = min_w_general
epochs = epochs_general

for min_w in min_w_vector:

    dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z,
                        tau_z_post, tau_w, epochs=5, max_w=max_w, min_w=min_w, save_w_history=False, pre_rule=pre_rule)

    w = dic['w']
    w_self.append(w[pattern, pattern])
    w_transition.append(w[pattern, pattern_from])
    w_inhition.append(w[pattern_from, pattern])


if invert:
    w_inhibition = [-w for w in w_inhition]
ax6.plot(min_w_vector, w_transition, 'o-', color='blue', linewidth=4, markersize=13, label='Exc')
ax6.plot(min_w_vector, w_inhibition, '^-', color='red', linewidth=6, markersize=18, label='Inh')
ax6.plot(min_w_vector, w_self, '*-', color='green', linewidth=2, markersize=15, label='Self')

if add_max_and_min_markers:
    ax6.plot(min_w_vector, np.ones(len(w_self)) * max_w, '--', color='gray', label=r'$w_{max}$')
    ax6.plot(min_w_vector, -min_w_vector, ':', color='gray', label=r'$w_{min}$')

ax6.axhline(0, ls='--', color='black', label=0)
ax6.axvline(min_w_general, ls='-', color='black')


ax6.set_xlabel(r'$w_{min}$')
ax6.set_ylabel(r'$w$')

# The legends
if legends_instead_of_post:
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc=(0.58, 0.42), fancybox=True, frameon=True, fontsize=32, ncol=2)
# Save the figure
name = './plot_producers/training_rule_quantities_pre_rule_' + str(pre_rule) + '.eps'
fig.savefig(name, frameon=False, dpi=110, bbox_inches='tight')
plt.close()
