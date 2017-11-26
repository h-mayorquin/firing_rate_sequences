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

independent_labels = False  # Whehter you want an extra weight trained matrix or the labels

N = 10
tau_z = 0.050
tau_z_1 = 0.150
tau_z_post = 0.005
tau_w = 0.100
max_w = 30.0
min_w = -3.0

training_time = 0.100
inter_sequence_time = 1.000
dt = 0.001
epochs = 10
sequence1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sequences = [sequence1]

dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z, tau_z_post,
                    tau_w, epochs=epochs, max_w=max_w, min_w=min_w, save_w_history=True)

w = dic['w']
x_total = dic['x']
z_history = dic['z']
z_post_history = dic['z_post']
w_history = dic['w_history']

dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z_1, tau_z_post,
                    tau_w, epochs=epochs, max_w=max_w, min_w=min_w, save_w_history=True)

w1 = dic['w']

epoch_length = N * (training_time) + (inter_sequence_time )
T = epoch_length * epochs
time = np.arange(0, T + dt, dt)

w_10 = w_history[:, 1, 0]
w_01 = w_history[:, 0, 1]
w_11 = w_history[:, 1, 1]

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, :])

if not independent_labels:
    print('here 1')
    ax3 = fig.add_subplot(gs[0, 1])

norm = matplotlib.colors.Normalize(0, 2)
cmap = matplotlib.cm.inferno_r

# The first w plot
im = ax1.matshow(w, cmap=cmap)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax)

ax1.set_xlabel('Pre')
ax1.set_ylabel('Post')
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])



# The epochs plot
ax2.plot(time, w_10,  ':', color='blue', markersize=1, linewidth=2, label=r'$Exc_T$')
ax2.plot(time, w_01, '-', color='red',  markersize=1, linewidth=4, label=r'$Inh$')
ax2.plot(time, w_11, '--', color='green', markersize=1, linewidth=6, label=r'$Exc_{self}$')

tick_spacing = 2
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
ticks =  [0, 2,    4,   6,   8,   10,  12,  14, 16,  18, 20]

plt.xticks(ticks, labels)

if not independent_labels:
    print('here 2')
    # ax2.legend(loc=4)

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Weight')
# ax2.set_xlim([-1, 25])

# Either a matrix or a legend
if not independent_labels:
    print('here 3')
    im3 = ax3.matshow(w1, cmap=cmap)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax)

    ax3.set_xlabel('Pre')
    ax3.set_ylabel('Post')
    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc=(0.3, 0.15), fancybox=False, frameon=False, fontsize=28, ncol=3)

else:
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc=(0.6, 0.65), fancybox=True, frameon=True, fontsize=28, ncol=1)



fig.savefig('./plot_producers/training_rule.eps', frameon=False, dpi=110, bbox_inches='tight')