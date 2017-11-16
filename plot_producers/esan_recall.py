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

from network import train_network, run_network_recall
from connectivity import designed_matrix_sequences


def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_grayscale", colors, cmap.N)

# First the connectivity matrix
N = 10
self_excitation = 2.0
inhibition = 3.0
transition = 1.0

sequence1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sequences = [sequence1]

w = designed_matrix_sequences(N, sequences=sequences, self_excitation=self_excitation,
                              inhbition=inhibition, transition=transition)

# Now the recall
G = 100.0
tau_m = 0.010
T = 2.0
I_cue = 0
T_cue = 0.100
dt = 0.001
tau_z = 0.050
threshold = 0.3

dic = run_network_recall(N, w, G, threshold, tau_m, tau_z,  T, dt, I_cue, T_cue)


time = np.arange(0, T, dt)
x_history = dic['x']
z_history = dic['z']
current_history = dic['current']

gs = gridspec.GridSpec(3, 2)
fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

patterns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# patterns = sequence

norm = matplotlib.colors.Normalize(0, N)
cmap = matplotlib.cm.inferno

for pattern in patterns:
    width = pattern * 0.5 + 0.5
    if pattern == 0:
        label = 'Cue'
    else:
        label = str(pattern)

    ax1.plot(time, x_history[:, pattern], color=cmap(norm(pattern)), linewidth=width, label=label)
    ax2.plot(time, current_history[:, pattern], color=cmap(norm(pattern)), linewidth=width, label='x' + label)
    ax3.plot(time, z_history[:, pattern], color=cmap(norm(pattern)), linewidth=width, label='x' + label)

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_ylim([-0.1, 1.1])
ax3.set_ylim([-0.1, 1.1])

ax1.axhline(0, ls='--', color='black')
ax2.axhline(0, ls='--', color='black')
ax3.axhline(0, ls='--', color='black')

ax1.set_ylabel('Unit Activity')
ax2.set_ylabel('Current')
ax3.set_ylabel('Z-filters')

ax1.get_xaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
ax3.set_xlabel('Time (s)')

# Here we plot our connectivity matrix
ax_conn = fig.add_subplot(gs[:2, 1])
cmap = 'seismic'
cmap = matplotlib.cm.inferno
# cmap = grayify_cmap(cmap)
im = ax_conn.imshow(w, cmap=cmap)

ax_conn.set_xlabel('Pre')
ax_conn.set_ylabel('Post')
ax_conn.xaxis.set_ticklabels([])
ax_conn.yaxis.set_ticklabels([])
ax_conn.grid()

divider = make_axes_locatable(ax_conn)
cax = divider.append_axes('right', size='5%', pad=0.05)

# Let's define our own color map
print(cmap(0))
print(cmap(1))
print(cmap(2))
print(cmap(3))
cmap = matplotlib.colors.ListedColormap([cmap(0), cmap(1), cmap(2)])
bounds = [-3, 0, 1, 2]
print(cmap(0))
print(cmap(1))
print(cmap(2))
print(cmap(3))

norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
# fig.colorbar(im, cax=cax, orientation='vertical', cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, spacing='uniform')
fig.colorbar(im, cax=cax, orientation='vertical')

# Let's plot our legends
# ax_legend = fig.add_subplot(gs[2, 1])
# lines = ax1.get_lines()
handles, labels = ax1.get_legend_handles_labels()
# ax_legend.legend(ax1.get_legend_handles_labels())

fig.legend(handles=handles, labels=labels, loc=(0.6, 0.05), fancybox=True, frameon=True, fontsize=28, ncol=2)

# plt.show()
fig.savefig('./plot_producers/recall.eps', frameon=False, dpi=110, bbox_inches='tight')