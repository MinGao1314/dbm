import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from numpy.matlib import repmat

def sc_law(x,R=1):
    return 2/(np.pi*R**2) * np.sqrt(R**2 - x**2)


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def plot_dpp(eigenvalues_samples, title):
    fig = plt.figure(figsize=(15, 12))
    fig.subplots_adjust(hspace=.5)
    gs = GridSpec(2, 1, height_ratios=[1, 4])
    ax0 = plt.subplot(gs[0])

    ax0.scatter(eigenvalues_samples, np.zeros((len(eigenvalues_samples),)))
    adjust_spines(ax0, ['bottom'])
    # plt.savefig('gue_eigenvalues.png.png', bbox_inches='tight')

    # Distribution of the eigen values
    # Semi circle law
    ax1 = plt.subplot(gs[1], sharex=ax0)
    sc_nb_points = 200
    R = np.round(max(eigenvalues_samples))
    x = np.linspace(-R, R, sc_nb_points)
    ax1.plot(x, sc_law(x, R=R), 'red', label=r'Semi-circular law')

    # Histo
    num_bins = 20
    ax1.hist(eigenvalues_samples, num_bins, range=(-R, R), normed=1)

    # Plot options
    ax1.set_xlim([-R, R])
    plt.subplots_adjust(left=0.15)
    ax1.legend(loc='upper left', frameon=False)

    adjust_spines(ax1, ['left', 'bottom'])
    plt.savefig(title, bbox_inches='tight')


def plot_traj_with_histo(dbm, title):
    fig = plt.figure(1, figsize=(14, 6))
    gs = GridSpec(1, 4)

    traj_fig = fig.add_subplot(gs[0, 0:3])
    traj_hist = fig.add_subplot(gs[0, 3])

    # plot trajectories
    t = np.matlib.repmat(np.arange(0, dbm.tf+dbm.dt, dbm.dt), dbm.n_traj, 1)
    traj_fig.plot(t.T, dbm.eigen_values)

    # plot final values histogram
    eigen_values_final = dbm.eigen_values[-1]
    hist = traj_hist.hist(eigen_values_final, bins=10, normed=True, orientation='horizontal', label="Histogram at$\ t_f$")

    plt.setp(traj_hist.get_yticklabels(), visible=False)
    traj_hist.set_ylim(traj_fig.get_ylim())
    traj_hist.legend(prop={'size': 8})
    plt.savefig(title, bbox_inches='tight')


def plot_traj_with_histo_sclaw(dbm_rescale, R, title, n_bins=10):
    fig = plt.figure(1, figsize=(14, 6))
    fig.subplots_adjust(wspace=.1)
    gs = GridSpec(1, 2, width_ratios=[3, 1])

    # plot trajectories
    traj_fig = plt.subplot(gs[0])
    t = np.matlib.repmat(np.arange(0, dbm_rescale.tf+dbm_rescale.dt, dbm_rescale.dt), dbm_rescale.n_traj, 1)
    traj_fig.plot(t.T, dbm_rescale.eigen_values)
    traj_fig.set_ylim([-1.2*R, 1.2*R])

    # plot final values histogram
    traj_hist = plt.subplot(gs[1])
    hist = traj_hist.hist(dbm_rescale.eigen_values[-1], bins=n_bins, range=(-R, R), normed=True, orientation='horizontal',
                          label="Histogram at$\ t_f$")

    plt.setp(traj_hist.get_yticklabels(), visible=False)
    traj_hist.set_ylim([-1.2 * R, 1.2 * R])
    x = np.linspace(-R, R)
    traj_hist.plot(sc_law(x, R=R), x, color="red", label="Semicircle law")
    traj_hist.legend(prop={'size': 8})
    plt.savefig(title, bbox_inches='tight')