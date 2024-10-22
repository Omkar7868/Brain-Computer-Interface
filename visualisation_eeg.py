import matplotlib.pyplot as plt
import numpy as np
import mne


# plot ERPs for individual channels for both the condition (standard and odd stimuli)
def visualise_individual_channels(epochs, t_min, t_max):

    # get epoch data
    epochs._data = epochs.get_data()

    # get ERPs for target and non target stimuli
    target = epochs['Odd'].get_data().mean(axis=0)
    nontarget = epochs['Standard'].get_data().mean(axis=0)

    ch_labels = epochs.info['ch_names']

    # plot the ERPs for individual channels
    t = np.linspace(t_min, t_max, target.shape[1])
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    for i, ax in enumerate(axes.flatten()):
        ax.plot(t, target[i, :], label='Target')
        ax.plot(t, nontarget[i, :], 'tab:orange', label='Non-Target')
        # ax.plot([0, 0], [-5, 5], linestyle='dotted', color='black')
        ax.axvspan(0.25, 0.35, color='grey', alpha=0.3)
        ax.set_ylabel('\u03BCV')
        ax.set_xlabel('Time (s)')
        ax.set_title(ch_labels[i])
        ax.legend()
    fig.tight_layout()
    plt.show()



# plot ERPs by taking mean across channels for the two condition (standard and odd stimuli)

def plot_ERP(epochs):
    
    erp_target = epochs['Odd'].average()
    erp_non_target = epochs['Standard'].average()

    selected_channels = epochs.info['ch_names']
    evoked = dict(Target=erp_target, NonTarget=erp_non_target)
    picks = [ch for ch in selected_channels]

    fig, ax = plt.subplots()
    mne.viz.plot_compare_evokeds(evoked, picks=picks, combine="mean", axes=ax, show=False)

    # Add grey box from 0.2s to 0.4s
    plt.axvspan(0.25, 0.35, color='grey', alpha=0.3)
    plt.show()


