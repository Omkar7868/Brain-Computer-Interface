import mne
import pandas as pd
import numpy as np
import os


# Filter EEG data
def filter_eeg_data(raw, l_freq, h_freq, filter_type='iir', filter_order=5, notch_filter=bool, notch_freq=None):
    # apply bandpass filter (iir or fir)
    if filter_type == 'iir':
        iir_params = dict(order=filter_order, ftype="butter")
        raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, method="iir", iir_params=iir_params,
                                         verbose=False)
    else:
        raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose=False)

    # apply notch filter if required
    if notch_filter:
        raw_filtered = raw_filtered.copy().notch_filter(freqs=notch_freq, verbose=False)
    else:
        pass

    return raw_filtered


def create_mne_object(rootpath, data_file_name):

    ## Read data frame from the parquest file ##
    data_file = os.path.join(rootpath, data_file_name)
    data_df = pd.read_parquet(data_file)

    ## Creating mne Objects ##

    # set info for mne raw object
    sfreq = 250
    ch_type = ['eeg']
    ch_names = ['O1', 'O2', 'T3', 'T4']
    info = mne.create_info(s_freq=sfreq, ch_names=ch_names, ch_type=ch_type)

    # set montage
    montage = mne.channels.make_standard_montage('standard_1020')

    # read eeg data from dataframe
    eeg_data = data_df.dropna(subset=['o1', 'o2', 'o3', 'o4']).copy()
    raw_data = eeg_data[['o1', 'o2', 't3', 't4']]

    # create raw mne object
    raw = mne.io.RawArray(raw_data, info)
    raw.set_montage(montage)

    ## Creating events ##

    # get event data
    event_data = data_df.dropna(subset=['event_id']).copy()

    # get valid event ids (10000 = start of the experiment, 10001 - end of experiment, 1 - standard stimuli, 2 - odd stimuli)
    valid_events = event_data[event_data['event_id'].isin([10000.0, 1.0, 2.0, 10001.0])].copy()

    # get valid event ids indices
    valid_event_indices = valid_events.index.to_numpy() + 1  # adding plus 1 as the eeg data starts after 1 index after the event as eeg data row is na for the event id

    # creating mne events structure
    events = np.column_stack((valid_event_indices,
                              np.zeros(len(valid_event_indices)),
                              valid_events['event_id'].astype(int)
                              ))

    return raw, events


def create_eeg_segments(eeg_raw_data, events):

    # Define event dictionary (map event_ids to stimuli names)
    event_id = {'Standard': 1, 'Odd': 2}

    # Create epochs from raw data and events
    epochs = mne.Epochs(raw=eeg_raw_data, events=events, event_id=event_id, tmin=-0.2, tmax=0.8, preload=True)

    return epochs




