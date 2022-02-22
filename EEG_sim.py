# Copyright (C) 2022 Daniel Enériz Orta
# 
# This file is part of EEG_sim.
# 
# EEG_sim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# EEG_sim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with EEG_sim.  If not, see <http://www.gnu.org/licenses/>.
# 
# Author: Daniel Enériz (eneriz@unizar.es)
# EEG_sim.py (c) 2022
# Desc: description
# Created:  2022-02-18T13:52:54.799Z
# Modified: 2022-02-22T07:50:00.066Z
# 
# This file contains the code to simulate the EEG signal from a cap with N_chan
# electrodes, sampled at fs Hz. The data is taken from the Physionet's EEG Motor
# Movement/Imagery dataset (https://www.physionet.org/content/eegmmidb/1.0.0/). 
# The data is simulated with a National Instruments USB-6212 DAQ card, using one
# analog output channel, thus the channels are multiplexed in time.

#%%
import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
from get_data import get_data, normalize
from typing import Tuple

processedPath = './processed/'
normalizedPath = './normalized/'
#%% Data preprocessing
"""
x, y, subs = get_data('./Dataset/', )

try:
    os.mkdir(processedPath)
except FileExistsError:
    pass

np.save(processedPath+'samples.npy', x)
np.save(processedPath+'labels.npy', y)
np.save(processedPath+'subs.npy', subs)

x, y = normalize(processedPath)

try:
    os.mkdir(normalizedPath)
except FileExistsError:
    pass

np.save(normalizedPath+'samples.npy', x)
np.save(normalizedPath+'labels.npy', y)
"""
#%% Load dataset and define functions
x = np.load(normalizedPath+'samples.npy')
y = np.load(normalizedPath+'labels.npy')

def ExtractSignal(x: np.ndarray, y: np.ndarray, sub: int, label: int, instance: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts the selected labe data, in function of `sub`, `label` and
    `instance`.

    Args:
        x (np.ndarray): Normalized features dataset
        y (np.ndarray): Labels and subjet data
        sub (int): Selected subject to use data from. It could be anyone from 1
        to 109 except of the excluded (88, 92, 100, 104).
        label (int): Selected motor imagery class identifier. Must be in the
        range [0, 3].
        instance (int): Selected instance of the selected class. Must be in the
        range [0, 21], since there are 21 instances per class per subject.
    Returns:
        x_data (np.ndarray): EEG readings to simulate
        y_data (np.ndarray): Label and subject of the data
    """
    # Check if the selected subject is valid
    if sub not in range(1, 110) or sub in [88, 92, 100, 104]:
        raise ValueError('Invalid subject number. This subject is excluded.')
    
    # Check if the selected label is valid
    if label not in range(4):
        raise ValueError("Invalid label number. "
                         "It must be in the range [0, 3].")

    # Check if the selected instance is valid
    if instance not in range(22):
        raise ValueError("Invalid instance number. "
                         "It must be in the range [0, 21].")


    # Extract the data from the selected subject
    x_data = x[y[:, 1] == sub,:,:,0]
    y_data = y[y[:, 1] == sub, :]

    # Extract the data from the selected class
    x_data = x_data[y_data[:, 0] == label,:,:]
    y_data = y_data[y_data[:, 0] == label, :]

    # Extract the data from the selected instance
    x_data = x_data[instance,:,:]
    y_data = y_data[instance, :]
    
    return x_data, y_data

def GenerateSignal(x: np.ndarray, y: np.ndarray, channels: list = list(range(64)), T: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the EEG signal using the DAQ from the extracted data and
    multiplexing the channels.

    Args:
        x (np.ndarray): EEG readings to simulate
        y (np.ndarray): Label and subject of the data
        channels (list, optional): Channels to use in the simulation. Defaults
        to `range(64)`.
        T (int, optional): Time window of signal in seconds to generate.
        Defaults to 3 seconds.
    Returns:
        x_data (np.ndarray): Simulated EEG readings
        y_data (np.ndarray): Label and subject of the data
    """

    # Check if the selected channels are valid
    if not set(channels).issubset(set(range(64))):
        raise ValueError("Invalid channel subset. "
                         "It must be contained in the range [1, 64].")

    # Check if the selected channels are unique
    if len(channels) != len(set(channels)):
        raise ValueError("Invalid channels subset. "
                         "They must be unique.")
    
    fs = 160 # Sampling frequency
    N_chan = len(channels) # Number of channels
    N_samples = int(T*fs*N_chan) # Number of samples to generate

    # Configure the DAQ
    task = nidaqmx.Task()
    task.ao_channels.add_ao_voltage_chan(
        'Dev1/ao0',
        min_val=-1,
        max_val=1
    )
    task.timing.cfg_samp_clk_timing(
        fs*N_chan,
        sample_mode = nidaqmx.constants.AcquisitionType.FINITE, 
        samps_per_chan=N_samples
    )
        
    # Crop the data to the selected channels and time window
    x_data = x[channels, :T*fs]
    y_data = y
    
    # Generate the signal
    task.write(x_data.flatten('F')) 
        # This flattens the data in column-major order, where the first
        # dimension (channels) is the fastest changing dimension, thus
        # multiplexing the channels.
    task.start()
    task.wait_until_done()
    task.stop()
    task.close()

    # Return generated data and label
    return x_data, y_data
    
# %% Test signal generation
x_signal, y_signal = ExtractSignal(x, y, 1, 1, 1)
_, y_signal = GenerateSignal(x_signal, y_signal)
print(y_signal)
