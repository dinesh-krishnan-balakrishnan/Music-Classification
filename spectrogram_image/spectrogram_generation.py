# --------------------- IMPORTS --------------------- #

# Track Loading Imports
import utils
import librosa

# Spectrogram Generation Import
import matplotlib.pyplot as PLT

# Other Imports
import os
import gc

# -------------------- CONSTANTS -------------------- #

# Audio File Directory Constant
AUDIO_DIR = os.path.join('D:', 'fma_medium')

# Figure Constants
PLT.rcParams['figure.figsize'] = (32, 16)
PLT.ioff()

# Audio Sampling Constants
DURATION = 25
SAMPLING_RATE = 48_000

# --------------------- FUNCTION --------------------- #

def worker(args):
    track = args[0]
    spectrograms_CREATE = args[1]
        
    # Loading track data.
    try:
        track_pathname = utils.get_audio_path(AUDIO_DIR, track)
        data, _ = librosa.load(track_pathname, duration = DURATION, sr = SAMPLING_RATE)

        for spectrogram_filename in spectrograms_CREATE:            
            # Creating the spectrogram graph container.
            figure = PLT.figure()
            PLT.axis('off')

            # Generating the spectrogram.
            image = spectrograms_CREATE[spectrogram_filename](data, None, SAMPLING_RATE)

            # Saving the spectrogram to a file.
            PLT.savefig(spectrogram_filename, bbox_inches = 'tight', pad_inches = 0, dpi = 100)

            # Clearing memory to prevent leaks.
            PLT.cla(); PLT.clf(); PLT.close('all'); gc.collect();
            
    # Informs the leader process whether generating spectrograms for the current track has failed.
    except Exception as E: return track
    return None