import librosa
import librosa.display as LIBD
import numpy as NP

"""
There are various spectrogram models that can represent audio data. For ease of use, I've created
functions that take raw audio data and convert it into these various spectrogram models. The 
spectrogram visualizations can be seen in the Feature Visualization notebook.
"""

def linear_spectrogram(data, axes, SAMPLING_RATE):
    data = librosa.stft(data)
    data = librosa.amplitude_to_db(NP.abs(data), ref = NP.max)
    image = LIBD.specshow(data, sr = SAMPLING_RATE, x_axis = 's', y_axis = 'linear', ax = axes)
    return image

def log_spectrogram(data, axes, SAMPLING_RATE):
    data = librosa.stft(data)
    data = librosa.amplitude_to_db(NP.abs(data), ref = NP.max)
    image = LIBD.specshow(data, sr = SAMPLING_RATE, x_axis = 's', y_axis = 'log', ax = axes)
    return image

def mel_spectrogram(data, axes, SAMPLING_RATE):
    data = librosa.feature.melspectrogram(data, sr = SAMPLING_RATE)
    data = librosa.amplitude_to_db(data, ref = NP.max)
    image = LIBD.specshow(data, sr = SAMPLING_RATE, x_axis = 's', y_axis = 'mel', ax = axes)   
    return image

def log_mel_spectrogram(data, axes, SAMPLING_RATE):
    data = librosa.feature.melspectrogram(data, sr = SAMPLING_RATE)
    data = librosa.amplitude_to_db(data, ref = NP.max)
    image = LIBD.specshow(data, sr = SAMPLING_RATE, x_axis = 's', y_axis = 'log', ax = axes)   
    return image

def Q_power_spectrogram(data, axes, SAMPLING_RATE):
    data = librosa.cqt(data, sr = SAMPLING_RATE)
    data = librosa.amplitude_to_db(NP.abs(data), ref = NP.max)
    image = LIBD.specshow(data, sr = SAMPLING_RATE, x_axis = 's', y_axis = 'cqt_hz', ax = axes) 
    return image

def linear_chromagram(data, axes, SAMPLING_RATE):
    data = librosa.feature.chroma_stft(data, sr = SAMPLING_RATE)
    data = librosa.amplitude_to_db(data, ref = NP.max)
    image = LIBD.specshow(data, sr = SAMPLING_RATE, x_axis = 's', y_axis = 'chroma', ax = axes) 
    return image

def Q_power_chromagram(data, axes, SAMPLING_RATE):
    data = librosa.feature.chroma_cqt(data, sr = SAMPLING_RATE)
    data = librosa.amplitude_to_db(data, ref = NP.max)
    image = LIBD.specshow(data, sr = SAMPLING_RATE, x_axis = 's', y_axis = 'chroma', ax = axes) 
    return image

def tempogram(data, axes, SAMPLING_RATE):
    data = librosa.onset.onset_strength(data, sr = SAMPLING_RATE)
    data = librosa.feature.tempogram(onset_envelope = data, sr = SAMPLING_RATE)
    image = LIBD.specshow(data, sr = SAMPLING_RATE, x_axis = 's', y_axis = 'tempo', ax = axes) 
    return image