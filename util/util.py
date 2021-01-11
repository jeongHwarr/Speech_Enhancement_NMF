import os
import librosa
import numpy as np
import soundfile

def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)
    
def mag_spectro_to_log_spectro(magnituded_spectrogram):
    return np.log(magnituded_spectrogram*magnituded_spectrogram)

