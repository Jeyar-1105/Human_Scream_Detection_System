# utils/audio_utils.py
import numpy as np
import librosa

def extract_features(file_path, sr=22050, n_mfcc=40, max_len=100):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    return np.expand_dims(mfcc, axis=(0, -1))  # Shape: (1, 40, 100, 1)
