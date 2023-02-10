import os
import numpy as np
import pathlib

import simfile

from config import config

AUDIO_EXTENSIONS = (".mp3", ".ogg", ".wav")

def samples2frames(n_samples : int) -> int:
    """Convert the total number of samples to the number of feature frames"""
    return int(np.ceil(n_samples / config.audio.hop_length))

def time2frame(time : float) -> int:
    """Convert time in seconds to feature frame index"""
    return int(np.floor(time * config.audio.sample_rate / config.audio.hop_length)) # could be np.ceil

def frame2time(frame_idx : int) -> float:
    """Convert feature frame index to time in seconds"""
    return frame_idx * config.audio.hop_length / config.audio.sample_rate

