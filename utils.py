import os
import torch
import torchaudio
import tqdm
import numpy as np
import json
import pathlib

import simfile
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.notes.timed import time_notes

from config import config

NOTES = (NoteType.TAP, NoteType.HOLD_HEAD, NoteType.TAIL, NoteType.ROLL_HEAD)
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

def is_onset(frame_idx : int, onset_times : np.ndarray) -> bool:
    """Check if a frame is an onset."""
    return np.any(np.abs(onset_times - frame2time(frame_idx)) < (config.audio.hop_length / config.audio.sample_rate))

def locate_audio(song_path : os.PathLike) -> os.PathLike:
    """Locate audio file at a given path"""
    for file in os.listdir(song_path):
        for ext in AUDIO_EXTENSIONS:
            if file.endswith(ext):
                return pathlib.Path(song_path, file)
    return None

def get_difficulties(song_path, allowed_difficulties, allowed_types):
    sm, _ = simfile.opendir(song_path)

    return list(map(lambda chart : getattr(chart, "difficulty"), filter(lambda chart : chart.stepstype in allowed_types and chart.difficulty in allowed_difficulties, sm.charts)))

def get_onset_mask(song_path : os.PathLike, difficulty : str, n_frames : int) -> np.ndarray:
    """Generate a mask for onsets"""
    sm, _ = simfile.opendir(song_path)

    # get chart from simfile.charts
    chart = next(filter(lambda c: c.difficulty == difficulty, sm.charts))

    note_data = NoteData(chart)
    timing_data = TimingData(sm, chart)

    # get note times
    note_times = np.array([
        timed_note.time for timed_note in time_notes(note_data, timing_data)
        if timed_note.note.note_type in NOTES
    ])
    
    # calculate onset mask
    onset_mask = np.zeros(n_frames, dtype=np.float32)
    for note_time in note_times:
        frame_idx = time2frame(note_time)
    
        if frame_idx < 0:
            raise ValueError("Note time is before audio start")
        elif frame_idx >= n_frames:
            # raise ValueError("Note time is after audio end")
            continue
    
        onset_mask[frame_idx] = 1
    
    return onset_mask
