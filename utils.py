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
    onset_mask = np.zeros(n_frames, dtype=np.bool)
    for note_time in note_times:
        frame_idx = time2frame(note_time)
        
        if frame_idx < 0:
            raise ValueError("Note time is before audio start")
        elif frame_idx >= n_frames:
            raise ValueError("Note time is after audio end")
        
        onset_mask[frame_idx] = True
    
    return onset_mask

class StatsRecorder:
    def __init__(self, red_dims : tuple):
        """
        Accumulates normalization statistics across mini-batches.
        ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        self.red_dims = red_dims # which mini-batch dimensions to average over
        self.nobservations = 0   # running number of observations

    def update(self, data : torch.Tensor):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std  = data.std (dim=self.red_dims,keepdim=True)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims do not match previous observations.")
            
            # find mean of new mini batch
            new_mean = data.mean(dim=self.red_dims, keepdim=True)
            new_std = data.std(dim=self.red_dims, keepdim=True)
            
            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*new_mean # update running mean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*new_std**2 +m*n/(m+n)**2 * (tmp - new_mean)**2 # update running variance
            self.std  = torch.sqrt(self.std)
                                 
            # update total number of seen samples
            self.nobservations += n

def analyse_dataset(dataloader : torch.utils.data.DataLoader, max=None) -> None:
    """Get the mean and standard deviation of features in the dataloader"""    
    # input: (B, T, F, C) (batch, time, bins, channels)
    
    # set normalize and precache to False
    config.audio.normalize = False
    
    
    stats = StatsRecorder((0, 1, 3)) # z score


    for i, (features, _, _) in enumerate(tqdm.tqdm(dataloader, desc='Analysing dataset', total=max if max is not None else len(dataloader))):
        stats.update(features)

        if max is not None and i >= max:
            break
    
        
    # save to config.paths.stats as packed npz

    np.savez_compressed(
        config.paths.stats,
        mean=stats.mean.numpy(),
        std=stats.std.numpy(),
    )
    
    print(f"Saved stats to {config.paths.stats}")

    # print shape
    print(f"Mean shape: {stats.mean.shape}")
    print(f"Std shape: {stats.std.shape}")
    
    # set normalize and precache to True
    config.audio.normalize = True
    