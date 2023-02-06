import os
import torch
import torchaudio
import numpy as np

import simfile
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.notes.timed import time_notes

from config import config

NOTES = (NoteType.TAP, NoteType.HOLD_HEAD, NoteType.TAIL, NoteType.ROLL_HEAD)
AUDIO_EXTENSIONS = (".mp3", ".ogg", ".wav")

def samples2frames(n_samples : int) -> int:
    """Convert the number of samples to number frames, accounting for context"""
    return int(np.ceil(n_samples / config.audio.hop_length)) - (config.onset.past_context + config.onset.future_context)

def frame2time(frame_idx : int) -> float:
    """Convert feature frame index to time in seconds"""
    return frame_idx * config.audio.hop_length / config.audio.sample_rate

def is_onset(frame_idx : int, onset_times : np.ndarray) -> bool:
    """Check if a frame is an onset."""
    return np.any(np.abs(onset_times - frame2time(frame_idx)) < config.onset.threshold)

def locate_audio(song_path : os.PathLike) -> os.PathLike:
    """Locate audio file at a given path"""
    for file in os.listdir(song_path):
        for ext in AUDIO_EXTENSIONS:
            if file.endswith(ext):
                return os.path.join(song_path, file)
    return None

def load_audio(audio_path : os.PathLike) -> torch.Tensor:
    """Loads audio file from given path, resamples and converts to mono if needed."""
    
    audio, sr = torchaudio.load(audio_path)

    # resample if needed
    if sr != config.audio.sample_rate:
        audio = torchaudio.transforms.Resample(sr, config.audio.sample_rate)(audio)

    # convert to mono
    if len(audio.shape) > 1:
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=False)
        else:
            audio = audio.squeeze(0)

    return audio

def extract_features(audio : torch.Tensor) -> torch.Tensor:
    """Calculate mel spectrogram features from audio"""

    features = []
    for n_fft in config.audio.n_ffts:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.audio.sample_rate,
            n_fft=n_fft,
            hop_length=config.audio.hop_length,
            n_mels=config.audio.n_bins,
            f_min=config.audio.fmin,
            f_max=config.audio.fmax
        )(audio)
        
        if config.audio.log:
            mel = torch.log(mel + config.audio.log_eps)

        features.append(mel)

    # stack features to be (C, F, T) (channels, features, time)
    features = torch.stack(features, dim=0)
    # transpose to be (T, F, C)
    features = features.permute(2, 1, 0)

    return features

def get_onset_mask(song_path : os.PathLike, difficulty : str, n_frames : int) -> np.ndarray:
    """Get the times of all notes in a song"""
    sm, _ = simfile.opendir(song_path)

    # get chart from simfile.charts
    chart = next(filter(lambda c: c.difficulty == difficulty, sm.charts))

    note_data = NoteData(chart)
    timing_data = TimingData(sm, chart)

    # get note times
    times = [
        timed_note.time for timed_note in time_notes(note_data, timing_data)
        if timed_note.note.note_type in NOTES
    ]

    times = np.array(times)

    mask = np.zeros(n_frames, dtype=np.bool)
    for time in times:
        frame = frame2time(time)
        mask[int(frame)] = True
    
    return mask

def get_features(audio_path : os.PathLike) -> torch.Tensor:
    """Get features for a song"""
    audio = load_audio(audio_path)
    features = extract_features(audio)

    return features