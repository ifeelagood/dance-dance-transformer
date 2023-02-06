import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio

import simfile
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.notes.timed import time_notes

from config import config

NOTES = (NoteType.TAP, NoteType.HOLD_HEAD, NoteType.TAIL, NoteType.ROLL_HEAD)

def load_audio(path) -> torch.Tensor:
    audio, sr = torchaudio.load(path)

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

def get_features(audio : torch.Tensor) -> torch.Tensor:
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

def get_note_times(song_path, difficulty) -> np.ndarray:
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

    return np.array(times)


def samples2frames(n_samples : int) -> int:
    """Convert the number of samples to number of mel log spectrogram frames"""
    return int(np.ceil(n_samples / config.audio.hop_length)) - (config.onset.past_context + config.onset.future_context)

def frame2time(frame_idx : int) -> float:
    """Convert mel log spectrogram frame index to time in seconds"""
    return frame_idx * config.audio.hop_length / config.audio.sample_rate

def is_onset(frame_idx : int, onset_times : np.ndarray) -> bool:
    """Check if a frame is an onset. Thresold should be set to the length of the longest note"""
    return np.any(np.abs(onset_times - frame2time(frame_idx)) < config.onset.threshold)

class ExampleGenerator:
    def __init__(self):
        with open(config.paths.manifest, 'r') as f:
            self.manifest = json.load(f)

        self.song_name = None
        self.difficulty = None
        self.features = None
        self.onset_times = None


    def __len__(self):
        return np.sum(
            [
                samples2frames(song["n_samples"]) * len(song["difficulties"]) 
                for song in self.manifest.values()
            ]
        )

    def __iter__(self):
        for song_name, song in self.manifest.items():
            if song_name != self.song_name: 
                # if new song, load new features and simfile
                self.song_name = song_name
                self.difficulty = None

                # load features
                audio = load_audio(config.paths.raw / song["pack_name"] / song_name / song["audio_name"])
                self.features = get_features(audio)
                self.feature_time = np.arange(self.features.shape[0]) * config.audio.hop_length / config.audio.sample_rate
            
            for difficulty in song["difficulties"]:
                if difficulty != self.difficulty:
                    self.difficulty = difficulty
                    self.onset_times = get_note_times(config.paths.raw / song["pack_name"] / song_name, difficulty)

                for frame_idx in range(config.onset.past_context, self.features.shape[0] - config.onset.future_context):
                    # get features
                    features = self.features[frame_idx - config.onset.past_context : frame_idx + config.onset.future_context + 1]
                    label = is_onset(frame_idx, self.onset_times)

                    yield features, label

if __name__ == "__main__":
    import time
    import tqdm
    gen = ExampleGenerator()
    start = time.time()

    for i, (features, label) in enumerate(tqdm.tqdm(gen)):
        pass

    print(f"Generated {i} examples in {time.time() - start} seconds")


