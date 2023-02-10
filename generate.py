import os
import webdataset as wds
import torch
import torchaudio 
import pathlib
import collections
import tqdm
import json
import threading

import numpy as np

import simfile
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.notes.timed import time_notes

from utils import time2frame
from transform import OnsetTransform
from config import config

AUDIO_EXTENSIONS = (".wav", ".ogg", ".mp3")
VALID_NOTES = (NoteType.TAP, NoteType.HOLD_HEAD, NoteType.TAIL, NoteType.ROLL_HEAD)
INVALID_SHARD_CHARS = (".", "/", "\\", "*")

def strip_song_name(song_name):
    return "".join([c for c in song_name if c not in INVALID_SHARD_CHARS])


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


def generate_onset_mask(song_path : os.PathLike, difficulty : str, n_frames : int) -> np.ndarray:
    """Generate a mask for onsets"""
    sm, _ = simfile.opendir(song_path)

    # get chart from simfile.charts
    chart = next(filter(lambda c: c.difficulty == difficulty, sm.charts))

    note_data = NoteData(chart)
    timing_data = TimingData(sm, chart)

    # get note times
    note_times = np.array([
        timed_note.time for timed_note in time_notes(note_data, timing_data)
        if timed_note.note.note_type in VALID_NOTES
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

def get_songs() -> list:
    """Get a list of paths to songs"""
    songs = []
    for pack_name in config.dataset.packs:
        for song_path in pathlib.Path(config.paths.raw / pack_name).iterdir():
            songs.append(song_path)

    return songs

def get_shard_filename(split : str) -> str:
    name = config.dataset.shard.name
    zero_pad = config.dataset.shard.zero_pad

    return name.format(split=split, index="%0" + str(zero_pad) + "d") # TODO fix hack

def generate_shards(songs : list, split : str):
    """Generate shards for the dataset."""

    assert split in ("train", "valid"), "Split must be either 'train' or 'valid'"
    
    dataset_path = config.paths.shards / get_shard_filename(split)

    sink = wds.ShardWriter(str(dataset_path), maxcount=config.dataset.shard.count)
    transform = OnsetTransform()

    total_frames = 0

    for song_path in tqdm.tqdm(songs, desc=f"Generating shards - {split}"):
        audio_path = locate_audio(song_path)

        # load audio
        audio, sr = torchaudio.load(audio_path)

        # get features
        features = transform(audio, sr)

        # get total number of frames, add to total
        num_frames = features.shape[0]
        total_frames += num_frames

        # get difficulties
        difficulties = get_difficulties(song_path, config.charts.difficulties, config.charts.types)

        # create onset masks for each difficulty
        onsets = np.zeros((len(difficulties), num_frames), dtype=np.float32)
        for i, difficulty in enumerate(difficulties):
            onsets[i] = generate_onset_mask(song_path, difficulty, num_frames)

        # get the chart difficulties
        difficulties = np.array([config.charts.difficulties.index(difficulty) for difficulty in difficulties])

        # to tensors
        onsets = torch.from_numpy(onsets)
        difficulties = torch.from_numpy(difficulties)

        # save to webdataset
        sample = {
            "__key__": strip_song_name(song_path.name),
            "features.pth": features,
            "onsets.pth": onsets,
            "difficulties.pth":  difficulties,
            "num_frames.cls": features.shape[0],
        }

        sink.write(sample)
    
    sink.close()

def generate_dataset(train_size=0.8, threaded=True):

    songs = get_songs()

    np.random.seed(config.common.seed)
    np.random.shuffle(songs)

    split_idx = int(train_size * len(songs))
    train_songs = songs[:split_idx]
    valid_songs = songs[split_idx:]

    if threaded:
        t1 = threading.Thread(target=generate_shards, args=(train_songs, "train"))
        t2 = threading.Thread(target=generate_shards, args=(valid_songs, "valid"))

        t1.start()
        t2.start()

        t1.join()
        t2.join()

    else:
        generate_shards(train_songs, "train")
        generate_shards(valid_songs, "valid")

if __name__ == "__main__":
    generate_dataset()