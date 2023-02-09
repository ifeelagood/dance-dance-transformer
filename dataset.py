import torch
import torchaudio
import webdataset as wds

import numpy as np
import json

from config import config
from utils import *
import glob
import time
import datetime


from transform import OnsetTransform

def train_valid_split(train_size=0.8):
    """Split manifest into train and validation sets"""

    with open(config.paths.manifest) as f:
        manifest = json.load(f)


    # shuffle by unique song names
    unique_songs = list(set(manifest.keys()))

    np.random.seed(config.common.seed)
    np.random.shuffle(unique_songs)

    # split into new manifest
    train_manifest = {}
    valid_manifest = {}

    for song_name in unique_songs:
        if len(train_manifest) < len(unique_songs) * train_size:
            train_manifest[song_name] = manifest[song_name]
        else:
            valid_manifest[song_name] = manifest[song_name]

    return train_manifest, valid_manifest


def get_shard_filename(split : str) -> str:
    name = config.dataset.shard.name
    zero_pad = config.dataset.shard.zero_pad

    return name.format(split=split, index="%0" + str(zero_pad) + "d") # TODO fix hack

def get_info() -> dict:
    info = {
        "created_at": datetime.datetime.now(),
        "shard": config.dataset.shard.__dict__,
        "audio": config.audio.__dict__
    }

    return info

def has_info_changed() -> bool:
    """Returns True if generate information has changed."""

    current_info = get_info()

    with open(config.paths.info, 'r') as f:
        generated_info = json.load(f)

    return current_info["shard"] == generated_info["shard"] and current_info["audio"] == generated_info["audio"]


def generate_shards(manifest : dict, split : str):
    """Generate shards for the dataset."""

    assert split in ("train", "valid"), "Split must be either 'train' or 'valid'"

    dataset_path = config.paths.shards / get_shard_filename(split)

    sink = wds.ShardWriter(str(dataset_path), maxcount=config.dataset.shard.size)
    transform = OnsetTransform()

    total_frames = 0
    for i, (song_name, song) in enumerate(tqdm.tqdm(manifest.items(), desc=f"Creating dataset '{split}'")):
        # get path to song
        song_path = config.paths.raw / song["pack_name"] / song_name
        
        # load audio
        audio, sr = torchaudio.load(song_path / song["audio_name"])

        # get features
        features = transform(audio, sr)

        # get total number of frames, add to total
        num_frames = features.shape[0]
        total_frames += num_frames

        # create onset masks for each difficulty
        onsets = np.zeros((len(song["difficulties"]), num_frames), dtype=np.float32)
        for i, difficulty in enumerate(song["difficulties"]):
            onsets[i] = get_onset_mask(song_path, difficulty, num_frames)

        # get the chart difficulties
        difficulties = np.array([config.charts.difficulties.index(difficulty) for difficulty in song["difficulties"]])

        # to tensors
        onsets = torch.from_numpy(onsets)
        difficulties = torch.from_numpy(difficulties)

        # save to webdataset
        sample = {
            "__key__": song_name,
            "features.pth": features,
            "onsets.pth": onsets,
            "difficulties.pth":  difficulties,
            "num_frames.cls": features.shape[0],
        }

        sink.write(sample)
    
    sink.close()

def delete_shards():
    for file in config.paths.shards.iterdir():
        if file.is_file():
            file.unlink()



def sample_iterator(src):
    if config.audio.normalize:
        if not config.paths.stats.exists():
            raise FileNotFoundError("config.audio.normalize is True, but stats file does not exist. Please generate with utils.analyse_dataset()")

        with np.load(config.paths.stats) as stats:
            mean = torch.from_numpy(stats["mean"])
            std = torch.from_numpy(stats["std"])

    for (features, onsets, difficulties, num_frames) in src:    
        for i in range(difficulties.shape[0]):
            for j in range(config.onset.context_radius, num_frames - config.onset.context_radius):
                feature = features[j - config.onset.context_radius : j + config.onset.context_radius + 1]
                onset = onsets[i, j]

                # one hot difficulty
                difficulty = torch.zeros(len(config.charts.difficulties))
                difficulty[difficulties[i]] = 1
                
                # normalize
                if config.audio.normalize:
                    feature = (feature - mean) / std

                yield feature, difficulty, onset

def get_dataset(split):
    shards = list(map(str, config.paths.shards.glob(f"{split}-" + "[0-9]" * 6 + ".tar")))

    dataset = wds.WebDataset(shards, nodesplitter=wds.split_by_node)
    dataset = dataset.compose(wds.split_by_worker)
    dataset = dataset.decode()
    dataset = dataset.to_tuple("features.pth", "onsets.pth", "difficulties.pth", "num_frames.cls")
    dataset = dataset.compose(sample_iterator)

    return dataset


if __name__ == "__main__":
    if not config.paths.shards.exists():
        config.paths.shards.mkdir()

    train_manifest, valid_manifest = train_valid_split()
    
    
    train_dataset = get_dataset("valid")

    print("Train dataset:")

    with open(config.paths.shards / "train.json", 'r') as f:
        dataset_info = json.load(f)


    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0)

