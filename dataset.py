import torch
import torchaudio
import webdataset as wds

import numpy as np
import json

from config import config
from utils import *
import glob


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

def is_dataset_updated(split):
    info_path = config.paths.webdataset / (split + ".json")
    
    if not info_path.exists():
        return False

    with open(info_path) as f:
        dataset_info = json.load(f)

    # check if dataset info matches config
    if dataset_info["audio"] != config.audio.__dict__:
        return False
    if dataset_info["packs"] != config.dataset.packs:
        return False
    
    if config.audio.normalize:
        with np.load(config.paths.stats) as stats:
            mean = torch.from_numpy(stats["mean"]).squeeze(0)
            std = torch.from_numpy(stats["std"]).squeeze(0)

        if dataset_info["mean"] != mean.numpy().tolist():
            return False
        if dataset_info["std"] != std.numpy().tolist():
            return False

    return True

def save_dataset_info(split, num_frames=None, mean=None, std=None):
    dataset_info = {
        "audio": config.audio.__dict__,
        "packs": config.dataset.packs,
        "mean": mean.numpy().tolist() if mean is not None else None,
        "std": std.numpy().tolist() if std is not None else None,
        "total_frames": num_frames,
    }

    with open(config.paths.webdataset / (split + ".json"), "w") as f:
        json.dump(dataset_info, f)


def create_webdataset(manifest, split):
    dataset_path = config.paths.webdataset / f"{split}-%06d.tar"

    sink = wds.ShardWriter(str(dataset_path), maxcount=config.dataloader.shard_size)
    transform = OnsetTransform()


    total_frames = 0
    for i, (song_name, song) in enumerate(tqdm.tqdm(manifest.items(), desc="Creating webdataset")):
        # load audio
        audio_path = config.paths.raw / song["pack_name"] / song_name / song["audio_name"]
        audio, sr = torchaudio.load(audio_path)

        # get features
        features = transform(audio, sr)

        # get total number of frames, add to total
        num_frames = features.shape[0] # total number of frames
        total_frames += num_frames

        # create onset and difficulty masks
        onsets = np.zeros((len(song["difficulties"]), num_frames), dtype=np.float32)
        difficulties = np.array([config.charts.difficulties.index(difficulty) for difficulty in song["difficulties"]])

        for i, difficulty in enumerate(song["difficulties"]):
            # get onset mask
            onsets[i] = get_onset_mask(config.paths.raw / song["pack_name"] / song_name, difficulty, num_frames)

        # onset and difficulty to tensor
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
    save_dataset_info(split, total_frames)


def sample_iterator(src):
    if config.audio.normalize:
        if not config.paths.stats.exists():
            raise FileNotFoundError("config.audio.normalize is True, but stats file does not exist. Please generate with utils.analyse_dataset()")

        with np.load(config.paths.stats) as stats:
            mean = torch.from_numpy(stats["mean"]).squeeze(0)
            std = torch.from_numpy(stats["std"]).squeeze(0)

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

def get_dataset(split, batch_size=None):
    shards = list(map(str, config.paths.webdataset.glob(f"{split}-" + "[0-9]" * 6 + ".tar")))

    dataset = wds.WebDataset(shards, nodesplitter=wds.split_by_node)
    dataset = dataset.compose(wds.split_by_worker)
    dataset = dataset.decode()
    dataset = dataset.to_tuple("features.pth", "onsets.pth", "difficulties.pth", "num_frames.cls")
    dataset = dataset.compose(sample_iterator)
    
    if batch_size:
        dataset = dataset.batched(batch_size)

    return dataset

if __name__ == "__main__":
    if not config.paths.webdataset.exists():
        config.paths.webdataset.mkdir()

    train_manifest, valid_manifest = train_valid_split()
    
    create_webdataset(train_manifest, "train")
    create_webdataset(valid_manifest, "valid")

    train_dataset = get_dataset("train")
    valid_dataset = get_dataset("valid")
    
    for i, (feature, difficulty, onset) in tqdm.tqdm(enumerate(train_dataset)):
        print(feature.shape, difficulty.shape, onset.shape)
        if i == 10:
            break