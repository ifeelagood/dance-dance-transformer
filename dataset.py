import torch
import torchaudio
import webdataset as wds

import tqdm

import numpy as np
import json

from config import config
from utils import *
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

class StatsRecorder:
    """Accumulate normalisation statistics across mini-batches."""
    def __init__(self, dim : tuple):
        self.dim = dim
        self.n_observations = None
        self.n_dimensions = None

    def update(self, data : torch.Tensor):
        # initialise stats and dimensions on first batch
        if self.n_observations is None:
            self.mean = data.mean(dim=self.dim, keepdim=True)
            self.std = data.std(dim=self.dim, keepdim=True)
            self.n_observations = data.shape[0]
            self.n_dimensions = data.shape[1]

            return None, None

        else:
            if data.shape[1] != self.n_dimensions:
                raise ValueError("Data dimensions do not match previous observations")

            # calculate minibatch mean
            new_mean = data.mean(dim=self.dim, keepdim=True)
            new_std = data.std(dim=self.dim, keepdim=True)

            # update number of observations
            m = float(self.n_observations)
            n = data.shape[0]

            # save old mean and std
            old_mean = self.mean
            old_std = self.std

            self.mean = m/(m+n)*old_mean + n/(m+n)*new_mean # update running mean
            self.std  = m/(m+n)*old_std**2 + n/(m+n)*new_std**2 +m*n/(m+n)**2 * (old_mean - new_mean)**2 # update running variance
            self.std  = torch.sqrt(self.std)

            # update num observations
            self.n_observations += n

            # return the difference between the old and new
            mean_delta = torch.sum(torch.abs(self.mean - old_mean)).item()
            std_delta = torch.sum(torch.abs(self.std - old_std)).item()

            return mean_delta, std_delta

    def save(self, path, squeeze=True):
        mean = self.mean.numpy()
        std = self.std.numpy()

        if squeeze:
            mean = np.squeeze(mean, axis=0)
            std = np.squeeze(std, axis=0)

        np.savez_compressed(path, mean=mean, std=std)


def analyse_dataset():
    # set normalise to false
    config.audio.normalize = False

    # create recorder
    recorder = StatsRecorder(dim=(0, 1, 3)) 

    dataset = get_dataset(split="train", batch_size=512)

    try:
        print("Recording stats (interrupt to save)")
        pbar = tqdm.tqdm(dataset, desc="Anaylsing Dataset")
        for batch in pbar:
            features = batch[0]
            mean_delta, std_delta = recorder.update(features)

            pbar.set_postfix({"Δmean": mean_delta, "Δstd": std_delta})
            
        pbar.close()
        recorder.save(config.paths.stats)
    
    except KeyboardInterrupt:
        print("Δmean={mean_delta}\tΔstd={std_delta}")
        print("Recieved Keyboard interrupt. saving...")

        recorder.save(config.paths.stats)
        return 


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
    shards = list(map(str, config.paths.shards.glob(f"{split}-" + "[0-9]" * config.dataset.shard.zero_pad + ".tar")))

    dataset = wds.WebDataset(shards, nodesplitter=wds.split_by_node)
    dataset = dataset.compose(wds.split_by_worker)
    dataset = dataset.decode()
    dataset = dataset.to_tuple("features.pth", "onsets.pth", "difficulties.pth", "num_frames.cls")
    dataset = dataset.compose(sample_iterator)
    
    if batch_size:
        dataset = dataset.batched(batch_size)

    return dataset

if __name__ == "__main__":
    train_manifest, valid_manifest = train_valid_split()
    train_dataset = get_dataset("valid")

    print("Train dataset:")

    with open(config.paths.shards / "train.json", 'r') as f:
        dataset_info = json.load(f)


    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0)