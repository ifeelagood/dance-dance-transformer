import torch
import webdataset as wds

import tqdm

import numpy as np
import json

from config import config
from utils import *
import datetime


from transform import OnsetTransform


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

def get_dataset(split):
    shards = list(map(str, config.paths.shards.glob(f"{split}-" + "[0-9]" * config.dataset.shard.zero_pad + ".tar")))

    dataset = wds.WebDataset(shards, nodesplitter=wds.split_by_node)
    dataset = dataset.compose(wds.split_by_worker)
    dataset = dataset.decode()
    dataset = dataset.to_tuple("features.pth", "onsets.pth", "difficulties.pth", "num_frames.cls")
    dataset = dataset.compose(sample_iterator)
    
    return dataset

def get_dataloader(dataset, batch_size, batched_dataloder=True, **kwargs):

    if not batched_dataloder:
        dataset.batched(batch_size)
        batch_size = None

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs)

    return dataloader

if __name__ == "__main__":
    dataset = get_dataset("train")

    dataloader = get_dataloader(dataset, batch_size=256, batched_dataloder=False, num_workers=config.dataloader.num_workers)

    for _ in tqdm.tqdm(dataloader):
        pass