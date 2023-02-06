import torch
import torchaudio

import numpy as np
import json

from config import config
from utils import *

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


class ExampleIterator:
    def __init__(self, manifest):
        self.manifest = manifest

        self.song = None
        self.difficulty = None
        self.features = None
        self.onset_mask = None

    def __len__(self):
        return np.sum([
            samples2frames(song["n_samples"]) * len(song["difficulties"]) 
            for song in self.manifest.values()
        ])

    def __iter__(self):
        for song_name, song in self.manifest.items():
            if song_name != self.song: 
                self.song = song_name
                self.features = get_features(config.paths.raw / song["pack_name"] / song_name / song["audio_name"])

            for difficulty in song["difficulties"]:
                if difficulty != self.difficulty:
                    self.difficulty = difficulty
                    self.onset_mask = get_onset_mask(config.paths.raw / song["pack_name"] / song_name, difficulty, self.features.shape[0])

                for frame_idx in range(config.onset.past_context, self.features.shape[0] - config.onset.future_context):
                    # get features
                    features = self.features[frame_idx - config.onset.past_context : frame_idx + config.onset.future_context + 1]

                    # get labels
                    labels = self.onset_mask[frame_idx - config.onset.past_context : frame_idx + config.onset.future_context + 1]

                    # yield example
                    yield features, labels

    def __next__(self):
        return next(self.__iter__())

    

def chunk_manifest(manifest, chunk_size):
    """Split manifest into chunks of size chunk_size"""

    chunks = []
    chunk = {}

    for song_name, song in manifest.items():
        chunk[song_name] = song

        if len(chunk) >= chunk_size:
            chunks.append(chunk)
            chunk = {}

    # add the last chunk
    if len(chunk) > 0:
        chunks.append(chunk)

    return chunks


class OnsetDataset(torch.utils.data.IterableDataset):
    def __init__(self, manifest):
        self.manifest = manifest

    def __len__(self):
        return len(ExampleIterator(self.manifest))
    

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # get the number of workers
        if worker_info is None:
            num_workers = 0
        else:
            num_workers = worker_info.num_workers

        if num_workers > 1:
            # split manifest into chunks
            chunks = chunk_manifest(self.manifest, len(self) // num_workers)

            # get the chunk for this worker
            chunk = chunks[torch.utils.data.get_worker_info().id]

            # return an iterator for the chunk
            return ExampleIterator(chunk)

        else:
            # return an iterator for the whole manifest
            return ExampleIterator(self.manifest)



        

if __name__ == "__main__":
    import time
    import tqdm
    train_manifest, valid_manifest = train_valid_split()

    train_dataset = OnsetDataset(train_manifest)
    valid_dataset = OnsetDataset(valid_manifest)

    def time_function(f):
        start = time.time()
        f()
        end = time.time()
        return end - start

    def single_process():
        # iterate with a single worker
        for features, labels in tqdm.tqdm(train_dataset, total=len(train_dataset)):
            pass
    
    def multi_process():
        dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4)
        for features, labels in tqdm.tqdm(dataloader, total=len(train_dataset)):
            pass

    

    print("Single process: {:.2f} seconds".format(time_function(single_process)))
    print("Multi process: {:.2f} seconds".format(time_function(multi_process)))