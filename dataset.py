import torch
import torchaudio

import numpy as np
import json

from config import config
from utils import *


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

def feature_generator(manifest):
    """"""
    current_song = None
    current_difficulty = None
    current_features = None
    current_onset_mask = None
    transform = OnsetTransform()
    
    
    for song_name, song in manifest.items():
        if song_name != current_song:
            current_song = song_name
            current_features, sr = torchaudio.load(config.paths.raw / song["pack_name"] / song_name / song["audio_name"])
            current_features = transform(current_features, sr)
            current_difficulty = None # reset difficulty

        for difficulty in song["difficulties"]:
            if difficulty != current_difficulty: 
                current_difficulty = difficulty
                current_onset_mask = get_onset_mask(config.paths.raw / song["pack_name"] / song_name, difficulty, song["n_samples"]) # use n_samples rather than current_features.shape[0].
                
            for frame_idx in range(0, current_features.shape[0] - config.onset.sequence_length):
                # get features
                features = current_features[frame_idx : frame_idx + config.onset.sequence_length]

                # get single label
                label = current_onset_mask[frame_idx + config.onset.sequence_length - 1] # last frame in sequence.

                # label to np float32
                label = torch.from_numpy(np.array(label, dtype=np.float32))


                # yield example
                yield features, label

    
class OnsetDataset(torch.utils.data.IterableDataset):
    def __init__(self, manifest):
        super().__init__()
        self.manifest = manifest

    def __len__(self):
        return np.sum([
            samples2frames(song["n_samples"]) * len(song["difficulties"]) 
            for song in self.manifest.values()
        ])

    def __iter__(self):
        return feature_generator(self.manifest)


def get_chunk(manifest, worker_id, num_workers):

    # cut manifest into a chunk
    n_songs = len(manifest)
    chunk_size = n_songs // num_workers # integer division
    overflow = n_songs % num_workers    # remainder
    
    # get start and end indices
    start_idx = worker_id * chunk_size 
    end_idx = start_idx + chunk_size
    
    # add overflow to last chunk
    if worker_id == num_workers - 1:
        end_idx += overflow

    # get the chunk
    chunk = {k: manifest[k] for k in list(manifest.keys())[start_idx:end_idx]}
    
    return chunk


# to be used above with num_workers > 0
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None: # single process
        return
    
    if worker_info.num_workers > 1:
        chunk = get_chunk(worker_info.dataset.manifest, worker_info.id, worker_info.num_workers)
        worker_info.dataset.manifest = chunk
    


if __name__ == "__main__":
    import tqdm
    train_manifest, valid_manifest = train_valid_split()

    train_dataset = OnsetDataset(train_manifest)
    valid_dataset = OnsetDataset(valid_manifest)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=512,
        num_workers=0
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=512,
        num_workers=0
    )
    
    print(config.stats)