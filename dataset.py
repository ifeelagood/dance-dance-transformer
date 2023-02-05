import os
import pickle
import numpy as np
import torch

from config import config



def get_indices():
    for song_name in os.listdir(config.paths.examples):
        for difficulty_name in os.listdir(config.paths.examples / song_name):
            for example_index in range(int(open(config.paths.examples / song_name / "n_examples.txt").read())):
                yield song_name, difficulty_name, example_index

def train_test_split(train_size=0.8, random_state=42):
    songs = list(os.listdir(config.paths.examples))
    
    np.random.seed(random_state)
    np.random.shuffle(songs)
    
    train_songs = songs[:int(len(songs) * train_size)]
    valid_songs = songs[int(len(songs) * train_size):]

    train_indices = []
    valid_indices = []
    
    for song_name, difficulty_name, example_index in get_indices():
        if song_name in train_songs:
            train_indices.append((song_name, difficulty_name, example_index))
        else:
            valid_indices.append((song_name, difficulty_name, example_index))
            
    return train_indices, valid_indices

            
class StepmaniaDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):

        if train:
            self.indices = train_test_split()[0]
        else:
            self.indices = train_test_split()[1]
        
        self.example = None
        self.label = None
        self.song_name = None
        self.difficulty_name = None
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        song_name, difficulty_name, example_index = self.indices[index]
        
        if self.song_name != song_name or self.difficulty_name != difficulty_name:
            self.song_name = song_name
            self.difficulty_name = difficulty_name
            self.example = None
            
        if self.example is None:
            example_path = config.paths.examples / song_name / difficulty_name / "examples.npy"
            label_path = config.paths.examples / song_name / difficulty_name / "labels.npy"
            
            self.example = np.load(example_path)
            self.label = np.load(label_path)
            
        
        return self.example[example_index], self.label[example_index]
    
dataset = StepmaniaDataset()
print(len(dataset))

# time dataset
import time
 
start = time.time()
for i in range(len(dataset)):
    print(dataset[i])
    
print(time.time() - start)