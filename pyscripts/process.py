import os
import pathlib
import pickle
import multiprocessing
import tqdm
from tqdm.contrib.concurrent import process_map
import time

import numpy as np
import pandas as pd

import librosa

import simfile 
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.notes.timed import time_notes


SIMFILE_EXTENSIONS = (".ssc", ".sm")
AUDIO_EXTENSIONS = (".wav", ".ogg", ".mp3")
NOTES = (NoteType.TAP, NoteType.HOLD_HEAD, NoteType.TAIL, NoteType.ROLL_HEAD)

def clean_filename(filename):
    """Remove invalid characters from a filename for windows"""
    return "".join(c for c in filename if c not in r'\/:*?"<>|')


def locate_audio(song_path : pathlib.Path) -> pathlib.Path:
    """Locate audio file at a given path"""

    for file in os.listdir(song_path):
        for ext in AUDIO_EXTENSIONS:
            if file.endswith(ext):
                return song_path / file
                
    return None


def extract_features(config, audio):
    """Extract mel spectrogram features from audio"""
    
    features = []

    for n_fft in config.audio.n_ffts:
        mel = librosa.feature.melspectrogram(
            y=audio, 
            sr=config.audio.sample_rate, 
            n_fft=n_fft, 
            hop_length=config.audio.hop_length, 
            n_mels=config.audio.n_bins, 
            fmin=config.audio.fmin, 
            fmax=config.audio.fmax
        )
        
        if config.audio.log:
            mel = np.log(mel + config.audio.log_eps)

        features.append(mel)
    
    # stack features to be (C, F, T) (channels, features, time)
    features = np.stack(features, axis=0)
    # transpose to be (T, F, C)
    features = np.transpose(features, (2, 1, 0))
    
    return features

def generate_examples(config, features, sm):

    n_frames = features.shape[0]
    n_examples = n_frames - config.onset.past_context - config.onset.future_context - 1
    sequence_length = config.onset.past_context + config.onset.future_context + 1
    
    title = clean_filename(sm.title)
    
    # create a time vector for the features
    time = np.arange(n_frames) * config.audio.hop_length / config.audio.sample_rate

    for chart in sm.charts:
        if chart.difficulty not in config.charts.difficulties:
            continue
        
        if chart.stepstype not in config.charts.types:
            continue
        
        # time notes
        note_data = NoteData(chart)
        timing_data = TimingData(sm, chart)
                
        note_times = [
            timed_note.time for timed_note in time_notes(note_data, timing_data)
            if timed_note.note.note_type in NOTES
        ]
        
        # create an array to store the examples
        examples = np.zeros((n_examples, sequence_length, config.audio.n_bins, len(config.audio.n_ffts)), dtype=np.float32)
        labels = np.zeros((n_examples, 1), dtype=np.float32)
        
        # generate examples
        for i in range(config.onset.past_context + 1, n_frames - config.onset.future_context, 1):
            example = features[i - config.onset.past_context : i + config.onset.future_context + 1]
            
            # check if target frame is an onset
            note_in_example = False
            for note_time in note_times:
                if time[i] <= note_time <= time[i+1]:
                    note_in_example = True
                    break
                
            examples[i - config.onset.past_context - 1] = example
            labels[i - config.onset.past_context - 1] = note_in_example
        
        
        
        # save examples and labels
        examples_path = config.paths.examples / title / chart.difficulty / "examples.npy"
        labels_path = config.paths.examples / title / chart.difficulty / "labels.npy"
        
        if not examples_path.parent.exists():
            examples_path.parent.mkdir(parents=True)
    
        if not labels_path.parent.exists():
            labels_path.parent.mkdir(parents=True)
            
        np.save(examples_path, examples)
        np.save(labels_path, labels)


            
    # save number of examples to a text file
    n_examples_path = config.paths.examples / title / "n_examples.txt"

    with open(n_examples_path, "w") as f:
        f.write(str(n_examples))

def process_song(config, pack_name, song_name):
    song_path = config.paths.raw / pack_name / song_name

    # locate audio file
    audio_path = locate_audio(song_path)
    if audio_path is None:
        print(f"WARNING: {song_path} does not have an audio file")
        return

    # load audio file
    audio, sr = librosa.load(audio_path, sr=config.audio.sample_rate)
    assert sr == config.audio.sample_rate, "Sample rate mismatch"

    # extract features
    features = extract_features(config, audio) # (T, F, C)

    # load simfile
    sm, _ = simfile.opendir(song_path)

    # generate examples and save them
    generate_examples(config, features, sm)
    
def star(args):
    return process_song(*args)
    
def preprocess(config, num_threads=16):    
    total = sum(len(os.listdir(config.paths.raw / pack_name)) for pack_name in config.dataset.packs)

    arglist = []
    for pack_name in config.dataset.packs:
        pack_path = config.paths.raw / pack_name
        
        for song_name in os.listdir(pack_path):
            arglist.append((config, pack_name, song_name))



    r = process_map(star, arglist, max_workers=num_threads)
