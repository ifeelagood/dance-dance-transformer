import torchaudio
import torch
import pathlib
import os
import collections
import json

from utils import NOTES, AUDIO_EXTENSIONS, locate_audio, get_difficulties, get_onset_mask, samples2frames



def create_manifest(config):
    manifest = collections.OrderedDict() # preserve order

    for pack_name in config.dataset.packs:
        for song_path in pathlib.Path(config.paths.raw / pack_name).iterdir():
            # get audio path
            audio_path = locate_audio(song_path)

            # load audio
            audio, sr = torchaudio.load(audio_path)

            # calculate n_samples when resampled
            n_samples = int(audio.shape[1] * config.audio.sample_rate / sr)

            # get difficulties
            difficulties = get_difficulties(song_path, config.charts.difficulties, config.charts.types)


            # get name from audio
            song_name = song_path.name
            audio_name = audio_path.name

            # add to manifest
            manifest[song_name] = {
                "pack_name": pack_name,
                "audio_name": audio_name, 
                "n_samples": n_samples,
                "difficulties": difficulties,
            }
        

    with open(config.paths.manifest, "w") as f:    
        json.dump(manifest, f)

