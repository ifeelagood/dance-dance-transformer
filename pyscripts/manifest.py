import torchaudio
import torch
import pathlib
import os
import simfile
import collections
import json

AUDIO_EXTENSIONS = (".wav", ".ogg", ".mp3")
 
def locate_audio(song_path : pathlib.Path) -> pathlib.Path:
    """Locate audio file at a given path"""

    for file in os.listdir(song_path):
        for ext in AUDIO_EXTENSIONS:
            if file.endswith(ext):
                return song_path / file
                
    return None

def count_samples(audio_path, sample_rate):

    audio, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)

    if len(audio.shape) > 1:
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=False)
        else:
            audio = audio.squeeze(0)

    return audio.shape[0]


def get_difficulties(song_path, allowed_difficulties, allowed_types):
    sm, _ = simfile.opendir(song_path)

    return list(filter(lambda chart : chart.stepstype in allowed_types and chart.difficulty in allowed_difficulties, sm.charts))


def create_manifest(config):
    manifest = collections.OrderedDict() # preserve order

    for pack_name in config.dataset.packs:
        for song_path in pathlib.Path(config.paths.raw / pack_name).iterdir():
            # get audio path
            audio_path = locate_audio(song_path)


            # get number of samples
            n_samples = count_samples(audio_path, config.audio.sample_rate)

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
                "difficulties": difficulties
            }
        
            print(manifest[song_name])

    with open(config.paths.manifest, "w") as f:    
        json.dump(manifest, f)

