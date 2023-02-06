import os

import torch
import torchaudio
import pytorch_lightning as pl

from config import config

class OnsetTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, audio, sr):
        # resample if needed
        if sr != config.audio.sample_rate:
            audio = torchaudio.transforms.Resample(sr, config.audio.sample_rate)(audio)

        # convert to mono
        if len(audio.shape) > 1:
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=False)
            else:
                audio = audio.squeeze(0)

        # calculate features
        features = []
        for n_fft in config.audio.n_ffts:
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=config.audio.sample_rate,
                n_fft=n_fft,
                hop_length=config.audio.hop_length,
                n_mels=config.audio.n_bins,
                f_min=config.audio.fmin,
                f_max=config.audio.fmax
            )(audio)
            
            if config.audio.log:
                mel = torch.log(mel + config.audio.log_eps)

            features.append(mel)

        # stack features to be (C, F, T) (channels, features, time)
        features = torch.stack(features, dim=0)

        # transpose to (T, F, C)
        features = features.permute(2, 1, 0)
        
        # normalize features
        features = (features - config.audio.mean) / config.audio.std

        
        return features
        