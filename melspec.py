""" Compute Mel spectrograms of wave files.

This module uses melspec code from NVIDIA's BigVGAN
(https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py), which facilitates
subsequent audio resynthesis via TTS.

N.B. This is legacy code from our 2024 experiments; hopefully we can
directly integrate melspec and TTS code from the NeMo[tts] codebase.

Copyright (c) 2022 NVIDIA CORPORATION. 
  Licensed under the MIT license.
Adapted from https://github.com/jik876/hifi-gan under the MIT license.
  LICENSE is in incl_licenses directory.
"""
from librosa.filters import mel as librosa_mel_fn
import torch
import torchaudio.functional as F


N_MELS = 80
N_FFT = 1024
HOP_LEN = 256
WIN_LEN = 1024
BVG_SR = 22050
F_MIN = 0.0
F_MAX = 8000.0


def dynamic_range_compression_torch(S, C=1, clip_val=1e-5):
    """ Compress dynamic range of magnitude spectrogram."""
    return torch.log(torch.clamp(S, min=clip_val) * C)


# N.B. these params are imported from BigVGAN (see docstring)
def mel_spectrogram(y, n_fft: int = 1024, num_mels: int = 80,
                    samp_rate: int = 22050, hop_size: int = 256,
                    win_size: int = 1024, fmin: float = 0.0,
                    fmax: float = 8000.0, center: bool = False):
    """ Compute mel spectrogram of input using librosa.
    
    N.B. the default params here generate melspecs that are narrow-band and 
    quite "coarse" (n_mels=80), so they don't display pretty (like Praat defaults),
    but these values are standard in speech processing applications
    (and we needed them for audio synthesis in our 2024 experiments).
    """
    BIGVGAN_SR = 22050
    # resample to required rate; no-op if rates match
    y_resamp = F.resample(y, samp_rate, BIGVGAN_SR)

    if torch.min(y_resamp) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y_resamp) > 1.:
        print('max value is ', torch.max(y))
    
    mel_basis = {}
    hann_window = {}

    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=samp_rate, n_fft=n_fft,
                             n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y_resamp.device)] = torch.from_numpy(mel).float().to(y_resamp.device)
        hann_window[str(y_resamp.device)] = torch.hann_window(win_size).to(y_resamp.device)

    y_pad = torch.nn.functional.pad(y_resamp.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y_pad = y_pad.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(input=y_pad, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
                      window=hann_window[str(y_pad.device)],
                      center=center, pad_mode='reflect',
                      normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y_pad.device)], spec)
    spec = dynamic_range_compression_torch(spec)

    return spec
