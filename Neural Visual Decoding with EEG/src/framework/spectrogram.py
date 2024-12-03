import torch
import torch.nn as nn

class Spectrogram_Module(nn.Module):
    def __init__(self, n_fft=256, hop_length=64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        B = x.shape[0] # batch size

        batch_spectrogram = []
        for batch in range(B):
            spec = torch.stft(x[batch], n_fft=256, hop_length=64, normalized=True, window=torch.hann_window(256), return_complex=True)
            spec = torch.sqrt(torch.real(spec)**2 + torch.imag(spec)**2)
            spec = spec.reshape(spec.size(0), -1, spec.size(2), 3)

            batch_spectrogram.append(spec)
        batch_spectrogram = torch.stack(batch_spectrogram,dim=0) # size [B, N, 43, 7, 3]
        return batch_spectrogram

