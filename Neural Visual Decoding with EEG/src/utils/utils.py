import torch
import torch.nn as nn

from scipy.signal import butter, lfilter

class ButterworthFilter(nn.Module):
    def __init__(self, order, lowcut, highcut, fs, axis=-1):
        super().__init__()
        self.axis = axis
        self.b, self.a = butter(order, [lowcut, highcut], btype='band', fs=fs)
    
    def forward(self, x):
        self.b = torch.tensor(self.b, dtype=x.dtype, device=x.device)
        self.a = torch.tensor(self.a, dtype=x.dtype, device=x.device)

        b = self.b.cpu()
        a = self.a.cpu()
        sig = x.cpu()
        y = lfilter(b, a, sig)
        y = torch.tensor(y)
        return y


class NotchFilter(nn.Module):
    def __init__(self, order, lowcut, highcut, freq, Q, fs, axis=-1):
        super().__init__()
        # First, bandpass filter with Butterworth
        self.bwf = ButterworthFilter(order, lowcut, highcut, fs, axis)
        # Second, apply Notch filter
        self.axis = axis
        self.b, self.a = butter(order, [freq - Q, freq + Q], btype='bandstop', fs=fs)
    
    def forward(self, x):
        self.b = torch.tensor(self.b, dtype=x.dtype, device=x.device)
        self.a = torch.tensor(self.a, dtype=x.dtype, device=x.device)

        # Recall we pass data in batch
        B = x.shape[0] # batch size

        x = self.bwf(x)
        b = self.b.cpu()
        a = self.a.cpu()
        sig = x.cpu()
        y = lfilter(b, a, sig)
        y = torch.tensor(y)
        return y


class MinMaxNormalizer():
    def __init__(self):
        self.min = None
        self.range = None

    def __call__(self, x):
        if self.min is None:
            self.min = x.min(dim=-1, keepdim=True)[0]
            self.range = x.max(dim=-1, keepdim=True)[0] - self.min
        return 2 * (x - self.min) / self.range - 1


class ButterworthFilterBank(nn.Module):
    def __init__(self, order, cut_frequencies, fs, f_notch, Q_notch, axis=-1):
        super().__init__()
        self.axis = axis
        self.num_filters = len(cut_frequencies) - 1
        self.filters = nn.ModuleList()
        for i in range(self.num_filters):
            lowcut, highcut = cut_frequencies[i], cut_frequencies[i + 1]
            if 50 >= lowcut and 50 <= highcut:
                self.filters.append(NotchFilter(order, lowcut, highcut, f_notch, Q_notch, fs, axis))
            else:
                self.filters.append(ButterworthFilter(order, lowcut, highcut, fs, axis))
        self.filter_for_spectrogram = NotchFilter(order, 14, 71, f_notch, Q_notch, fs, axis)
        #self.normalizer = MinMaxNormalizer()
    
    def forward(self, x):
        # Recall we pass data in batch
        B = x.shape[0] # batch size

        batch_output = []
        for batch in range(B):

            output = []
            for i in range(self.num_filters):
                elem = torch.tensor(self.filters[i](x[batch]))
                output.append(elem)

            output = torch.stack(output, dim=0)
            batch_output.append(output)

        batch_output = torch.stack(batch_output,dim=0)
        normalizer = MinMaxNormalizer()
        batch_output = normalizer(batch_output)
        
        
        batch_output_spec = []
        for batch in range(B):
            
            output_spec = self.filter_for_spectrogram(x[batch])
            batch_output_spec.append(output_spec)
            
        batch_output_spec = torch.stack(batch_output_spec,dim=0)
        
        
        return batch_output, batch_output_spec

