""" phone_spec_rnn.pyv"""
import torch.nn as nn


class PhoneSpecCNN(nn.Module):
    """ Convolution neural net (CNN) for classifying short spectra into vowel categories.
    
    Architecture: nn.Conv2d layers plus a Pooling layer and a Linear layer to generate the logits.
    See https://www.kaggle.com/code/kyrobc/audio-mnist-classifier-with-98-accuracy-15-min
    """ 
    def __init__(self):
        super().__init__()
        self.conv_layers = []
        
        def build_conv_layer(in_channels, out_channels, kern, strd, pad):
            conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=kern if type(kern) is tuple else (kern, kern), 
                             stride=strd if type(strd) is tuple else (strd, strd), 
                             padding=pad if type(pad) is tuple else (pad, pad))
            relu = nn.LeakyReLU()
            # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
            nn.init.kaiming_uniform_(conv.weight, a=0.01, nonlinearity='leaky_relu')
            bn = nn.BatchNorm2d(out_channels)
            return [conv, relu, bn]
        
        self.conv_layers += build_conv_layer(1, 8, 5, 1, 1)
        self.conv_layers += build_conv_layer(8, 16, 3, 1, 1)
        self.conv_layers += build_conv_layer(16, 32, 3, 1, 1)
        self.conv_layers += build_conv_layer(32, 64, 3, 1, 1)

        self.conv = nn.Sequential(*self.conv_layers)        
        self.pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=2)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        return self.drop(x)
