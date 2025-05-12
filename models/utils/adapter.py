import torch
from torch import nn

class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4, act=nn.GELU, mode='linear'):
        super().__init__()
        self.mode = mode

        if mode == 'conv': #25M
            self.proj_up = nn.Conv2d(in_channels, in_channels//ratio, kernel_size=3, stride=1, padding=1)
            self.act = act()
            self.proj_down = nn.Conv2d(in_channels//ratio, out_channels, kernel_size=3, stride=1, padding=1)
        elif mode == 'linear': #8.8M
            self.proj_up = nn.Linear(in_channels, in_channels//ratio)
            self.act = act()
            self.proj_down = nn.Linear(in_channels//ratio, out_channels)
        elif mode == 'conv_linear':
            self.proj_up = nn.Conv2d(in_channels, in_channels//ratio, kernel_size=3, stride=1, padding=1)
            self.act = act()
            self.proj_down = nn.Linear(in_channels//ratio, out_channels)
        elif mode == 'linear_conv': #9.8M
            self.proj_up = nn.Linear(in_channels, in_channels//ratio)
            self.act = act()
            self.proj_down = nn.Conv2d(in_channels//ratio, out_channels, kernel_size=3, stride=1, padding=1)


    def init_zero(self, layer):
        for param in layer.parameters():
            nn.init.zeros_(param)
    
    def init_one(self, layer):
        for param in layer.parameters():
            nn.init.ones_(param)
    
    def init_constant(self, layer, const=1e-6):
        for param in layer.parameters():
            nn.init.constant_(param, const)
    
    def init_layer(self, layer):
        nn.init.ones_(layer.weight)
        nn.init.zeros_(layer.bias)


    def forward(self, x):
        if self.mode == 'conv':
            x_adapt = self.proj_up(x)
            x_adapt = self.act(x_adapt)
            x_adapt = self.proj_down(x_adapt)
        elif self.mode == 'linear':
            x_adapt = x.permute(0,2,3,1)
            x_adapt = self.proj_up(x_adapt)
            x_adapt = self.act(x_adapt)
            x_adapt = self.proj_down(x_adapt)
            x_adapt = x_adapt.permute(0,3,1,2)
        elif self.mode == 'conv_linear':
            x_adapt = self.proj_up(x)
            x_adapt = self.act(x_adapt)
            x_adapt = x_adapt.permute(0,2,3,1)
            x_adapt = self.proj_down(x_adapt)
            x_adapt = x_adapt.permute(0,3,1,2)
        elif self.mode == 'linear_conv':
            x_adapt = x.permute(0,2,3,1)
            x_adapt = self.proj_up(x_adapt)
            x_adapt = self.act(x_adapt)
            x_adapt = x_adapt.permute(0,3,1,2)
            x_adapt = self.proj_down(x_adapt)

        return x_adapt 


class Adapter_linear(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4, act=nn.GELU, mode='conv'):
        super().__init__()
        self.mode = mode

        if mode == 'conv':
            self.proj_up = nn.Conv2d(in_channels, in_channels//ratio, kernel_size=3, stride=1, padding=1)
            self.act = act()
            self.proj_down = nn.Conv2d(in_channels//ratio, out_channels, kernel_size=3, stride=1, padding=1)
        elif mode == 'linear':
            self.proj_up = nn.Linear(in_channels, in_channels//ratio)
            self.act = act()
            self.proj_down = nn.Linear(in_channels//ratio, out_channels)
        elif mode == 'conv_linear':
            self.proj_up = nn.Conv2d(in_channels, in_channels//ratio, kernel_size=3, stride=1, padding=1)
            self.act = act()
            self.proj_down = nn.Linear(in_channels//ratio, out_channels)
        elif mode == 'linear_conv':
            self.proj_up = nn.Linear(in_channels, in_channels//ratio)
            self.act = act()
            self.proj_down = nn.Conv2d(in_channels//ratio, out_channels, kernel_size=3, stride=1, padding=1)


    def init_zero(self, layer):
        for param in layer.parameters():
            nn.init.zeros_(param)
    
    def init_one(self, layer):
        for param in layer.parameters():
            nn.init.ones_(param)
    
    def init_constant(self, layer, const=1e-6):
        for param in layer.parameters():
            nn.init.constant_(param, const)
    
    def init_layer(self, layer):
        nn.init.ones_(layer.weight)
        nn.init.zeros_(layer.bias)


    def forward(self, x):
        if self.mode == 'conv':
            x_adapt = self.proj_up(x)
            x_adapt = self.act(x_adapt)
            x_adapt = self.proj_down(x_adapt)
        elif self.mode == 'linear':
            x_adapt = x.permute(0,2,3,1)
            x_adapt = self.proj_up(x_adapt)
            x_adapt = self.act(x_adapt)
            x_adapt = self.proj_down(x_adapt)
            x_adapt = x_adapt.permute(0,3,1,2)
        elif self.mode == 'conv_linear':
            x_adapt = self.proj_up(x)
            x_adapt = self.act(x_adapt)
            x_adapt = x_adapt.permute(0,2,3,1)
            x_adapt = self.proj_down(x_adapt)
            x_adapt = x_adapt.permute(0,3,1,2)
        elif self.mode == 'linear_conv':
            x_adapt = x.permute(0,2,3,1)
            x_adapt = self.proj_up(x_adapt)
            x_adapt = self.act(x_adapt)
            x_adapt = x_adapt.permute(0,3,1,2)
            x_adapt = self.proj_down(x_adapt)

        return x_adapt 