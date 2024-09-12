import torch
import torch.nn as nn


class Reconstructor(nn.Module):
    def __init__(self, input_channels, output_channels, output_size, xvalid, yvalid):
        super().__init__()

        self.xvalid = xvalid
        self.yvalid = yvalid

        self.downsampler = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.LeakyReLU(),
            nn.Conv2d(64, 64,             kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.LeakyReLU(),
            nn.Conv2d(64, 128,            kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.LeakyReLU(),
            nn.Conv2d(128, 128,           kernel_size=3, stride=2, padding=1),  # Downsample by 2x
            nn.LeakyReLU(),
            # nn.Conv2d(128, 128,           kernel_size=3, stride=1, padding=1),  # No downsampling
            # nn.LeakyReLU(),
            # nn.Conv2d(128, 128,           kernel_size=3, stride=1, padding=1),  # No downsampling
            # nn.LeakyReLU(),
            nn.Conv2d(128, 128,           kernel_size=3, stride=2, padding=1),  # Downsample by 2x
            nn.LeakyReLU(),
            nn.Conv2d(128, 256,           kernel_size=3, stride=2, padding=1),  # Downsample by 2x
            nn.LeakyReLU(),
            # nn.Conv2d(256, 256,           kernel_size=3, stride=2, padding=1),  # No downsampling
            # nn.LeakyReLU(),
            # Additional layers can be added if more downsampling is needed
        )

        # Final layer to adjust the output to exactly m x m
        self.final_conv = nn.Conv2d(256, output_channels, kernel_size=3, padding=1)
        self.final_pool = nn.AdaptiveAvgPool2d(output_size)  # Output size mxm

    def forward(self, x):
        x = self.downsampler(x)
        x = self.final_conv(x)
        x = self.final_pool(x)

        # Mask out the invalid region
        x_out = torch.zeros_like(x)
        x_out[:,:,self.xvalid, self.yvalid] = x[:,:,self.xvalid, self.yvalid]

        return x_out


class Reconstructor_2(nn.Module):
    def __init__(self, input_channels, output_channels, output_size, xvalid, yvalid):
        super().__init__()

        self.xvalid = xvalid
        self.yvalid = yvalid

        self.downsampler = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.LeakyReLU(),
            nn.Conv2d(64, 64,             kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.LeakyReLU(),
            nn.Conv2d(64, 128,            kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.LeakyReLU(),
            nn.Conv2d(128, 128,           kernel_size=3, stride=2, padding=1),  # Downsample by 2x
            nn.LeakyReLU(),
            nn.Conv2d(128, 128,           kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.LeakyReLU(),
            nn.Conv2d(128, 128,           kernel_size=3, stride=1, padding=1),  # No downsampling
            nn.LeakyReLU(),
            nn.Conv2d(128, 128,           kernel_size=3, stride=2, padding=1),  # Downsample by 2x
            nn.LeakyReLU(),
            nn.Conv2d(128, 256,           kernel_size=3, stride=2, padding=1),  # Downsample by 2x
            nn.LeakyReLU(),
            nn.Conv2d(256, 256,           kernel_size=3, stride=2, padding=1),  # No downsampling
            nn.LeakyReLU(),
            # Additional layers can be added if more downsampling is needed
        )

        # Final layer to adjust the output to exactly m x m
        self.final_conv = nn.Conv2d(256, output_channels, kernel_size=3, padding=1)
        self.final_pool = nn.AdaptiveAvgPool2d(output_size)  # Output size mxm

    def forward(self, x):
        x = self.downsampler(x)
        x = self.final_conv(x)
        x = self.final_pool(x)

        # Mask out the invalid region
        x_out = torch.zeros_like(x)
        x_out[:,:,self.xvalid, self.yvalid] = x[:,:,self.xvalid, self.yvalid]

        return x_out