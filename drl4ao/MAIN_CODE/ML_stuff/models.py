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
            nn.Conv2d(128, 128,           kernel_size=3, stride=1, padding=1),  # Downsample by 2x
            nn.LeakyReLU(),
            nn.Conv2d(128, 256,           kernel_size=3, stride=1, padding=1),  # Downsample by 2x
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


############ UNET ############

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)         
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)         
        self.relu = nn.LeakyReLU()     
        
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))     
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)     
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

    

class build_unet(nn.Module):
    def __init__(self, xvalid, yvalid):
        super().__init__()
        """ Valid Pixels """
        self.xvalid = xvalid
        self.yvalid = yvalid

        """ Encoder """
        self.e1 = encoder_block(4, 64) # Image size to 12x12
        self.e2 = encoder_block(64, 128) # Image size to 6x6
        self.e3 = encoder_block(128, 256) # Image size to 3x3
        # self.e4 = encoder_block(256, 512)         
        """ Bottleneck """
        self.b = conv_block(256, 512) # Image size stays 3x3     
        """ Decoder """
        self.d1 = decoder_block(512, 256) # Image size to 6x6
        self.d2 = decoder_block(256, 128) # Image size to 12x12
        self.d3 = decoder_block(128, 64) # Image size to 24x24
        # self.d4 = decoder_block(128, 64)         
        # """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=3, padding=1)     
        
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        # s4, p4 = self.e4(p3)         
        """ Bottleneck """
        b = self.b(p3)         
        """ Decoder """
        # d1 = self.d1(b, s4)
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)         
        # """ Classifier """
        outputs = self.outputs(d3)

        cropped_outputs = outputs[:,:,1:22,1:22]

        x_out = torch.zeros_like(cropped_outputs)
        x_out[:,:,self.xvalid, self.yvalid] = cropped_outputs[:,:,self.xvalid, self.yvalid]

        return x_out


class Unet_big(nn.Module):
    def __init__(self, xvalid, yvalid):
        super().__init__()
        """ Valid Pixels """
        self.xvalid = xvalid
        self.yvalid = yvalid

        """ Encoder """
        self.e1 = encoder_block(4, 128) # Image size to 12x12
        self.e2 = encoder_block(128, 128) # Image size to 6x6
        self.e3 = encoder_block(128, 256) # Image size to 3x3
        # self.e4 = encoder_block(256, 512)         
        """ Bottleneck """
        self.b = conv_block(256, 512) # Image size stays 3x3     
        """ Decoder """
        self.d1 = decoder_block(512, 256) # Image size to 6x6
        self.d2 = decoder_block(256, 128) # Image size to 12x12
        self.d3 = decoder_block(128, 128) # Image size to 24x24
        # self.d4 = decoder_block(128, 64)         
        # """ Classifier """
        self.outputs = nn.Conv2d(128, 1, kernel_size=3, padding=1)     
        
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        # s4, p4 = self.e4(p3)         
        """ Bottleneck """
        b = self.b(p3)         
        """ Decoder """
        # d1 = self.d1(b, s4)
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)         
        # """ Classifier """
        outputs = self.outputs(d3)

        cropped_outputs = outputs[:,:,1:22,1:22]

        x_out = torch.zeros_like(cropped_outputs)
        x_out[:,:,self.xvalid, self.yvalid] = cropped_outputs[:,:,self.xvalid, self.yvalid]

        return x_out
    

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)