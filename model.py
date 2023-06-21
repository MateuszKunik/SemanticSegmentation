import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            # First block of convolutional layers
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second block of convolutional layers
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        # Initializing a module list to store neural networks.
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Creating a down part of UNet, also called "encoder".
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Initializing a bottleneck.
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Creating a up part of UNet, also called "decoder".
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Initializing a final convolution layer for specific output.
        self.final_conv = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        """
        Pytorch forward function for a subclass of nn.Module.
        """

        skip_connections = []

        # Passing input through encoder.
        for down in self.downs:
            x = down(x)

            # Saving outputs layers for skip connection.
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # Reversing the order.
        skip_connections = skip_connections[::-1]

        # Passing input with skip connection through decoder.
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # Resizing if necessary.
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection[2:])

            # Concatenating skip connection with actual layers.
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)