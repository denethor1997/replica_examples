import torch
import torch.nn

class ConvBlock(nn.Module):
    def __init__(self, in_map, out_map, 
                 kernel=3, stride=1, activation=True):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_map, out_map, kernel, stride, (kernel)//2),
            nn.BatchNorm2d(out_map)
        )

        if activation:
            self.block.add_module('conv_block_relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.block(x)

        return out

class DeconvBlock(nn.Module):
    def __init__(self, in_map, out_map,
                 kernel=3, stride=2, padding=1):
        super(DeconvBlock, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_map, out_map, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_map)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, output_size):
        out = self.deconv(x, output_size=output_size)
        out = self.bn(out)
        out = self.relu(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_map, out_map, downsample=False):
        super(ResidualBlock, self).__init__()

        self.block2 = ConvBlock(out_map, out_map, 3, 1, False)
        
        stride = 1
        if downsample:
            stride = 2
        
        self.block1 = ConvBlock(in_map, out_map, 3, stride)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_map, out_map, 1, stride),
            nn.BatchNorm2d(out_map)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        residual = self.downsample(x)
        out = residual + out
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_map, out_map):
        super(Encoder, self).__init__()

        self.block1 = ResidualBlock(in_map, out_map, True)
        self.block2 = ResidualBlock(out_map, out_map)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)

        return out

class Decoder(nn.Module):
    def __init__(self, in_map, out_map, padding=1):
        super(Decoder, self).__init__()

        self.conv1 = ConvBlock(in_map, in_map//4, 1)
        self.deconv1 = DeconvBlock(in_map//4, in_map//4, 3, 2, padding)
        self.conv2 = ConvBlock(in_map//4, out_map, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.deconv1(out, output_size=output_size)
        out = self.conv2(out)

        return out


class LinkNet(nn.Module):
    def __init__(self, class_num=11):
       super(LinkNet, self).__init__()

       self.conv1 = ConvBlock(3, 64, 7, 2)
       self.pool = nn.MaxPool2d(3, 2, padding=1)

       self.encoder1 = Encoder(64, 64)
       self.encoder2 = Encoder(64, 128)
       self.encoder3 = Encoder(128, 256)
       self.encoder4 = Encoder(256, 512)

       self.decoder1 = Decoder(64, 64)
       self.decoder2 = Decoder(128, 64)
       self.decoder3 = Decoder(256, 128)
       self.decoder4 = Decoder(512, 256)

       self.deconv1 = DeconvBlock(64, 32)
       self.conv2 = ConvBlock(32, 32, 3)
       self.deconv2 = DeconvBlock(32, class_num, 2, 2, 0)

    def forward(self, x):
        conv_down_out = self.conv1(x)
        pool_out = self.pool(conv_down_out)

        encoder1_out = self.encoder1(pool_out)
        encoder2_out = self.encoder2(encoder1_out)
        encoder3_out = self.encoder3(encoder2_out)
        encoder4_out = self.encoder4(encoder3_out)

        decoder4_out = self.decoder4(encoder4_out, encoder3_out.size()) + encoder3_out
        decoder3_out = self.decoder3(decoder4_out, encoder2_out.size()) + encoder2_out
        decoder2_out = self.decoder2(decoder3_out, encoder1_out.size()) + encoder1_out
        decoder1_out = self.decoder1(decoder2_out, pool_out.size())

        deconv_out = self.deconv1(decoder1_out, conv_down_out.size())
        conv2_out = self.conv2(deconv_out)
        out = self.deconv2(conv2_out, x.size())

        return out

        
