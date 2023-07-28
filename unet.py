# Code adapted from https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py

import torch
import torch.nn as nn


""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.dropout1 = nn.Dropout2d(0.2)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.dropout2 = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.dropout1(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
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


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 256)
        self.e2 = encoder_block(256, 256)
        self.e3 = encoder_block(256, 512)
        # self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 256)
        # self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(256, 1, kernel_size=1, padding=0)
        # self.outbn = nn.BatchNorm2d(1)
        # self.norm = nn.ReLU()
        self.norm = nn.ReLU()

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        # s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p3)

        """ Decoder """
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        # d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d3)
        # outputs = self.outbn(outputs)
        outputs = self.norm(outputs)

        return outputs/outputs.max()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sim_ARPES import generate_batch
    # inputs = torch.randn((2, 32, 256, 256))
    # e = encoder_block(32, 64)
    # x, p = e(inputs)
    # print(x.shape, p.shape)
    #
    # d = decoder_block(64, 32)
    # y = d(p, x)
    # print(y.shape)

    # inputs = torch.randn((2, 1, 64, 64))
    inputs, targets = generate_batch(2)
    print(inputs.shape)
    inputs = torch.unsqueeze(torch.tensor(inputs), 1)

    model = UNet()
    y = model(inputs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(inputs[0][0], cmap='gray_r', origin='lower')
    ax1.set_title(f'Input min: {inputs[0][0].min():.2f}, max: {inputs[0][0].max():.2f}')
    ax2.imshow(y[0][0].detach().numpy(), cmap='gray_r', origin='lower')
    ax2.set_title(f'Inferred min: {y[0][0].min():.2f}, max: {y[0][0].max():.2f}')
    plt.show()
    print(y.shape)