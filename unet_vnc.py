import torch
import torch.nn as nn
import torchvision



class  Block(nn.Module):
    """A representation for the basic convolutional building block of the unet
    Parameters
    ----------
    in_ch : int  number of input channels to the block
    out_ch : int number of output channels of the block
    """
    def __init__(self, in_ch, out_ch): #initialize conv layers
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x): # use conv layers in the forwad pass
        """Performs the forward pass of the block.
        Parameters
        ----------
        x : torch.Tensor
            the input to the block
        Returns
        -------
        x : torch.Tensor
            the output of the forward pass
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x



# ENCODER: 
class Encoder(nn.Module):

    """
    A representation for the encoder part of the unet.
    Similar to a convolutional network. 
    Develop a global represenation of the image as a series of contractions of 
    larger and larger neighborhoods 
    """

    def __init__(self, chs=(1,64,128,256,512,1024)):
        super().__init__()

        # convolutional blocks--> increase the number of features channels
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        
        # max pooling --> resolution reduction
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = [] # store outputs, # save features to concatenate to decoder blocks
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        ftrs.append(x) 
        return ftrs





# DECODER
class Decoder(nn.Module):
    """
    aims to distirbute and mix the info that comes from the global features and the local details given by skip Connections
# at each block the resolution is increased (upsampling) and the number of channels is halved
    """
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        # up-convolution is done with ConvTranspose2D in PyTorch
        self.upconvs    = nn.ModuleList( # Transpose conv layers --> increase resol
            [nn.ConvTranspose2d(chs[i], chs[i], 2, 2) for i in range(len(chs)-1)]
            )
        # list of decoder blocks, standard conv layers --> reduce number of features channels
        self.dec_blocks = nn.ModuleList(
            [Block(2*chs[i], chs[i+1]) for i in range(len(chs)-1)]) # Multiplied by 2, bc number of channels from previous output but also concatenate wih encoder
             
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x) #  transposed convolution
            enc_ftrs = encoder_features[i]
            x        = torch.cat([x, enc_ftrs], dim=1) # concatenate these features to x
            x        = self.dec_blocks[i](x) # convolution block
        return x


# x = torch.randn(1, 1, 128, 128)
# encoder = Encoder()
# decoder = Decoder()
# encoder_ftrs = encoder(x)
# # print(encoder_ftrs)
# encoder_features_reversed = encoder_ftrs[::-1]


# output = decoder(encoder_features_reversed[0], encoder_features_reversed[1:])
# print(output.shape)



class UNet(nn.Module):
    """
    A representation for a unet
    Parameters
    ----------
    enc_chs : tuple holds the number of input channels of each block in the encoder
    dec_chs : tuple holds the number of input channels of each block in the decoder
    num_classes : int number of output classes of the segmentation
    """
    
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
        num_classes=1,
    ):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Sequential( # output layer, # 1 deconv to compress 32 channels into one channel final output
            nn.Conv2d(dec_chs[-1], num_classes, 1),) 

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        # Reversing order of encoder: to go from highest level of abstraction to low level
        # enc_ftrs[::-1][0]: level with the most abstract and global info
        # enc_ftrs[::-1][1:] select all elements of encoder list, except first one
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        return out


# enc_chs=(1, 64, 128, 256)
# dec_chs=(256, 128, 64, 32)

# unet = UNet(enc_chs, dec_chs)
# unet = UNet()
# x    = torch.randn(32, 1, 64, 64)
# output = unet(x)
# print(output.shape)
# print(unet(x).shape)


# x = torch.randn(1, 1, 128, 128)
# encoder = Encoder()
# decoder = Decoder()
# encoder_ftrs = encoder(x)
# # print(encoder_ftrs)
# encoder_features_reversed = encoder_ftrs[::-1]


# output = decoder(encoder_features_reversed[0], encoder_features_reversed[1:])
# print(output.shape)