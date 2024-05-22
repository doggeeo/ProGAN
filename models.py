import torch
from torch import nn
from torch.nn import functional as F

factors=[1,1,1,1,2,4,8,16,32]

class Wconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(Wconv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = Wconv2d(in_channels, out_channels)
        self.conv2 = Wconv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        if self.use_pn:
            return self.pn(self.leaky(self.conv2(self.pn(self.leaky(self.conv1(x))))))
        else:
            return self.leaky(self.conv2(self.leaky(self.conv1(x))))

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3): #steps
        super(Generator, self).__init__()

        # initial takes 1x1 -> 4x4
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            Wconv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = Wconv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels // factors[i])
            conv_out_c = int(in_channels // factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(Wconv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

class Discriminator(nn.Module):
    def __init__(self,in_channels,img_channels=3): #steps
        super().__init__()
        self.prog_blocks,self.rgb_layers=nn.ModuleList([]),nn.ModuleList([])
        self.leaky=nn.LeakyReLU(0.2)

        for i in range(len(factors)-1,0,-1):
            conv_in_c = int(in_channels // factors[i])
            conv_out_c = int(in_channels // factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_c,conv_out_c,use_pixelnorm=False))
            self.rgb_layers.append(Wconv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

        self.initial_rgb=Wconv2d(img_channels,in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.av_pool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            Wconv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            Wconv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(in_channels,1),
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha*out+(1-alpha)*downscaled

    def batch_std(self,x):
        batch_stat=(torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_stat], dim=1)

    def forward(self,x,alpha,steps):
        cur_steps=len(self.prog_blocks)-steps
        out=self.leaky(self.rgb_layers[cur_steps](x))
        if steps==0:
            out=self.batch_std(out)
            return self.final_block(out).view(out.shape[0],-1)
        downscaled=self.leaky(self.rgb_layers[cur_steps+1](self.av_pool(x)))
        out = self.av_pool(self.prog_blocks[cur_steps](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_steps+1,len(self.prog_blocks)):
            out = self.av_pool(self.prog_blocks[step](out))

        out = self.batch_std(out)
        return self.final_block(out).view(out.shape[0], -1)