import torch.nn as nn
from .memory_module import MemModule

class AutoEncoderCov2DMem(nn.Module):
    def __init__(self, opt):
        super(AutoEncoderCov2DMem, self).__init__()
        print('AutoEncoderCov2DMem')
        self.image_channel_size = opt.img_chn_size
        self.conv_channel_size  = opt.conv_chn_size
        self.mem_dim = opt.memory_dim
        self.shrink_thres = opt.shrink_threshold
        self.device  = opt.device

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.image_channel_size, out_channels=self.conv_channel_size, kernel_size=1, stride=2,),
            nn.BatchNorm2d(num_features=self.conv_channel_size,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.conv_channel_size, out_channels=self.conv_channel_size*2, kernel_size=3, stride=2,),
            nn.BatchNorm2d(num_features=self.conv_channel_size*2,),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.conv_channel_size*2, out_channels=self.conv_channel_size*4, kernel_size=3, stride=2,),
            nn.BatchNorm2d(num_features=self.conv_channel_size*4,),
            nn.ReLU(inplace=True)
            )   # (B, 64, 2, 2)

        self.mem_rep = MemModule(mem_dim=self.mem_dim, fea_dim=self.conv_channel_size*4, shrink_thres=self.shrink_thres, device=self.device)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.conv_channel_size*4, out_channels=self.conv_channel_size*2, kernel_size=3, stride=2, output_padding=1,),
            nn.BatchNorm2d(num_features=self.conv_channel_size*2,),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.conv_channel_size*2, out_channels=self.conv_channel_size, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=self.conv_channel_size,),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.conv_channel_size, out_channels=self.image_channel_size, kernel_size=3, stride=2, output_padding=1,),
            )   # (B, 1, 28, 28)
    
    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att}