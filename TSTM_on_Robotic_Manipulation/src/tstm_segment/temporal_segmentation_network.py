import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = DepthwiseSeparableConv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_state is None:
            h = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3),
                          device=x.device, dtype=x.dtype)
            c = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3),
                          device=x.device, dtype=x.dtype)
        else:
            h, c = hidden_state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SimpleCNN_ConvLSTM(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, hidden_dim=128, kernel_size=3, use_depthwise_lstm=False, force_lightweight=False):
        super(SimpleCNN_ConvLSTM, self).__init__()
        # Converged config: only support hidden_dim in {32, 256};
        # always lightweight (no BN, single conv block) and depthwise ConvLSTM.
        if hidden_dim == 32:
            enc_channels = [16, 32, 32]
            dec_channels = [32, 16]
        elif hidden_dim == 256:
            enc_channels = [64, 128, 256]
            dec_channels = [128, 64]
        else:
            raise ValueError("SimpleCNN_ConvLSTM only supports hidden_dim in {32, 256} under the converged config")
        use_double_conv = False
        use_batchnorm = False
        
        # Encoder
        encoder_layers = []
        in_ch = input_channels
        for i, out_ch in enumerate(enc_channels):
            encoder_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm2d(out_ch))
            encoder_layers.append(nn.ReLU(inplace=True))
            if use_double_conv:
                encoder_layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
                if use_batchnorm:
                    encoder_layers.append(nn.BatchNorm2d(out_ch))
                encoder_layers.append(nn.ReLU(inplace=True))
            if i < len(enc_channels) - 1:
                encoder_layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ConvLSTM: fixed to Depthwise variant
        self.conv_lstm = DepthwiseConvLSTMCell(
            input_dim=enc_channels[-1],
            hidden_dim=hidden_dim,
            kernel_size=kernel_size
        )
        self.use_depthwise_lstm = True
        
        # Decoder
        decoder_layers = []
        upsample_mode = 'nearest'
        decoder_layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False if upsample_mode=='bilinear' else None))
        decoder_layers.append(nn.Conv2d(hidden_dim, dec_channels[0], kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(dec_channels[0]))
        decoder_layers.append(nn.ReLU(inplace=True))
        decoder_layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False if upsample_mode=='bilinear' else None))
        decoder_layers.append(nn.Conv2d(dec_channels[0], dec_channels[1], kernel_size=3, padding=1))
        if use_batchnorm:
            decoder_layers.append(nn.BatchNorm2d(dec_channels[1]))
        decoder_layers.append(nn.ReLU(inplace=True))
        decoder_layers.append(nn.Conv2d(dec_channels[1], num_classes, kernel_size=1))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.hidden_dim = hidden_dim
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.use_double_conv = use_double_conv
        self.use_batchnorm = use_batchnorm
    
    def forward(self, x: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                return_all_frames: bool = True
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # If 4D input, add a time dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        # Encode
        x_flat = x.reshape(B * T, C, H, W)
        features = self.encoder(x_flat)
        _, C_feat, H_feat, W_feat = features.shape
        features = features.reshape(B, T, C_feat, H_feat, W_feat)
        # ConvLSTM over time steps
        temporal_features = []
        for t in range(T):
            feat_t = features[:, t]
            h, c = self.conv_lstm(feat_t, hidden_state)
            hidden_state = (h, c)
            temporal_features.append(h)
        # Decode
        if return_all_frames:
            temporal_features_stacked = torch.stack(temporal_features, dim=1)
            B_decode, T_decode, C_hidden, H_feat, W_feat = temporal_features_stacked.shape
            temporal_features_flat = temporal_features_stacked.reshape(B_decode * T_decode, C_hidden, H_feat, W_feat)
            outputs_flat = self.decoder(temporal_features_flat)
            output = outputs_flat.reshape(B_decode, T_decode, 1, 84, 84)
        else:
            last_feature = temporal_features[-1]
            output = self.decoder(last_feature)
        return output, (h, c)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    if pred.dim() == 5:
        pred = pred.flatten(0, 1)
        target = target.flatten(0, 1)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * dice
