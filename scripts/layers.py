import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pe_layers import PE3_Layer_matrix

logger = logging.getLogger(__name__)

class ManhattanSelfAttention(nn.Module):
    def __init__(self, d_model, gamma=0.9, decomposed=True):
        super(ManhattanSelfAttention, self).__init__()
        self.d_model = d_model
        self.gamma = gamma
        self.decomposed = decomposed
        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)

    def _compute_spatial_decay_2d(self, height, width, device):
        """Full 2D Manhattan distance decay matrix (non-decomposed)."""
        seq_len = height * width
        indices = torch.arange(seq_len, device=device)
        coords = torch.stack([indices // width, indices % width], dim=1)  # [seq_len, 2]
        manhattan_dist = torch.cdist(coords.float(), coords.float(), p=1)  # [seq_len, seq_len]
        spatial_decay = self.gamma ** manhattan_dist
        logger.debug(f"Full 2D spatial decay shape: {spatial_decay.shape}")
        return spatial_decay

    def _compute_spatial_decay_1d(self, size, device):
        """1D decay matrix for decomposed MaSA."""
        indices = torch.arange(size, device=device)
        dist = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))  # [size, size]
        spatial_decay = self.gamma ** dist
        return spatial_decay

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        seq_len = height * width
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size, seq_len, channels)

        query = self.query_dense(x_flat)
        key = self.key_dense(x_flat)
        value = self.value_dense(x_flat)

        if self.decomposed:
            # Decomposed MaSA: Compute attention along H and W separately
            # Reshape for height-wise attention
            x_h = x.permute(0, 2, 1, 3).reshape(batch_size * height, width, channels)
            q_h = self.query_dense(x_h)
            k_h = self.key_dense(x_h)
            v_h = self.value_dense(x_h)
            attn_h_scores = torch.bmm(q_h, k_h.transpose(1, 2)) / (self.d_model ** 0.5)
            d_h = self._compute_spatial_decay_1d(width, x.device)
            attn_h = F.softmax(attn_h_scores, dim=-1) * d_h.unsqueeze(0)
            out_h = torch.bmm(attn_h, v_h).view(batch_size, height, width, channels)

            # Width-wise attention
            x_w = out_h.permute(0, 2, 1, 3).reshape(batch_size * width, height, channels)
            q_w = self.query_dense(x_w)
            k_w = self.key_dense(x_w)
            v_w = self.value_dense(x_w)
            attn_w_scores = torch.bmm(q_w, k_w.transpose(1, 2)) / (self.d_model ** 0.5)
            d_w = self._compute_spatial_decay_1d(height, x.device)
            attn_w = F.softmax(attn_w_scores, dim=-1) * d_w.unsqueeze(0)
            output = torch.bmm(attn_w, v_w).view(batch_size, width, height, channels)

            output = output.permute(0, 3, 2, 1)  # [batch, channels, height, width]
        else:
            # Full MaSA
            attn_scores = torch.bmm(query, key.transpose(1, 2)) / (self.d_model ** 0.5)
            spatial_decay = self._compute_spatial_decay_2d(height, width, x.device)
            attn_weights = F.softmax(attn_scores, dim=-1) * spatial_decay.unsqueeze(0)
            output = torch.bmm(attn_weights, value)
            output = output.view(batch_size, height, width, channels).permute(0, 3, 1, 2)

        logger.debug(f"MaSA output shape: {output.shape}")
        return output

class DoubleConvMaSA(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, batch_norm=True, use_masa=False, gamma=0.9):
        super(DoubleConvMaSA, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
            layers.insert(4, nn.BatchNorm2d(out_channels))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        if use_masa:
            # Use decomposed MaSA for efficiency in early layers
            layers.append(ManhattanSelfAttention(out_channels, gamma=gamma, decomposed=True))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class NexBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rate=2, use_masa=True, gamma=0.9):
        super(NexBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size // 2) * dilation_rate,
                dilation=dilation_rate
            ),
            nn.ReLU(inplace=True)
        )
        # After conv, the number of channels is out_channels, so PE3_Layer_matrix should expect out_channels
        self.pe = PE3_Layer_matrix(input_channels=out_channels, units=out_channels)
        self.double_conv_masa = DoubleConvMaSA(out_channels, out_channels, use_masa=use_masa, gamma=gamma)

    def forward(self, x):
        x = self.conv(x)
        x = self.pe(x)
        x_att = self.double_conv_masa(x)
        return x, x_att
