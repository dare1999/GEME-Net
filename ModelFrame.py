import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from xpos_relative_position import XPOS
from MAGCN_e import STMultiAdaptiveGCN
from CMSA import CMSAModel
import config


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1, map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(
            in_channels=self.num_2d_kernels,
            out_channels=self.num_2d_kernels,
            kernel_size=map_k,
            stride=1,
            padding=map_k // 2,
            groups=G,
            bias=False
        )
        self.bias = None
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2 if padding is None else padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class RepConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(RepConvBlock, self).__init__()
        self.conv = RepConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=None, groups=1, map_k=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Hswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Conv1dBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride=1, dilation=1, dropout=0.2):
        super(Conv1dBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        self.chomp = Chomp1d(self.padding)
        self.bn = nn.BatchNorm1d(n_outputs)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.chomp, self.bn, self.relu, self.dropout)

    def forward(self, x):
        return self.net(x)


class Chomp2d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size[0], :-self.chomp_size[1]].contiguous()


class CausalConv2D(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation):
        super().__init__()
        self.padding = (
            (kernel_size[0] - 1) * dilation[0],
            (kernel_size[1] - 1) * dilation[1],
        )
        self.conv = nn.Conv2d(n_inputs, n_outputs, kernel_size=kernel_size, stride=stride, padding=self.padding, dilation=dilation)
        self.chomp = Chomp2d(self.padding)

    def forward(self, x):
        return self.chomp(self.conv(x))


class CausalConv1DBlock(nn.Module):

    def __init__(self, n_inputs1, n_inputs2, n_outputs, kernel_size, stride, dilation, dropout=None):
        super(CausalConv1DBlock, self).__init__()
        self.causalconv1d1 = Conv1dBlock(n_inputs1, 128, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.causalconv1d2 = Conv1dBlock(n_inputs2, 32, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.causalconv1d3 = Conv1dBlock(128 + 32, n_outputs, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, time, x):
        out_1 = self.causalconv1d1(x)  # [B, 128, T]
        B_time, C_time, T, D_time = time.shape
        time_for_conv1d = time.reshape(B_time, C_time * D_time, T)  # [B, D_time, T]
        out_2 = self.causalconv1d2(time_for_conv1d)  # [B, 32, T]
        fused_input = torch.cat([out_1, out_2], dim=1)  # [B, 128+32, T]
        out = self.causalconv1d3(fused_input)  # [B, n_outputs, T]
        return out

class EventAwareMOEDecomp(nn.Module):

    def __init__(self, input_dim, num_experts=5, event_dim=8):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.AvgPool2d(kernel_size=(k, 1), stride=1, padding=(k // 2, 0))
            for k in [3, 6, 12, 24, 48]
        ])

        self.event_encoder = nn.Sequential(
            nn.Linear(event_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts)
        )

        # 特征融合层
        self.fusion = nn.Conv2d(input_dim * 2, input_dim, kernel_size=1)

    def forward(self, x, event_feat):
        batch, C, T, V = x.shape
        trend_components = [expert(x)[:, :, :T, :] for expert in self.experts]  # 裁剪时间维度

        event_weights = self.event_encoder(event_feat)  # => [B, 5]
        event_weights = torch.sigmoid(event_weights)
        event_weights = F.softmax(event_weights, dim=-1)  # => [B,5]
        event_weights = event_weights.unsqueeze(-1).unsqueeze(-1)

        combined_trend = sum(
            w * comp for w, comp in zip(event_weights.unbind(dim=2), trend_components)
        )
        return self.fusion(torch.cat([x, combined_trend], dim=1)) + x


class EventDrivenFEA(nn.Module):
    def __init__(self, embed_dim, num_heads, event_dim=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.event_proj = nn.Sequential(
            nn.Linear(event_dim, num_heads),
            nn.Sigmoid()
        )
        self.mode_selector = nn.Linear(1 + num_heads, embed_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_weights = None

    def forward(self, x, event_feat):
        B, T, V, C = x.shape
        x_flatten = x.view(B, T * V, C)

        x_freq = torch.fft.rfft(x_flatten, dim=1, norm='ortho')
        freq_mags = torch.abs(x_freq)

        mean_freq = freq_mags.mean(dim=2).unsqueeze(-1)  # [B, seq_len, 1]

        event_emb = self.event_proj(event_feat).unsqueeze(1)  # [B,1,1,num_heads]
        event_emb = event_emb.squeeze(2)
        expanded_event_emb = event_emb.expand(-1, (T * V // 2 + 1), -1)  # [B, seq_len, num_heads]

        selector_input = torch.cat([mean_freq, expanded_event_emb], dim=-1)
        selected_modes = torch.sigmoid(self.mode_selector(selector_input))  # [B, seq_len, C]

        x_freq = x_freq.real * selected_modes
        x_filtered = torch.fft.irfft(x_freq, n=T * V, dim=1, norm='ortho')

        qkv = self.qkv(x_filtered).reshape(B, T * V, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        self.attn_weights = attn#.detach().cpu()

        attn = F.softmax(attn, dim=-1)
        self.attn_weights = attn @ v#.detach().cpu()
        out = (attn @ v).transpose(1, 2).reshape(B, T * V, self.embed_dim)
        out = self.out_proj(out).view(B, T, V, self.embed_dim)

        return out


class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        super(SimpleRetention, self).__init__()
        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size
        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma
        self.xpos = XPOS(head_size)

    def forward(self, Q, K, V):
        sequence_length = Q.shape[1]
        D = self._get_D(sequence_length).to(Q.device)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        ret = (Q @ K.transpose(-1, -2)) * D.unsqueeze(0)
        return ret @ V

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)
        D = (self.gamma ** (n - m)) * (n >= m).float()
        D[D != D] = 0
        return D


class MultiScaleRetention(nn.Module):

    def __init__(self, hidden_size, heads, double_v_dim=False):
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = self.head_size * 2 if double_v_dim else self.head_size
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach()

        self.retentions = nn.ModuleList([
            SimpleRetention(self.head_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])

    def forward(self, Q, K, V):
        Y = []
        for i in range(self.heads):
            Q_i = Q[:, :, i * self.head_size:(i + 1) * self.head_size]
            K_i = K[:, :, i * self.head_size:(i + 1) * self.head_size]
            V_i = V[:, :, i * self.head_v_dim:(i + 1) * self.head_v_dim]

            Y.append(self.retentions[i](Q_i, K_i, V_i))

        return torch.cat(Y, dim=-1)

class Encoder(nn.Module):
    def __init__(self, num_features, embedding_dim, time_steps, d_model, attention_embed_dim, kernel_size=3):
        super(Encoder, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.time_steps = time_steps
        self.attention_embed_dim = attention_embed_dim

        self.encoder = nn.Sequential(
            Conv1dBlock(num_features * embedding_dim, num_features * embedding_dim // 2, kernel_size),
            Conv1dBlock(num_features * embedding_dim // 2, d_model, kernel_size)
        )
        self.layer = nn.Linear(d_model * time_steps, attention_embed_dim * time_steps)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.num_features * self.embedding_dim, self.time_steps)
        encoder_out = self.encoder(x)
        encoder_out = encoder_out.view(batch_size, -1)  # [B, d_model*time_steps]
        layer_out = self.layer(encoder_out)  # [B, attention_embed_dim * time_steps]

        out = layer_out.view(batch_size, self.time_steps, self.attention_embed_dim)  # 调整为[B, T, attention_embed_dim]
        return out

class DynamicImpactEncoder(nn.Module):

    def __init__(self, event_types=3, time_decay=[300, 600, 180], alpha=[0.3, 0.5, 0.2]):
        super().__init__()
        self.register_buffer('time_decay', torch.tensor(time_decay))
        self.register_buffer('alpha', torch.tensor(alpha).view(1, -1))

    def forward(self, events):
        B, T, _ = events.size()
        impact = torch.zeros_like(events)
        for t in range(T):
            time_diff = torch.arange(T - t, dtype=torch.float32, device=events.device)
            decay = torch.exp(-time_diff.view(1, -1) / self.time_decay.view(-1, 1))
            decay = decay.permute(1, 0).unsqueeze(0)  # [1, T-t, 3]
            weighted_events = (events[:, t:, :] * decay)  # [B, T-t, 3]
            summed = weighted_events.sum(dim=1)          # [B, 3]
            impact[:, t, :] = summed * self.alpha         # [B,3] * [1,3] → [B,3]
        return impact


class CVCalculator(nn.Module):

    def __init__(self, window_size=60):
        super().__init__()
        self.window = window_size

    def forward(self, flow):
        # flow: [B, T, V]
        rolling_mean = flow.unfold(1, self.window, 1).mean(dim=-1)
        rolling_std = flow.unfold(1, self.window, 1).std(dim=-1)
        cv = rolling_std / (rolling_mean + 1e-6)  # [B, T-win+1, V]
        return F.pad(cv, (0, 0, self.window - 1, 0))


class GrangerCausalModule(nn.Module):
    def __init__(self, max_lag=6, thresh=0.1):
        super().__init__()
        self.max_lag = max_lag
        self.thresh = thresh

    def forward(self, flow, events):
        B, T, V = flow.size()
        E = events.size(2)
        K = self.max_lag
        masks = []
        for v in range(V):
            Xs = []
            for d in range(1, K + 1):
                Xs.append(flow[:, K - d:T - d, v:v + 1])  # [B, T-K, 1]
            for d in range(1, K + 1):
                Xs.append(events[:, K - d:T - d, :])  # [B, T-K, E]

            X = torch.cat(Xs, dim=-1)  # [B, T-K, F], F = K + K*E + K
            y = flow[:, K:, v]  # [B, T-K]

            sol = torch.linalg.lstsq(X, y).solution  # [B, F]

            betas = sol[:, K: K + K * E]  # [B, K*E]
            betas = betas.view(B, K, E)  # [B, K, E]
            beta_event, _ = betas.abs().max(dim=1)  # [B, E]

            mask_v = (beta_event > self.thresh).float()  # [B, E]
            masks.append(mask_v)

        return torch.stack(masks, dim=1)


class EventDrivenAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, event_dim=3):

        super().__init__()
        self.embed_dim = embed_dim            # C_head
        self.num_heads = num_heads            # H
        self.head_dim = embed_dim // num_heads

        self.event_proj = nn.Linear(event_dim, num_heads)

        self.qkv      = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_weights = None

    def forward(self, x, dynamic_impact, granger_mask):
        B, T, V, C = x.shape
        TV = T * V

        x_flat = x.reshape(B, TV, C)                         # [B, TV, C_head]

        qkv = (
            self.qkv(x_flat)
            .view(B, TV, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        ew = torch.sigmoid(self.event_proj(dynamic_impact))  # [B, T, H]
        ew = ew.permute(0, 2, 1).unsqueeze(-1)               # [B, H, T, 1]

        ew_full = ew.repeat(1, 1, 1, V).reshape(B, self.num_heads, TV, 1)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        scores = scores * ew_full + (1 - ew_full) * (-1e9)

        var_mask = (granger_mask.sum(-1) > 0).float()  # [B, V]
        mask_q = var_mask.repeat_interleave(T, dim=1)  # [B, TV]
        mask_q = mask_q.view(B, 1, TV, 1)  # [B,1,TV,1]
        scores = scores.masked_fill(mask_q == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out  = (attn @ v)
        self.attn_weights = attn#.detach().cpu()
        out  = out.transpose(1, 2).reshape(B, TV, C)         # [B, TV, C_head]

        out = self.out_proj(out)                             # [B, TV, C_head]

        return out.view(B, T, V, C)                          # [B, T, V, C_head]


class SparseFlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity=0.5, block_size=32, causal=False):
        super(SparseFlashAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparsity = sparsity
        self.block_size = block_size
        self.causal = causal

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = x.to(x.device)
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        sparsity_mask = torch.rand(batch_size, self.num_heads, seq_len, device=x.device) > self.sparsity

        q = q * sparsity_mask.unsqueeze(-1)
        k = k * sparsity_mask.unsqueeze(-1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_weights = None

    def forward(self, query, key, value):
        """
        query, key, value: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = query.size()

        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        self.attn_weights = attn_weights#.detach().cpu()

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.leaky_relu(out)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))  # F.relu(self.fc1(x))
        x = self.fc2(x)  # F.sigmoid(self.fc2(x))
        return x


class Model(nn.Module):
    def __init__(self, adj, in_channels, lable_dim, time_len, device, config, kernel_size=3, stride=1, dilation=1):
        super().__init__()

        self.config = config
        self.in_channels = in_channels
        self.config.in_channels = in_channels
        self.adj = adj  # 邻接矩阵
        self.lable_dim = lable_dim
        self.device = device
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.time_len = time_len

        self.time_embed = nn.Embedding(24, 8)

        self.repconvblock = RepConvBlock(
            in_channels=self.in_channels,
            out_channels=config.repconv_out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

        self.causal_conv_q = CausalConv2D(config.repconv_out_channels, config.repconv_out_channels,
                                          (self.kernel_size, self.kernel_size), self.stride,
                                          (self.dilation, self.dilation))
        self.causal_conv_k = CausalConv2D(config.repconv_out_channels, config.repconv_out_channels,
                                          (self.kernel_size, self.kernel_size), self.stride,
                                          (self.dilation, self.dilation))

        self.moe_decomp = EventAwareMOEDecomp(input_dim=self.in_channels, event_dim=config.event_emb_dim)

        self.event_feat_proj = nn.Linear(config.event_dim, config.event_emb_dim)

        self.event_driven_feat = EventDrivenFEA(
            embed_dim=config.freq_attn_channels,
            num_heads=config.num_heads,
            event_dim=config.event_emb_dim
        )

        self.retnet = MultiScaleRetention(
            hidden_size=config.retnet_hidden_dim,
            heads=config.retnet_heads
        )

        self.magcn = STMultiAdaptiveGCN(
            in_channels=config.repconv_out_channels,
            out_channels=config.magcn_out_channels,
            A=self.adj,
            time_emb_dim=8
        )

        self.cmsanet = CMSAModel(
            img_feat_dim=config.magcn_out_channels,
            lang_feat_dim=config.cmsa_lang_feat_dim,
            spatial_dim=config.spatial_dim,
            cmsa_dim=config.cmsa_dim,
            fused_dim=config.fused_dim,
            output_dim=1
        )

        self.encoder = Encoder(
            config.fused_dim,
            config.encoder_embed_dim,
            self.time_len,
            config.encoder_d_model,
            config.attention_embed_dim,
            self.kernel_size
        )

        self.dsattention = SparseFlashAttention(
            embed_dim=config.attention_embed_dim,
            num_heads=config.attention_heads,
        )
        self.mhattention = MultiHeadAttention(
            embed_dim=config.attention_embed_dim,
            num_heads=config.attention_heads
        )

        self.timetable_refconv = CausalConv1DBlock(
            n_inputs1=config.refconv_input_dim,
            n_inputs2=config.event_dim,
            n_outputs=config.attention_embed_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation
        )

        self.resnet = ResidualBlock(channels=config.resnet_in_channels)

        self.decoder = Decoder(
            input_dim=(config.attention_embed_dim // 4) * self.time_len * 8,
            output_dim=self.lable_dim
        )

        self.fusion_conv = nn.Conv2d(
            in_channels=1 + config.freq_attn_channels,
            out_channels=config.fusion_out_channels,
            kernel_size=1,
            stride=1
        )

        self.retnet_proj = nn.Linear(config.fusion_out_channels, config.retnet_hidden_dim)

        self.dynamic_impact_encoder = DynamicImpactEncoder()
        self.cv_calculator = CVCalculator(window_size=self.time_len)
        self.granger_causal = GrangerCausalModule()

        C_head = config.attention_embed_dim // config.area_dim
        self.event_driven_attention = EventDrivenAttention(
            embed_dim=C_head,
            num_heads=config.attention_heads,
            event_dim=3
        )

    def forward(self, x, time):
        batchsize, _, time_dim, area_dim = x.shape

        hour = time[:, 0].long() % 24
        time_emb = self.time_embed(hour)  # [B, 8]

        rep_out = self.repconvblock(x)

        event_feat = self.event_feat_proj(time.mean(2))  # [B, 3] -> [B, 8]

        decomp_out = self.moe_decomp(x, event_feat)  # [B, 1, T, V]

        temporal_feat = decomp_out.permute(0, 2, 3, 1)
        freq_attn_out = self.event_driven_feat(temporal_feat, event_feat)  # [B, T, V, C]
        freq_attn_out = freq_attn_out.permute(0, 3, 1, 2)  # [B, C, T, V]

        fused_retnet_input = torch.cat([decomp_out, freq_attn_out], dim=1)
        fused_retnet_input = self.fusion_conv(fused_retnet_input)  # [B, fusion_out, T, V]

        B, C, T, V = fused_retnet_input.shape
        retnet_qkv = fused_retnet_input.permute(0, 2, 3, 1).reshape(B, T * V, C)
        retnet_qkv = self.retnet_proj(retnet_qkv)
        retnet_out = self.retnet(retnet_qkv, retnet_qkv, retnet_qkv)

        retnet_out = retnet_out.view(batchsize, time_dim, area_dim, -1).permute(0, 3, 1, 2)

        time_emb = time_emb.mean(dim=2)
        magcn_out = self.magcn(rep_out, time_emb=time_emb)

        cmas_out = self.cmsanet(magcn_out, retnet_out)

        encoder_out = self.encoder(cmas_out)

        events_input = time.squeeze(1)
        dynamic_impact = self.dynamic_impact_encoder(events_input)  # [B, T, E]
        flow = x.squeeze(1)
        granger_mask = self.granger_causal(flow, events_input)  # [B, V, E]

        B, T, E = encoder_out.shape
        V = self.config.area_dim
        C_head = E // V

        enc_4d = encoder_out.view(B, T, V, C_head)
        eda_out = self.event_driven_attention(
            enc_4d, dynamic_impact, granger_mask
        )
        encoder_out = eda_out.view(B, T, E)  # -> [16, 64, 396]


        dsattention_out = self.dsattention(encoder_out)
        mhattention_out = self.mhattention(dsattention_out, encoder_out, encoder_out)
        mhattention_out = mhattention_out.view(batchsize, time_dim, self.config.area_dim, -1).permute(0, 3, 1, 2)
        rep_out1 = rep_out.permute(0, 1, 3, 2).reshape(batchsize, -1, time_dim)

        fused_output = self.timetable_refconv(time, rep_out1)
        fused_output = fused_output.reshape(batchsize, -1, time_dim, self.config.area_dim)  # [B, C, T, V]

        resnet_input = torch.cat([fused_output, mhattention_out], dim=1)
        resnet_out = self.resnet(resnet_input)

        pre_y = self.decoder(resnet_out)
        return pre_y


    def Mse_Loss(self, lable, pred):
        return F.mse_loss(pred, lable)
