import math
import torch
import torch.nn as nn


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class DownModule(nn.Module):
    def __init__(self):
        super(DownModule, self).__init__()
        self.down = lambda x: x

    def forward(self, x):
        return self.down(x)


class STCAttention(nn.Module):
    def __init__(self, out_channels, num_jpts):
        super(STCAttention, self).__init__()
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)
        self.conv_ta = nn.Conv1d(out_channels, 1, 7, padding=3)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)
        rr = 2
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        se = x.mean(-2)
        se1 = self.sigmoid(self.conv_sa(se))
        y = x * se1.unsqueeze(-2) + x
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        return y


class TGCModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], stride=1):
        super(TGCModule, self).__init__()
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            pad = (ks - 1) // 2
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (ks, 1), padding=(pad, 0), stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * len(kernel_sizes), len(kernel_sizes), 1),
            nn.Softmax(dim=1)
        )
        for conv in self.convs:
            conv_init(conv[0])
            bn_init(conv[1], 1)

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        fused = torch.cat(features, dim=1)
        attn_weights = self.attention(fused)
        out = sum([attn_weights[:, i].unsqueeze(1) * features[i] for i in range(len(self.convs))])
        return out


class MultiAdaptiveGCM(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3,
                 attention=True, time_emb_dim=8):
        super(MultiAdaptiveGCM, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        num_jpts = A.shape[-1]

        # 时间动态权重模块
        self.time_encoder = nn.LSTM(time_emb_dim, 64, batch_first=True)
        self.dynamic_weight = nn.Linear(64, num_subset)

        self.conv_d = nn.ModuleList()
        for i in range(num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
        self.PA = nn.Parameter(A)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        for i in range(num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        if attention:
            self.attention = STCAttention(out_channels, num_jpts)
        else:
            self.attention = lambda x: x
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(num_subset):
            conv_branch_init(self.conv_d[i], num_subset)

    def forward(self, x, time_emb=None):
        N, C, T, V = x.size()
        beta_values = torch.zeros(N, self.num_subset, device=x.device)

        batch_size = x.size(0)
        time_emb = time_emb.view(batch_size, -1, 8)


        if time_emb is not None:
            _, (h_n, _) = self.time_encoder(time_emb)
            beta_values = torch.sigmoid(self.dynamic_weight(h_n.squeeze(0)))

        y = None
        A = self.PA
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A_adapt = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # [N, V, V]
            static_A = self.PA[i].unsqueeze(0).expand(N, -1, -1)
            static_A = static_A.expand(N, -1, -1)
            if time_emb is not None:
                dynamic_weight = beta_values[:, i].view(-1, 1, 1)
                A_i = static_A + self.alpha * A_adapt + dynamic_weight * A_adapt
            else:
                A_i = static_A + self.alpha * A_adapt
            A2_x = x.view(N, C * T, V)
            matmul_result = torch.matmul(A2_x, A_i)
            z = self.conv_d[i](matmul_result.view(N, C, T, V))
            y = z + (y if y is not None else 0)
        y = self.bn(y)
        y = self.attention(y)
        y += self.down(x)
        return self.relu(y)


class STMultiAdaptiveGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 attention=True, time_emb_dim=3):
        super(STMultiAdaptiveGCN, self).__init__()
        self.magcm = MultiAdaptiveGCM(in_channels, out_channels, A, attention=attention,
                                      time_emb_dim=time_emb_dim)
        self.relu = nn.ReLU(inplace=True)
        self.tcms1 = nn.ModuleList([
            TGCModule(out_channels, out_channels * 2, [3, 5], stride),
            TGCModule(out_channels, out_channels, [5, 7], stride),
            TGCModule(out_channels, out_channels * 2, [7, 9], stride)
        ])
        self.tcms2 = nn.ModuleList([
            TGCModule(out_channels * 2, out_channels, [7, 9], stride),
            DownModule(),
            TGCModule(out_channels * 2, out_channels, [3, 5], stride)
        ])
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TGCModule(in_channels, out_channels, [1], stride)

    def forward(self, x,time_emb=None):
        x_ = self.magcm(x,time_emb=time_emb)
        y = None
        for tcm1, tcm2 in zip(self.tcms1, self.tcms2):
            y_ = tcm2(tcm1(x_))
            y = y_ + y if y is not None else y_
        return self.relu(y + self.residual(x))