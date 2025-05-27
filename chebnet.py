import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ChebConv(nn.Module):
    def __init__(self, in_c, out_c, K, bias=True, normalize=True, dropout=0.1):
        super(ChebConv, self).__init__()
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))
        init.xavier_normal_(self.weight)
        self.dropout = nn.Dropout(dropout)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        L = ChebConv.get_laplacian(graph, self.normalize)
        mul_L = self.cheb_polynomial(L).unsqueeze(1)
        result = torch.matmul(mul_L, inputs)
        result = torch.matmul(result, self.weight)
        result = torch.sum(result, dim=0) + self.bias
        return result

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)
        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            for k in range(2, self.K):
                multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                           multi_order_laplacian[k-2]
        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K):
        super(ChebNet, self).__init__()
        self.block1 = Cheb_block(in_c, hid_c, out_c, K)
        self.block2 = Cheb_block(out_c, hid_c, out_c, K)
        self.fc = nn.Linear(out_c, 1)
        self.act = nn.LeakyReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]
        flow_x = data["flow_x"].to(device)
        output_1 = self.block1(flow_x, graph_data)
        output_2 = self.block2(output_1, graph_data)
        output_2 = self.fc(output_2)
        return self.act(output_2.unsqueeze(2))

class Cheb_block(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K):
        super(Cheb_block, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)
        # self.att = SingleHeadAttention(hid_c)
        # self.att = SelfAttention(hid_c, 8)
        self.bn1 = nn.BatchNorm1d(hid_c)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.gru = GRUModel(input_dim=out_c, hidden_dim=16, num_layers=2, output_dim=in_c)
        self.trans = TemporalTransformer(d_model=hid_c, nhead=8, num_layers=2)
        # self.tcn = TCNBlock(in_channels=32, out_channels=hid_c)
        # self.res = DynamicResidual(out_c)
        self.act = nn.LeakyReLU()

    def forward(self, flow_x, graph_data):
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)
        output_1 = self.act(self.conv1(flow_x, graph_data))
        # print(output_1.shape, "output_1 shape")
        # output_1 = self.tcn(output_1)

        # output_1 = self.att(output_1)
        output_1 = self.trans(output_1)
        output_2 = self.conv2(output_1, graph_data)
        output_2 = self.gru(output_2)
        output_2 = self.act(output_2 +flow_x)  # 残差连接
        # print(output_2.shape, "output_2 shape")
        return output_2




class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TemporalTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 64)

    def forward(self, x):
        x = self.transformer(x)
        x = x.permute(1, 0, 2).permute(1, 0, 2)
        # print(x.shape, "x shape")
        return self.fc(x)




























class DynamicResidual(nn.Module):
    def __init__(self, channels):
        super(DynamicResidual, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        weight = self.sigmoid(self.weight)
        return weight * x + (1 - weight) * residual

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, spatial_data, temporal_data):
        attn_output, _ = self.cross_attn(temporal_data, spatial_data, spatial_data)
        return attn_output

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        out = self.act(self.norm(self.conv1(x)))
        out = self.act(self.norm(self.conv2(out)))
        return out



class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru_onefeature = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ensure that the hidden state is on the same device as the input tensor x
        device = x.device
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device).requires_grad_()

        output, h_n = self.gru_onefeature(x, h_0.detach())
        output = self.fc(output[:, :, :])
        return output


class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        out, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        out = self.dropout(out)
        out = self.norm(out + src)
        return out