import os
import math
import sys

import torch
# import torchlight
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_conv,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        self.drop = nn.Dropout(dropout_conv, inplace=True)

    def forward(self, x, A):
        assert A.size(1) == self.kernel_size
        x = self.drop(self.conv(x))
        x = torch.einsum('nctv,ntvw->nctw', (x, A))

        return x.contiguous(), A


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn=False,
                 stride=1,
                 dropout_tcn=0,
                 dropout_conv=0,
                 residual=True):
        super(st_gcn, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, dropout_conv,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.ReplicationPad2d((0, 0, 2 * ((kernel_size[0] - 1) // 2), 0)),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_tcn, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class social_stgcnn(nn.Module):
    def __init__(self,
                 max_nodes):
        super(social_stgcnn, self).__init__()
        seq_len = 15
        kernel_size = 3
        self.edge_importance_weighting = True

        self.st_gcn_networks = nn.ModuleList()
        self.st_gcn_networks.append(
            st_gcn(512, 64, (kernel_size, seq_len), 1, residual=False, dropout_tcn=0.5, dropout_conv=0.5))
        for i in range(1, 3):
            self.st_gcn_networks.append(st_gcn(64, 64, (kernel_size, seq_len), 1,
                                               residual=False, dropout_tcn=0.5, dropout_conv=0.2))

        self.st_gcn_networks_loc = nn.ModuleList()
        self.st_gcn_networks_loc.append(
            st_gcn(4, 64, (kernel_size, seq_len), 1, residual=False, dropout_tcn=0.5, dropout_conv=0))
        for i in range(1, 1):
            self.st_gcn_networks_loc.append(st_gcn(64, 64, (kernel_size, seq_len), 1,
                                                   residual=False, dropout_tcn=0.5, dropout_conv=0.2))

        self.max_nodes = max_nodes

        # initialize parameters for edge importance weighting
        if self.edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(seq_len, self.max_nodes, self.max_nodes, requires_grad=True))
                for i in self.st_gcn_networks
            ])
            [nn.init.kaiming_normal_(self.edge_importance[i]) for i in range(len(self.edge_importance))]
            self.edge_importance_loc = nn.ParameterList([
                nn.Parameter(torch.ones(seq_len, self.max_nodes, self.max_nodes, requires_grad=True))
                for i in self.st_gcn_networks_loc
            ])
            [nn.init.kaiming_normal_(self.edge_importance_loc[i]) for i in range(len(self.edge_importance_loc))]

        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
            self.edge_importance_loc = [1] * len(self.st_gcn_networks_loc)

        self.bn = nn.BatchNorm1d(512 * max_nodes)
        self.bn_loc = nn.BatchNorm1d(4 * max_nodes)
        self.dec = nn.LSTM(64+64, 128, num_layers=1, bias=True,
                           batch_first=True, bidirectional=False)
        nn.init.xavier_normal_(self.dec.weight_hh_l0)
        self.dropout_dec = nn.Dropout(p=0.4)

        self.fcn = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fcn.weight)

    def forward(self, v, a, loc, node_label,class_label):

        b, t, n, c, h, w = v.size()
        v = v.view(b * t * n, c, h, w)
        v = F.avg_pool2d(v, v.size()[2:])
        _, c_pool, w_pool, h_pool = v.size()
        v = v.view(b, t, n, c_pool, w_pool, h_pool)
        v = v.permute(0, 3, 4, 5, 1, 2).contiguous()
        v = v.view(b, c_pool * w_pool * h_pool, t, n)

        # Batch normalisation
        v = v.permute(0, 3, 1, 2).contiguous()
        v = v.view(b, n * c_pool * w_pool * h_pool, t)
        v = self.bn(v)
        v = v.view(b, n, c_pool * w_pool * h_pool, t)
        v = v.permute(0, 2, 3, 1).contiguous()  # batch,features,time,nodes

        b_loc, t_loc, n_loc, c_loc = loc.size()
        loc = loc.permute(0, 2, 3, 1).contiguous()
        loc = self.bn_loc(loc.view(b_loc, n_loc * c_loc, t_loc))
        loc = loc.view(b_loc, n_loc, c_loc, t_loc)
        loc = loc.permute(0, 2, 3, 1).contiguous()

        b_node, t_node, n_node, lab_node = node_label.size()
        node_label = node_label.permute(0, 3, 1, 2)

        # loc = torch.cat((loc, node_label), dim=1)
        # v = torch.cat((v, node_label), dim=1)

        for graph, imp in zip(self.st_gcn_networks, self.edge_importance):
            v, _ = graph(v, a*imp)

        for graph, imp_loc in zip(self.st_gcn_networks_loc, self.edge_importance_loc):
            loc, _ = graph(loc, a*imp_loc)

        v = v[:, :, :, 0]
        loc = loc[:, :, :, 0]

        x = torch.cat((v, loc), dim=1)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.dec(x)
        x = torch.tanh(self.dropout_dec(x))

        x = self.fcn(x[:, -1])
        x = x.view(x.size(0), -1)

        return x

