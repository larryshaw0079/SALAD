import numpy as np

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, data_size, input_channel, latent_size):
        super(ConvEncoder, self).__init__()
        self.__dict__.update(locals())

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []

        for i in range(self.layer_num + 1):
            current_out_channel = self.max_channel_num // 2 ** (self.layer_num - i)

            if i == 0:
                self.conv_list.append(nn.Conv1d(in_channels=self.input_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
            else:
                self.conv_list.append(nn.Conv1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
                self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channel = current_out_channel

        self.conv_layers = nn.Sequential(*self.conv_list)

        self.linear_layers = nn.Sequential(
            nn.Linear(self.final_size * self.max_channel_num, latent_size)
        )

    def forward(self, x):
        out = torch.unsqueeze(x, dim=1)
        out = self.conv_layers(out)
        out = out.view(-1, self.final_size * self.max_channel_num)
        out = self.linear_layers(out)

        return out


class ConvDecoder(nn.Module):
    def __init__(self, data_size, input_channel, latent_size):
        super(ConvDecoder, self).__init__()
        self.__dict__.update(locals())

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []

        self.linear_layers = nn.Sequential(
            nn.Linear(latent_size, self.final_size * self.max_channel_num),
            nn.ReLU(True)
        )

        prev_channel = self.max_channel_num
        for i in range(self.layer_num):
            current_out_channel = self.max_channel_num // 2 ** (i + 1)
            self.conv_list.append(nn.ConvTranspose1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                     kernel_size=4, stride=2, padding=1))
            self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.ReLU(True))
            prev_channel = current_out_channel

        self.conv_list.append(nn.ConvTranspose1d(in_channels=current_out_channel, out_channels=input_channel,
                                                 kernel_size=4, stride=2, padding=1))
        self.conv_list.append(nn.Tanh())

        self.conv_layers = nn.Sequential(*self.conv_list)

    def forward(self, z):
        out = self.linear_layers(z)
        out = out.view(-1, self.max_channel_num, self.final_size)
        out = self.conv_layers(out)
        out = torch.squeeze(out, dim=1)

        return out


class DenseEncoder(nn.Module):
    def __init__(self, data_size, hidden_size, latent_size):
        super(DenseEncoder, self).__init__()
        self.__dict__.update(locals())

        self.layers = nn.Sequential(
            nn.Linear(data_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, x):
        out = self.layers(x)

        return out


class DenseDecoder(nn.Module):
    def __init__(self, data_size, hidden_size, latent_size):
        super(DenseDecoder, self).__init__()
        self.__dict__.update(locals())

        self.layers = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, data_size)
        )

    def forward(self, z):
        out = self.layers(z)

        return out


class ConvDiscriminator(nn.Module):
    def __init__(self, input_size, input_channel):
        super(ConvDiscriminator, self).__init__()
        self.__dict__.update(locals())

        self.layer_num = int(np.log2(input_size)) - 1
        self.max_channel_num = input_size * 2
        self.final_size = 1
        self.conv_list = []

        current_out_channel = None
        for i in range(self.layer_num + 1):
            current_out_channel = self.max_channel_num // 2 ** (self.layer_num - i)

            if i == 0:
                self.conv_list.append(nn.Conv1d(in_channels=self.input_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
            else:
                self.conv_list.append(nn.Conv1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
                self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channel = current_out_channel

        self.conv_layers = nn.Sequential(*self.conv_list)

        self.linear_layers = nn.Sequential(
            nn.Linear(current_out_channel, 1)
        )

    def forward(self, x):
        out = torch.unsqueeze(x, dim=1)
        out = self.conv_layers(out)
        # out = out.view(-1, self.final_size * self.max_channel_num)
        out = out.view(out.size(0), -1)
        out = self.linear_layers(out)

        return out


class DenseDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DenseDiscriminator, self).__init__()
        self.__dict__.update(locals())

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

    def forward(self, z):
        out = self.layers(z)

        return out


# class DataDiscriminator(nn.Module):
#     def __init__(self, data_size, hidden_size):
#         super(DataDiscriminator, self).__init__()
#         self.__dict__.update(locals())
#
#         self.layers = nn.Sequential(
#             nn.Linear(data_size, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(hidden_size, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_size, 1),
#             # nn.BatchNorm1d(1),
#             # nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         out = self.layers(x)
#
#         return out
