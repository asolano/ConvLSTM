from audioop import bias
import torch.nn as nn
import torch
from ConvLSTM import ConvLSTM
from SHLT import SupermaskConv


class Seq2Seq(nn.Module):
    def __init__(
        self,
        num_channels,
        num_kernels,
        kernel_size,
        padding,
        activation,
        frame_size,
        num_layers,
    ):

        super(Seq2Seq, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1",
            ConvLSTM(
                in_channels=num_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=padding,
                activation=activation,
                frame_size=frame_size,
            ),
        )

        #"batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels, affine=False)
        )

        # Add rest of the layers
        for l in range(2, num_layers + 1):

            self.sequential.add_module(
                f"convlstm{l}",
                ConvLSTM(
                    in_channels=num_kernels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    frame_size=frame_size,
                ),
            )

            self.sequential.add_module(
                #f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels, affine=False)
            )

        # Add Convolutional Layer to predict output frame
        #self.conv = nn.Conv2d(
        self.conv = SupermaskConv(
            in_channels=num_kernels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

    def forward(self, X):

        # Forward propagation through all the layers
        output = self.sequential(X)

        # Return only the last output frame
        output = self.conv(output[:, :, -1])

        return nn.Sigmoid()(output)

