import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QubitNormalization(nn.Module):
    def __init__(self, num_wires, temp):
        super().__init__()

        self.num_wires = num_wires

        self.register_buffer(
            "temp",
            torch.tensor(float(temp), dtype=torch.float32)
        )

    def forward(self, x):

        # Numerically stable equivalent of:
        # sqrt(softmax(x / temperature))

        scaled = x / (2.0 * self.temp)

        # Stable softmax computation
        scaled = scaled - scaled.max(
            dim=1, keepdim=True
        ).values

        amplitudes = torch.exp(scaled)

        # L2 normalize the amplitude vector
        amplitudes = F.normalize(
            amplitudes,
            p=2,
            dim=1,
            eps=1e-12
        )

        target_size = 2 ** self.num_wires

        if amplitudes.shape[1] < target_size:
            amplitudes = F.pad(
                amplitudes,
                (0, target_size - amplitudes.shape[1])
            )

        return amplitudes


class Unitary(nn.Module):
    def __init__(
        self,
        num_filters,
        num_layers,
        num_wires,
        device
    ):
        super(Unitary, self).__init__()

        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.device = device
        self.pi_half = math.pi / 2

    def forward(self, weight, phi):

        final_product = torch.eye(
            int(2 ** self.num_wires),
            dtype=torch.complex64,
            device=self.device
        ).unsqueeze(0).repeat(
            self.num_filters, 1, 1
        )

        for layer in range(self.num_layers):

            current_weight = weight[:, layer, :, :]

            angle = (
                current_weight[..., 0] *
                self.pi_half
            )

            cos_term = torch.cos(angle)
            sin_term = torch.sin(angle)

            exp_term1 = torch.exp(
                1j * phi[:, layer, :]
            )

            exp_term2 = torch.exp(
                1j * current_weight[..., 1]
            )

            U = torch.zeros(
                self.num_filters,
                self.num_wires,
                2,
                2,
                dtype=torch.complex64,
                device=self.device
            )

            U[..., 0, 0] = cos_term
            U[..., 0, 1] = (
                -sin_term * exp_term2
            )
            U[..., 1, 0] = (
                sin_term * exp_term1
            )
            U[..., 1, 1] = (
                cos_term *
                exp_term1 *
                exp_term2
            )

            kron_product = U[:, 0]

            for i in range(1, self.num_wires):

                kron_product = torch.einsum(
                    'bij,bkl->bikjl',
                    kron_product,
                    U[:, i]
                ).view(
                    int(self.num_filters),
                    int(2 ** (i + 1)),
                    int(2 ** (i + 1))
                )

            final_product = torch.bmm(
                final_product,
                kron_product
            )

        return final_product


class Measurement(nn.Module):
    def __init__(self, num_wires, device):
        super(Measurement, self).__init__()

        self.num_wires = num_wires
        self.device = device
        self.obs_list = self._prepare_measurements()

    def _prepare_measurements(self):

        I = torch.eye(
            2,
            dtype=torch.complex64,
            device=self.device
        )

        Z = torch.tensor(
            [[1, 0], [0, -1]],
            dtype=torch.complex64,
            device=self.device
        )

        results = []

        for i in range(self.num_wires):

            operator = torch.tensor(
                1,
                dtype=torch.complex64,
                device=self.device
            )

            for j in range(self.num_wires):

                if i == j:
                    operator = torch.kron(
                        operator, Z
                    )
                else:
                    operator = torch.kron(
                        operator, I
                    )

            results.append(operator)

        return torch.stack(results).to(
            self.device
        )

    def forward(self):
        return self.obs_list


class Quanv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias=False,
        device="cpu",
        temp: float=1.0,
        shots: int=0
    ):
        super().__init__()

        self.num_wires = torch.ceil(
            torch.log2(
                torch.max(
                    torch.tensor([
                        torch.tensor(
                            in_channels * kernel_size,
                            dtype=torch.float32
                        ),
                        2
                    ])
                )
            )
        ).int().item()

        self.num_filters = (
            out_channels + self.num_wires - 1
        ) // self.num_wires

        self.num_layers = kernel_size
        self.out_channels = out_channels

        self.patcher = nn.Unfold(
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )

        self.device = device

        self.temp = torch.tensor(
            temp,
            dtype=torch.float32
        )

        self.shots = shots

        self.qubit_norm = QubitNormalization(
            self.num_wires,
            self.temp
        )

        self.unitary_op = Unitary(
            self.num_filters,
            self.num_layers,
            self.num_wires,
            device
        )

        self.measurement_op = Measurement(
            self.num_wires,
            device
        )

        self.weight = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.randn(
                    (
                        self.num_filters,
                        self.num_layers,
                        self.num_wires,
                        2
                    ),
                    device=self.device
                )
            ),
            requires_grad=True
        )

        self.bias = (
            nn.Parameter(
                torch.zeros(
                    out_channels,
                    device=self.device
                ),
                requires_grad=True
            )
            if bias else None
        )

        self.register_buffer(
            "phi",
            torch.rand(
                self.num_filters,
                self.num_layers,
                self.num_wires,
                device=self.device
            ),
            persistent=True
        )

    def adjust_probabilities_with_shots(self, x):

        p = torch.abs(x) ** 2

        noise_scale = torch.sqrt(
            p * (1 - p) / self.shots
        )

        x = x + torch.complex(
            torch.randn_like(x.real) * noise_scale,
            torch.randn_like(x.imag) * noise_scale
        )

        x = x / torch.sqrt(
            torch.sum(
                torch.abs(x) ** 2,
                dim=-1,
                keepdim=True
            )
        )

        return x

    def forward(self, x):

        x = torch.unsqueeze(x, -1)

        x = self.patcher(x)

        bs, ch_ker, l = x.shape

        x = x.permute(
            0, 2, 1
        ).contiguous().view(
            bs * l,
            ch_ker
        )

        x = self.qubit_norm(x).to(
            torch.cfloat
        )

        unitaries = self.unitary_op(
            self.weight,
            self.phi
        )

        x = torch.einsum(
            'bik,fjk->bfij',
            x.unsqueeze(dim=1),
            unitaries
        )

        x = (
            self.adjust_probabilities_with_shots(x)
            if self.shots > 0
            else x
        )

        intermediate = torch.einsum(
            'bfij,wjk->bfwik',
            x.conj(),
            self.measurement_op()
        )

        x = torch.einsum(
            'bfwik,bfik->bfw',
            intermediate,
            x
        )

        x = x.real.contiguous().view(
            bs * l,
            -1
        )

        if self.bias is not None:
            x = (
                x[:, :self.out_channels] +
                self.bias
            )
        else:
            x = x[:, :self.out_channels]

        x = x.contiguous().view(
            bs,
            l,
            self.out_channels
        ).permute(0, 2, 1)

        return x

class ChannelShuffle1D(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        # x shape: (batch, channels, time)
        batch, channels, length = x.shape

        if channels % self.groups != 0:
            raise ValueError(
                f"{channels} channels not divisible "
                f"by {self.groups} groups"
            )

        channels_per_group = channels // self.groups

        x = x.reshape(
            batch,
            self.groups,
            channels_per_group,
            length
        )

        x = x.transpose(1, 2).contiguous()

        return x.reshape(batch, channels, length)


class CrossResidualBlock(nn.Module):
    def __init__(self, in_channels, device):
        super().__init__()

        assert in_channels % 8 == 0, (
            "in_channels must be divisible by 8"
        )

        quarter = in_channels // 4
        half = in_channels // 2

        def quanv_norm(
            inp, outp, kernel, padding, temp
        ):
            return nn.Sequential(
                Quanv1d(
                    in_channels=inp,
                    out_channels=outp,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    dilation=1,
                    temp=temp,
                    device=device
                ),
                nn.BatchNorm1d(outp)
            )

        self.quanv1 = quanv_norm(
            in_channels,
            quarter,
            13, 6, 1.3
        )

        self.quanv2 = quanv_norm(
            quarter,
            quarter,
            21, 10, 1.1
        )

        self.quanv3 = quanv_norm(
            half,
            half,
            15, 7, 0.9
        )

        self.quanv4 = quanv_norm(
            in_channels,
            in_channels,
            11, 5, 0.7
        )

        self.shuffle1 = ChannelShuffle1D(4)
        self.shuffle2 = ChannelShuffle1D(8)

        self.norm = nn.BatchNorm1d(
            in_channels
        )

        self.acti = nn.Mish()

    def forward(self, x):

        identity = x

        out1 = self.quanv1(x)

        out2 = self.quanv2(out1)

        # Concatenate 2 + 2 = 4 channels
        out12 = torch.cat(
            [out1, out2],
            dim=1
        )

        out12 = self.shuffle1(out12)

        out3 = self.quanv3(out12)

        # Concatenate 2 + 2 + 4 = 8 channels
        out123 = torch.cat(
            [out1, out2, out3],
            dim=1
        )

        out123 = self.shuffle2(out123)

        out4 = self.quanv4(out123)

        out4 = self.acti(
            self.norm(out4)
        )

        return identity + out4


class QuanvNeXt(nn.Module):

    def __init__(
        self,
        num_channels,
        num_classes,
        embed_dim,
        num_layers,
        device
    ):
        super().__init__()

        self.embedding = Quanv1d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=8,
            stride=8,
            padding=0,
            dilation=1,
            device=device
        )

        self.blocks = nn.ModuleList([
            CrossResidualBlock(embed_dim, device)
            for _ in range(num_layers)
        ])

        self.projection = Quanv1d(
            in_channels=embed_dim,
            out_channels=num_classes,
            kernel_size=8,
            stride=8,
            padding=0,
            dilation=1,
            device=device
        )

    def forward(self, x, return_features=False):

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        # Pooled embeddings
        features = F.adaptive_avg_pool1d(
            x, 1
        ).squeeze(-1)

        logits = self.projection(x)

        logits = F.adaptive_avg_pool1d(
            logits, 1
        ).flatten(1)

        if return_features:
            return logits, features

        return logits