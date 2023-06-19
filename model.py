import torch
from torchvision import transforms
from math import log2, sqrt


class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()

        assert (
            kwargs["input_shape"][0] % kwargs["pixel_size"] == 0
        ), "Granularity must be a factor of tile size."
        assert (
            kwargs["pixel_size"] < kwargs["input_shape"][0]
        ), "Granularity must be less than tile size"

        n_pixels = kwargs["input_shape"][0] // kwargs["pixel_size"]
        self.n_layers = int(log2(kwargs["input_shape"][0] // n_pixels))
        self.decode = kwargs["decode"]
        assert self.n_layers < 9
        self.channel_mult = 9 - self.n_layers

        # Encoder
        self.encoder_layers = []
        for n in range(self.n_layers):
            self.encoder_layers.append(
                torch.nn.Conv2d(
                    in_channels=kwargs["in_channels"]
                    if n == 0
                    else 2**self.channel_mult * n,
                    kernel_size=(3, 3),
                    out_channels=2**self.channel_mult * (n + 1),
                    padding="valid",
                    dtype=torch.float32,
                )
            )
            self.encoder_layers.append(torch.nn.ReLU())
            self.encoder_layers.append(
                torch.nn.MaxPool2d(
                    kernel_size=(2, 2),
                )
            )
        self.encoder = torch.nn.Sequential(*self.encoder_layers)

        # Encoded Vector
        self.encoded_layers = [
            torch.nn.Conv2d(
                in_channels=2**self.channel_mult * (self.n_layers),
                kernel_size=(3, 3),
                out_channels=2**self.n_layers * 8,
                padding="valid",
                dtype=torch.float32,
            ),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(
                kernel_size=(int(n_pixels/8),int(n_pixels/8)),
                stride=(1,1),
            ),
            torch.nn.Softmax(dim=1),
        ]
        self.encoded = torch.nn.Sequential(*self.encoded_layers)

        # Decoder
        self.decoder_layers = []
        for n in range(self.n_layers):
            self.decoder_layers.append(torch.nn.Upsample(scale_factor=(2, 2)))
            self.decoder_layers.append(
                torch.nn.Conv2d(
                    in_channels=2**self.n_layers * 8
                    if n == 0
                    else 2**self.channel_mult * (self.n_layers - n + 1),
                    kernel_size=(3, 3),
                    out_channels=2**self.channel_mult * (self.n_layers - n),
                    padding=(2, 2),
                    dtype=torch.float32,
                )
            )
            self.decoder_layers.append(torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(*self.decoder_layers)

        # Decoded image
        self.decoded = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=2**self.channel_mult,
                kernel_size=(3, 3),
                out_channels=kwargs["in_channels"],
                padding=(2, 2),
                dtype=torch.float32,
            ),
            torch.nn.Sigmoid(),
            transforms.CenterCrop(kwargs["input_shape"]),
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)

        # Vector
        x = self.encoded(x)

        if self.decode:
            # Decode
            x = self.decoder(x)

            # Vector
            x = self.decoded(x)

        return x
