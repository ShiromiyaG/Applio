import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from rvc.lib.algorithm.commons import get_padding
from rvc.lib.algorithm.residuals import LRELU_SLOPE
from rvc.lib.algorithm.univhd import UnivHDDiscriminator
from rvc.lib.algorithm.san import SANConv1d, SANConv2d, san_tail


#: ``v4``'s periods.  Not rate-scaled: period 2 is the only branch that folds
#: near Nyquist.
V4_PERIODS = (2, 3, 5, 7, 11)

#: Pre-emphasis on ``v4``'s period branches.  Off: at 0.97 periods 5, 7 and 11
#: stayed near chance for 12k steps.
V4_PRE_EMPHASIS = 0.0

#: How much of the adversarial objective UnivHD is allowed to be on ``v4``.
#:
#: The paper's additive ``1.0`` was wrong for the branch set this version first
#: ran (four rate-scaled periods, single-band spectrogram branches); it has not
#: been re-measured on the current one.  On a 32 kHz pretrain with SAN on,
#: ``mean(real logit) - mean(fake logit)`` per branch between steps 2k and 8.5k
#: put UnivHD at 5.1-7.7 against 0.2-2.1
#: for the other eight heads, and it stayed there rather than converging toward
#: them.  The generator's term is ``(1 - dg)^2``, so a head separating by ~6
#: contributes ~10x an average branch's: one of nine heads was most of
#: ``loss_gen``, and most of the gradient that drove the decoder's grad norm to
#: 8-15 x 10^3.  0.15 leaves it the loudest single head without leaving it the
#: only one.  It applies to the feature-matching and discriminator terms too --
#: a head the generator is told to discount but that trains at full rate keeps
#: pulling away.
UNIVHD_WEIGHT = 0.15

#: The three multi-resolution spectrogram branches.  The 512-point branch's
#: 50-sample hop is what reads frame-rate modulation the other two average
#: away.  On ``v4`` they are ``MultiBandDiscriminatorR``.
V3_RESOLUTIONS = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]
V4_RESOLUTIONS = [[1024, 120, 1024], [2048, 240, 2048], [512, 50, 240]]


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-period discriminator.

    This class implements a multi-period discriminator, which is used to
    discriminate between real and fake audio signals. The discriminator
    is composed of a series of convolutional layers that are applied to
    the input signal at different periods.

    Args:
        use_spectral_norm (bool): Whether to use spectral normalization.
            Defaults to False.
    """

    def __init__(
        self,
        use_spectral_norm: bool = False,
        checkpointing: bool = False,
        version: str = "v2",
        sample_rate: int = 32000,
    ):
        super().__init__()

        univhd = False
        san = False
        msd = True
        hann = False
        multiband = False
        pre_emphasis = 0.0
        if version == "v1":
            periods = [2, 3, 5, 7, 11, 17]
            resolutions = []
        elif version == "v2":
            periods = [2, 3, 5, 7, 11, 17, 23, 37]
            resolutions = []
        elif version == "v3":
            periods = [2, 3, 5, 7, 11]
            resolutions = V3_RESOLUTIONS
        elif version == "v4":
            # RefineGAN2's discriminator: v3's periods with pre-emphasis,
            # multi-band spectrogram branches and the harmonic branch added.
            # UnivHD is 0.33 M parameters and 8% of the step, and the paper
            # reports it beating either branch family alone only when added to
            # one rather than replacing it.
            periods = V4_PERIODS
            resolutions = V4_RESOLUTIONS
            univhd = True
            multiband = True
            pre_emphasis = V4_PRE_EMPHASIS
            # SAN (arXiv 2301.12811) is part of this layout, not a switch on it.
            # It replaces every branch's last projection with a unit-norm
            # direction plus a scale, which changes ``conv_post``'s state-dict
            # keys -- so offering it on v2/v3 would only be a way to make an
            # existing Applio discriminator unloadable.
            san = True
            # No waveform (MSD) branch: the spectrogram branches already cover
            # what it reads.  Dropping it shifts every branch index, so like
            # SAN it is part of this layout rather than a switch.
            msd = False
            # Hann on the full-window spectrogram branches: a boxcar's -13 dB
            # sidelobes fill the inter-harmonic valleys these branches read.
            hann = True
        else:
            raise ValueError(f"Unknown discriminator version {version!r}.")

        self.version = version
        self.periods = list(periods)
        self.use_msd = msd
        self.checkpointing = checkpointing
        #: Read by ``train.py`` to pick the loss form.  An attribute rather than
        #: a version comparison, so the two cannot disagree.
        self.supports_san = san
        self.discriminators = torch.nn.ModuleList(
            ([DiscriminatorS(use_spectral_norm=use_spectral_norm, use_san=san)] if msd else [])
            + [
                DiscriminatorP(
                    p,
                    use_spectral_norm=use_spectral_norm,
                    use_san=san,
                    pre_emphasis=pre_emphasis,
                )
                for p in periods
            ]
            + [
                MultiBandDiscriminatorR(
                    r,
                    use_spectral_norm=use_spectral_norm,
                    use_san=san,
                    frequency_strides=(1, 2, 2),
                )
                if multiband
                else DiscriminatorR(
                    r,
                    use_spectral_norm=use_spectral_norm,
                    use_san=san,
                    hann_window=hann,
                )
                for r in resolutions
            ]
            + (
                [
                    UnivHDDiscriminator(
                        sample_rate=sample_rate,
                        use_spectral_norm=use_spectral_norm,
                        use_san=san,
                    )
                ]
                if univhd
                else []
            )
        )
        #: One loss weight per entry of ``discriminators``, same order, read by
        #: ``train.py`` and handed to the three adversarial losses.  Built here
        #: rather than at the call site so it cannot fall out of step with the
        #: assembly above: a list one entry short would weight the wrong heads.
        self.branch_weights = tuple(
            [1.0] * (int(msd) + len(periods) + len(resolutions))
            + ([UNIVHD_WEIGHT] if univhd else [])
        )

    def forward(self, y, y_hat, san_training: bool = False):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        # Only the discriminator update asks for the direction output: the
        # generator may not move the direction, so requesting it there would
        # build a graph nothing reads.
        san = bool(san_training) and self.supports_san
        for d in self.discriminators:
            if self.training and self.checkpointing:
                y_d_r, fmap_r = checkpoint(d, y, san_training=san, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(
                    d, y_hat, san_training=san, use_reentrant=False
                )
            else:
                y_d_r, fmap_r = d(y, san_training=san)
                y_d_g, fmap_g = d(y_hat, san_training=san)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    """
    Discriminator for the short-term component.

    This class implements a discriminator for the short-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal.
    """

    def __init__(self, use_spectral_norm: bool = False, use_san: bool = False):
        super().__init__()

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm_f(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.use_san = bool(use_san)
        # No ``norm_f`` on a SAN head: it normalises its own weight, and the two
        # reparametrisations would fight over the same tensor.
        self.conv_post = (
            SANConv1d(1024, 1, 3, 1, padding=1)
            if self.use_san
            else norm_f(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))
        )
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x, san_training: bool = False):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)


def pre_emphasize(x, coefficient):
    """``x[t] - coefficient * x[t-1]``; 0 returns ``x`` unchanged."""
    if not coefficient:
        return x
    return torch.cat((x[..., :1], x[..., 1:] - coefficient * x[..., :-1]), dim=-1)


class DiscriminatorP(torch.nn.Module):
    """
    Discriminator for the long-term component.

    This class implements a discriminator for the long-term component
    of the audio signal. The discriminator is composed of a series of
    convolutional layers that are applied to the input signal at a given
    period.

    Args:
        period (int): Period of the discriminator.
        kernel_size (int): Kernel size of the convolutional layers. Defaults to 5.
        stride (int): Stride of the convolutional layers. Defaults to 3.
        use_spectral_norm (bool): Whether to use spectral normalization. Defaults to False.
        pre_emphasis (float): Pre-emphasis coefficient on the input. Defaults to 0 (off).
    """

    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        use_san: bool = False,
        pre_emphasis: float = 0.0,
    ):
        super().__init__()
        self.period = period
        self.pre_emphasis = float(pre_emphasis)
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        in_channels = [1, 32, 128, 512, 1024]
        out_channels = [32, 128, 512, 1024, 1024]
        strides = [3, 3, 3, 3, 1]

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        (kernel_size, 1),
                        (s, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
                for in_ch, out_ch, s in zip(in_channels, out_channels, strides)
            ]
        )

        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(1024, 1, (3, 1), 1, padding=(1, 0))
            if self.use_san
            else norm_f(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        )
        self.lrelu = torch.nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x, san_training: bool = False):
        fmap = []
        x = pre_emphasize(x, self.pre_emphasis)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = torch.nn.functional.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)


class DiscriminatorR(torch.nn.Module):
    """
    Multi-resolution spectrogram discriminator.

    Args:
        resolution (list[int]): ``[n_fft, hop_length, win_length]``.
        use_spectral_norm (bool, optional): Spectral instead of weight norm. Defaults to False.
        use_san (bool, optional): SAN projection on ``conv_post``. Defaults to False.
        hann_window (bool, optional): Hann window where ``win_length == n_fft``; the
            short temporal window keeps the boxcar. Defaults to False.
    """

    def __init__(
        self,
        resolution,
        use_spectral_norm=False,
        use_san=False,
        hann_window=False,
    ):
        super().__init__()

        self.resolution = resolution
        self.lrelu_slope = 0.1
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = torch.nn.ModuleList(
            [
                norm_f(
                    torch.nn.Conv2d(
                        1,
                        32,
                        (3, 9),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    torch.nn.Conv2d(
                        32,
                        32,
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(32, 1, (3, 3), padding=(1, 1))
            if self.use_san
            else norm_f(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))
        )

        # Non-persistent, so no checkpoint gains a key.
        n_fft, _hop, win_length = self.resolution
        self.register_buffer(
            "window",
            torch.hann_window(int(win_length))
            if hann_window and int(win_length) == int(n_fft)
            else torch.ones(int(win_length)),
            persistent=False,
        )

    def forward(self, x, san_training: bool = False):
        fmap = []

        x = self.spectrogram(x).unsqueeze(1)

        for layer in self.convs:
            x = F.leaky_relu(layer(x), self.lrelu_slope)
            fmap.append(x)
        return san_tail(self, x, fmap, san_training)

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(
            x,
            (pad, pad),
            mode="reflect",
        ).squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )

        mag = torch.norm(torch.view_as_real(x), p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiBandDiscriminatorR(torch.nn.Module):
    """Spectrogram branch on a compressed complex STFT, one conv stack per band.

    Three input channels: log magnitude, and the real and imaginary parts of
    the STFT with its magnitude raised to ``compression`` (phase kept).  A
    linear magnitude leaves the noise floor and the upper bands numerically
    near zero, and has no phase at all.  The frequency axis is split into
    ``BANDS`` (fractions of the bins), each with its own stack, as in DAC's
    MRD, so the low band cannot claim every filter.

    Args:
        resolution (list[int]): ``[n_fft, hop_length, win_length]``.
        channels (int, optional): Width of every band stack. Defaults to 32.
        use_spectral_norm (bool, optional): Spectral instead of weight norm. Defaults to False.
        use_san (bool, optional): SAN projection on ``conv_post``. Defaults to False.
        compression (float, optional): Magnitude exponent. Defaults to 0.3.
        frequency_strides (tuple[int], optional): Frequency stride of the three
            strided layers. Defaults to (1, 1, 1).
    """

    BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
    # Power floor: keeps log and the compression gain finite in silence,
    # about 94 dB under a full-scale sine at these magnitudes.
    POWER_EPS = 1e-4

    def __init__(
        self,
        resolution,
        channels=32,
        use_spectral_norm=False,
        use_san=False,
        compression=0.3,
        frequency_strides=(1, 1, 1),
    ):
        super().__init__()
        self.resolution = resolution
        self.compression = float(compression)
        self.lrelu_slope = 0.1
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        n_fft, _hop, win_length = self.resolution
        n_bins = int(n_fft) // 2 + 1
        self.band_edges = tuple(
            (int(round(lo * n_bins)), int(round(hi * n_bins))) for lo, hi in self.BANDS
        )

        def band_stack():
            return torch.nn.ModuleList(
                [norm_f(torch.nn.Conv2d(3, channels, (3, 9), padding=(1, 4)))]
                + [
                    norm_f(
                        torch.nn.Conv2d(
                            channels, channels, (3, 9), stride=(s, 2), padding=(1, 4)
                        )
                    )
                    for s in frequency_strides
                ]
                + [norm_f(torch.nn.Conv2d(channels, channels, (3, 3), padding=(1, 1)))]
            )

        self.bands = torch.nn.ModuleList(band_stack() for _ in self.BANDS)
        self.use_san = bool(use_san)
        self.conv_post = (
            SANConv2d(channels, 1, (3, 3), padding=(1, 1))
            if self.use_san
            else norm_f(torch.nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))
        )
        # Hann where the window spans the whole transform; the short temporal
        # window keeps the boxcar.  Non-persistent, so no checkpoint gains a key.
        self.register_buffer(
            "window",
            torch.hann_window(int(win_length))
            if int(win_length) == int(n_fft)
            else torch.ones(int(win_length)),
            persistent=False,
        )

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        pad = int((n_fft - hop_length) / 2)
        x = F.pad(x, (pad, pad), mode="reflect").squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        # Through the power rather than ``abs`` so the gradient stays finite
        # at a zero bin.
        power = x.real.square() + x.imag.square() + self.POWER_EPS
        gain = power ** ((self.compression - 1.0) / 2.0)
        return torch.stack(
            (0.5 * torch.log(power), x.real * gain, x.imag * gain), dim=1
        )

    def forward(self, x, san_training: bool = False):
        # Only the STFT leaves autocast; after compression the input is in the
        # range the other branches see.
        with torch.autocast(x.device.type, enabled=False):
            x = self.spectrogram(x.float())
        layers = [[] for _ in self.bands[0]]
        for (lo, hi), stack in zip(self.band_edges, self.bands):
            h = x[:, :, lo:hi]
            for index, layer in enumerate(stack):
                h = F.leaky_relu(layer(h), self.lrelu_slope)
                layers[index].append(h)
        # One entry per layer, a tuple of its band maps: ``feature_loss`` takes
        # their joint mean, so the branch weighs what ``DiscriminatorR`` does
        # without copying the activations into one tensor.
        fmap = [tuple(maps) for maps in layers]
        return san_tail(self, torch.cat(layers[-1], dim=2), fmap, san_training)
