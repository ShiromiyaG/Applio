from typing import Sequence

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

from rvc.lib.algorithm.commons import init_weights, get_padding
from rvc.lib.algorithm.resampling import (
    AntiAliasedUpsample1d,
    filter_schedule,
)


# Stage rates per sample rate. Not the config's ascending ``upsample_rates``:
# this decoder wants them descending, so the last residual block has the least
# left to synthesise from scratch. They leave no trace in the state dict, which
# is why they are a table here and not a config key.
REFINEGAN_UPSAMPLE_RATES = {
    24000: (5, 4, 4, 3),
    32000: (5, 4, 4, 4),
    40000: (5, 5, 4, 4),
    48000: (6, 5, 4, 4),
}


def upsample_rates_for(sample_rate: int, hop_length: int):
    """The stage rates for a sample rate, checked against the hop."""

    rates = REFINEGAN_UPSAMPLE_RATES.get(int(sample_rate))
    if rates is None:
        raise ValueError(
            f"RefineGAN has no stage layout for {sample_rate} Hz; known rates "
            f"are {sorted(REFINEGAN_UPSAMPLE_RATES)}."
        )
    product = 1
    for rate in rates:
        product *= rate
    if product != int(hop_length):
        raise ValueError(
            f"RefineGAN's stages {rates} multiply to {product}, but the hop "
            f"length at {sample_rate} Hz is {hop_length}."
        )
    return rates


# Interpolation filter for the trunk's upsamplers, one entry per stage.
# Zero-stuffing copies the input spectrum to every multiple of the input rate;
# what the filter leaves of those copies is an image, and an image at
# ``k*R_in - j*f0`` moves against f0 exactly like a fold does. Only the last
# stage gets a long kernel: its image is the loudest at the output, and the
# early stages are short enough that a long kernel reads more of the padding
# than of the signal.
DEFAULT_UPSAMPLE_WIDTH = (12, 24, 32, 48)
DEFAULT_UPSAMPLE_ROLLOFF = (0.90, 0.95, 0.97, 0.97)
DEFAULT_UPSAMPLE_BETA = (6.0, 6.0, 6.0, 9.0)

# The excitation gain is one channel, so its upsample chain is free whatever
# the kernel length -- and it is the one path where an image is multiplied onto
# every harmonic as a sideband. It gets the longest design at every stage.
SOURCE_GAIN_WIDTH = 48
SOURCE_GAIN_ROLLOFF = 0.97
SOURCE_GAIN_BETA = 9.0


class ResBlock(nn.Module):
    """
    Residual block with multiple dilated convolutions.

    Args:
        channels (int): Number of channels.
        kernel_size (int, optional): Kernel size for the convolutional layers. Defaults to 7.
        dilation (tuple[int], optional): Dilation rates for the convolutional layers. Defaults to (1, 3, 5).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for d in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x

        return x


class AdaIN(nn.Module):
    """
    Noise-regularised activation, wrapped either side of every ResBlock.

    The noise is a training-time regulariser only; in eval this is a plain
    Leaky ReLU.

    Args:
        channels (int): Number of channels.
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
    """

    def __init__(
        self,
        *,
        channels: int,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        # safe to use in-place as it is used on a new x+gaussian tensor
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor):
        # skipped in eval: it is a regulariser, and it is 25% of the forward
        if not self.training:
            return self.activation(x)

        gaussian = torch.randn_like(x) * self.weight[None, :, None]

        return self.activation(x + gaussian)


class ParallelResBlock(nn.Module):
    """
    Runs several ResBlocks with different kernel sizes in parallel and averages them.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_sizes (tuple[int], optional): Kernel size of each parallel block. Defaults to (3, 7, 11).
        dilation (tuple[int], optional): Dilation rates inside each block. Defaults to (1, 3, 5).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation: tuple[int] = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.input_conv.apply(init_weights)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    AdaIN(
                        channels=out_channels,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    ResBlock(
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                    AdaIN(
                        channels=out_channels,
                        leaky_relu_slope=leaky_relu_slope,
                    ),
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)


class SineGenerator(nn.Module):
    """
    Sine + additive-noise harmonic excitation source.

    A source that fills the band flat -- the impulse train this replaced --
    arrives unshaped wherever the trunk does not reach, because ``source_gain``
    is one scalar per frame and can move the excitation's level but not its
    tilt: +18 dB against the reference at 4.4-5 kHz, with the crossover exactly
    at the trunk's 3960 Hz ceiling. On a fixed trunk the sine measured better
    anyway: held-out multi-scale mel 1.9714 -> 1.7418 with ``source_gain`` on.

    ``harmonic_num`` puts partials back into the source, alias-free and in tune
    by construction, and ``harmonic_tilt`` gives them a slope instead: partial
    ``j`` at ``j ** -tilt``, 1.0 being a sawtooth's -6 dB/octave, which lands
    within ~5 dB of a real voice across the whole band (relative to 300-600 Hz
    at f0=200: voice -5.5 dB at 1-2 kHz and -33.8 at 12-15.5 kHz, tilt 1.0
    -9.1 and -28.9). It ships at 0 -- one partial, everything else manufactured
    by the trunk -- because overfitting a single clip, this decoder at 0 already
    reproduces a target's harmonic contrast to within 0.7 dB up to 13 kHz, so a
    source short of partials is not what costs a trained model its high
    harmonics. ``dim`` sizes ``merge.0.weight``, so raising the count is a fresh
    pretrain.

    Args:
        samp_rate (int): Output sample rate in Hz.
        harmonic_num (int, optional): Partials above the fundamental. Defaults to 0.
        sine_amp (float, optional): Amplitude of the fundamental. Defaults to 0.1.
        noise_std (float, optional): Gaussian noise std in voiced regions, and the
            only stochastic material the decoder is handed there. Defaults to 0.003.
        voiced_threshold (float, optional): f0 above which a frame counts as voiced. Defaults to 0.0.
        harmonic_tilt (float, optional): Partial ``j`` gets amplitude ``j ** -tilt``. Defaults to 1.0.
    """

    # Fraction of Nyquist over which a partial fades out instead of being
    # switched off. f0 moves between frames, so a hard mask makes the top
    # partial blink on and off around the boundary -- a click at the frame rate.
    NYQUIST_TAPER = 0.1

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        harmonic_tilt: float = 1.0,
    ):
        super().__init__()
        self.sampling_rate = int(samp_rate)
        self.harmonic_num = int(harmonic_num)
        self.dim = self.harmonic_num + 1
        self.sine_amp = float(sine_amp)
        self.noise_std = float(noise_std)
        self.voiced_threshold = float(voiced_threshold)
        self.harmonic_tilt = float(harmonic_tilt)

        # Non-persistent: this module owns exactly one state-dict key, so the
        # tilt changes every partial's level while leaving the file untouched.
        orders = torch.arange(1, self.dim + 1, dtype=torch.float32)
        self.register_buffer(
            "harmonic_gain",
            orders ** (-self.harmonic_tilt),
            persistent=False,
        )

        self.merge = nn.Sequential(nn.Linear(self.dim, 1, bias=False), nn.Tanh())
        # Ones, not ``nn.Linear``'s default ``U(-1, 1)`` at fan_in 1: at
        # ``harmonic_num = 0`` that draw is the only thing scaling the source
        # and nothing downstream normalises it, so over 200 seeds it landed
        # anywhere in +-0.99 -- negative in 48% of them, a 719x spread in
        # excitation RMS. ``sine_amp`` is what states the intended amplitude.
        # At dim > 1 the partials sum instead, and ``harmonic_gain`` has
        # already set their levels, so this layer only has to add them up:
        # RMS 0.0707 at 1 partial, 0.0898 at 32, bounded by 0.0907 at any
        # count, which leaves the ``Tanh`` near-linear (-61.6 dB against its best
        # linear fit at 1 partial, -53.8 at 32).
        nn.init.ones_(self.merge[0].weight)

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(f0) * (f0 > self.voiced_threshold)

    def _nyquist_fade(self, f0_buf: torch.Tensor) -> torch.Tensor:
        """
        Per-partial gain reaching 0 at Nyquist, read off each partial's own
        frequency. Exactly ones at ``harmonic_num = 0``.

        ``_f02sine`` takes ``f / sr`` mod 1, so a partial past Nyquist folds
        back onto the band as an inharmonic line that walks against f0.
        """

        nyquist = self.sampling_rate / 2.0
        return ((nyquist - f0_buf) / (nyquist * self.NYQUIST_TAPER)).clamp(0.0, 1.0)

    def _f02sine(self, f0_values: torch.Tensor) -> torch.Tensor:
        """f0_values: (batch, length, dim), dim = fundamental + overtones."""

        # F0 in rad mod 1; the integer cycle count doesn't affect phase.
        rad_values = (f0_values / self.sampling_rate) % 1

        # Random initial phase per harmonic, none for the fundamental.
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        return torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

    # Inductor cannot compile the phase cumsum -- it lowers to a ``SplitScan``
    # whose codegen raises on torch 2.10, taking the whole decoder's compile
    # down with it. Everything up to ``merge`` is a pure function of f0 under
    # no_grad, so keeping it out of the graph costs no fusion.
    @torch.compiler.disable
    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """f0: (batch, length, 1) at the output rate. Returns (batch, length, 1)."""

        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            # Both are identity at ``harmonic_num = 0``: the gain is ``[1.0]``
            # and no fundamental sits within 10% of Nyquist.
            sine_waves = sine_waves * self.harmonic_gain
            sine_waves = sine_waves * self._nyquist_fade(f0_buf)

            uv = self._f02uv(f0)

            # Unvoiced regions are noise; voiced ones get a small dither.
            # ``merge`` sums ``dim`` independent draws, so without the
            # ``sqrt(dim)`` the dither the decoder receives would grow with the
            # harmonic count and ``noise_std`` would stop meaning what it says.
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves) / self.dim**0.5

            sine_waves = sine_waves * uv + noise

        # Merged with grad: one learned scalar per partial.
        return self.merge(sine_waves)


class RefineGAN2Generator(nn.Module):
    """
    RefineGAN2 generator for audio synthesis.

    Downsamples and upchannels an excitation, fuses it with the latent, and
    upsamples through parallel residual blocks. Against the original: a
    tilted harmonic sine instead of the truncated-sinc comb, descending stage
    rates, a windowed-sinc interpolation filter that crops its own group delay, an
    excitation gain projected from the conditioning, and f0 interpolated in log
    with a hard voiced/unvoiced gate. Every pointwise nonlinearity is a plain
    Leaky ReLU at its own rate.

    Args:
        sample_rate (int, optional): Sampling rate of the audio. Defaults to 32000.
        upsample_rates (tuple[int], optional): Upsampling rate of each stage, descending. Defaults to (8, 8, 2, 2).
        leaky_relu_slope (float, optional): Slope for the Leaky ReLU activation. Defaults to 0.2.
        num_mels (int, optional): Number of channels in the conditioning. Defaults to 128.
        start_channels (int, optional): Channels in the first downsampling block. Defaults to 16.
        gin_channels (int, optional): Channels for the global conditioning input. Defaults to 256.
        checkpointing (bool, optional): Whether to use checkpointing for memory efficiency. Defaults to False.
        upsample_initial_channel (int, optional): Channels at the top of the trunk. Defaults to 512.
        filter_width (int | Sequence[int], optional): Interpolation filter length, scalar or one per stage.
        rolloff (float | Sequence[float], optional): Fraction of the stage's Nyquist the filter keeps.
        filter_beta (float | Sequence[float], optional): Kaiser beta for the interpolation filter.
        source_gain (bool, optional): Scale the excitation by an intensity envelope
            projected from the conditioning, as RefineGAN's paper does with the mel. Defaults to False.
        source_noise_std (float, optional): Dither the excitation carries in voiced
            frames. Defaults to 0.01.
        source_harmonics (int, optional): Partials above the fundamental in the
            excitation. Sizes ``m_source.merge.0.weight``, so it cannot change on a
            resume. Defaults to 0.
        source_tilt (float, optional): Partial ``j`` gets amplitude ``j ** -tilt``.
            Leaves no state-dict key. Defaults to 1.0.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 32000,
        upsample_rates: tuple[int] = (8, 8, 2, 2),
        leaky_relu_slope: float = 0.2,
        num_mels: int = 128,
        start_channels: int = 16,
        gin_channels: int = 256,
        checkpointing: bool = False,
        upsample_initial_channel=512,
        filter_width: "int | Sequence[int]" = DEFAULT_UPSAMPLE_WIDTH,
        rolloff: "float | Sequence[float]" = DEFAULT_UPSAMPLE_ROLLOFF,
        filter_beta: "float | Sequence[float]" = DEFAULT_UPSAMPLE_BETA,
        source_gain: bool = False,
        source_noise_std: float = 0.01,
        source_harmonics: int = 0,
        source_tilt: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.checkpointing = checkpointing

        # The down path doubles start_channels per stage and the up path
        # expects the skip to be a quarter of the trunk, so the two only meet at
        # one value. Anything else builds, then fails deep in the up path on a
        # channel mismatch that says nothing about which setting was wrong.
        required = upsample_initial_channel // (4 * 2 ** (len(upsample_rates) - 1))
        if int(start_channels) != required:
            raise ValueError(
                f"start_channels must be {required} for "
                f"upsample_initial_channel={upsample_initial_channel} over "
                f"{len(upsample_rates)} stages, not {start_channels}."
            )

        # Scalar or one per stage, normalised in one place.
        count = len(upsample_rates)
        self.filter_width = filter_schedule(filter_width, count, "filter_width", 1)
        self.rolloff = filter_schedule(rolloff, count, "rolloff", 0.0)
        self.filter_beta = filter_schedule(filter_beta, count, "filter_beta", 0.0)

        # Checked here rather than in ``filter_schedule``, which has no way to
        # know the setting is a fraction: a rolloff above 1.0 asks the kernel to
        # pass beyond the stage's Nyquist, which is the one thing it is for.
        if any(value > 1.0 for value in self.rolloff):
            raise ValueError(
                f"rolloff is a fraction of the stage's Nyquist and cannot "
                f"exceed 1.0, received {self.rolloff}."
            )

        # ``int``, not the np.int64 np.prod returns: Dynamo wraps a numpy
        # scalar as a CPU tensor, and one CPU node makes Inductor emit a C++
        # kernel, which on Windows needs cl.exe and fails the whole compile.
        self.upp = int(np.prod(upsample_rates))

        # The excitation. ``source_noise_std`` is the only stochastic material
        # the decoder is handed in voiced frames, and real voiced speech above
        # 10 kHz is barely harmonic at all -- its floor between the partials
        # sits 2.6 dB under them. Swept on a 4-epoch pretrain at 32 kHz, 0.01
        # improves the 10 kHz deficit (-2.00 -> -1.21 dB) and the multi-scale
        # mel (0.723 -> 0.713) at once over the 0.003 this shipped with; 0.03
        # closes more of the band and costs the mel.
        #
        # The tilt leaves no state-dict key, so a checkpoint trained under one
        # loads into another silently. The count does not: it sizes
        # ``merge.0.weight``.
        self.source_noise_std = float(source_noise_std)
        self.source_harmonics = int(source_harmonics)
        self.source_tilt = float(source_tilt)
        self.m_source = SineGenerator(
            sample_rate,
            harmonic_num=self.source_harmonics,
            noise_std=self.source_noise_std,
            harmonic_tilt=self.source_tilt,
        )

        # ``start_channels``, not a literal 16.  It was hardcoded here while
        # the down path below was built from ``start_channels``, so any value
        # but 16 produced a channel mismatch at ``downsample_blocks[0]`` -- a
        # config knob that could only take one value, which is worse than no
        # knob.
        self.pre_conv = weight_norm(
            nn.Conv1d(1, start_channels, 7, 1, padding=3)
        )

        channels = start_channels
        size = self.upp
        self.downsample_blocks = nn.ModuleList([])
        self.df0 = []
        for i, u in enumerate(upsample_rates):

            new_size = int(size / upsample_rates[-i - 1])
            # T dimension factors for torchaudio.functional.resample
            self.df0.append([size, new_size])
            size = new_size

            new_channels = channels * 2
            self.downsample_blocks.append(
                weight_norm(nn.Conv1d(channels, new_channels, 7, 1, padding=3))
            )
            channels = new_channels

        channels = upsample_initial_channel

        self.mel_conv = weight_norm(
            nn.Conv1d(
                num_mels,
                channels // 2,
                7,
                1,
                padding=3,
            )
        )

        self.mel_conv.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, channels // 2, 1)

        # The paper scales its template by intensity values read off the mel;
        # this decoder is handed z, from which a least-squares fit recovers the
        # log intensity at r = 0.996. The sine carries no envelope of its own,
        # so this is worth having: held-out multi-scale mel on a fixed trunk
        # improves 1.97 -> 1.74, for 193 parameters.
        self.has_source_gain = bool(source_gain)
        if self.has_source_gain:
            self.source_gain = nn.Conv1d(num_mels, 1, 1)
            # Identity at initialisation (softplus(0.5413) = 1.0 with zero
            # weights), so switching this on starts from exactly the excitation
            # the run had before and the projection earns every departure.
            nn.init.zeros_(self.source_gain.weight)
            nn.init.constant_(self.source_gain.bias, 0.5413248546129181)

            # The gain multiplies the excitation, so a residual image in it
            # stamps a sideband onto every harmonic. This chain runs on
            # (B, 1, T), where taps are free, so every stage gets the longest
            # design rather than the trunk's schedule.
            self.source_gain_ups = nn.ModuleList(
                [
                    AntiAliasedUpsample1d(
                        rate,
                        filter_width=SOURCE_GAIN_WIDTH,
                        rolloff=SOURCE_GAIN_ROLLOFF,
                        filter_beta=SOURCE_GAIN_BETA,
                    )
                    for rate in upsample_rates
                ]
            )

        self.upsample_blocks = nn.ModuleList([])
        self.upsample_conv_blocks = nn.ModuleList([])

        for stage, rate in enumerate(upsample_rates):
            new_channels = channels // 2

            # Was nn.Upsample(mode="linear"), whose triangular kernel rejects
            # the first image by only 1.7-9.6 dB, stamping the frame grid into
            # the waveform as a mirrored partial either side of every harmonic.
            self.upsample_blocks.append(
                AntiAliasedUpsample1d(
                    rate,
                    filter_width=self.filter_width[stage],
                    rolloff=self.rolloff[stage],
                    filter_beta=self.filter_beta[stage],
                )
            )

            self.upsample_conv_blocks.append(
                ParallelResBlock(
                    in_channels=channels + channels // 4,
                    out_channels=new_channels,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                )
            )

            channels = new_channels

        self.conv_post = weight_norm(
            nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        )
        self.conv_post.apply(init_weights)

        self.out_tanh = nn.Tanh()

    # torchaudio builds its sinc kernel from Python ints on every call, which
    # Inductor compiles to a CPU kernel and fails on Windows without cl.exe.
    # Kept out of the graph rather than replaced: this filter is what keeps
    # each decimation from folding the harmonics it discards.
    @torch.compiler.disable
    def _decimate(self, x: torch.Tensor, orig_freq: int, new_freq: int):
        return torchaudio.functional.resample(
            x.contiguous(),
            orig_freq=orig_freq,
            new_freq=new_freq,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )

    @staticmethod
    def _expand_f0(f0: torch.Tensor, length: int) -> torch.Tensor:
        """
        f0 at the frame rate -> f0 at the output rate, (batch, 1, length).

        Interpolating in Hz makes the frame-rate ripple a constant absolute
        wobble, whose sidebands grow with the harmonic number; in log it is
        constant in cents instead. And interpolating across a voiced/unvoiced
        boundary ramps f0 toward zero while the gate stays open, which chirps
        every harmonic at once, so the gate is interpolated separately.
        """

        voiced = (f0 > 0).to(f0.dtype)
        # Interpolate the *pitch*, in log Hz, and the gate separately.
        log_f0 = torch.log(f0.clamp_min(1.0))
        log_f0 = F.interpolate(log_f0, size=length, mode="linear", align_corners=False)
        voiced = F.interpolate(voiced, size=length, mode="nearest")
        return torch.exp(log_f0) * voiced

    def _apply_source_gain(self, har_source: torch.Tensor, mel: torch.Tensor):
        """
        Scale the excitation by an intensity envelope read off the
        conditioning, which arrives at the frame rate as ``mel``.
        """

        if not self.has_source_gain:
            return har_source
        gain = F.softplus(self.source_gain(mel))
        for ups in self.source_gain_ups:
            gain = ups(gain)
        length = har_source.shape[-1]
        if gain.shape[-1] > length:
            gain = gain[..., :length]
        elif gain.shape[-1] < length:
            gain = F.pad(gain, (0, length - gain.shape[-1]), mode="replicate")
        return har_source * gain

    def forward(self, mel: torch.Tensor, f0: torch.Tensor, g: torch.Tensor = None):
        f0_size = mel.shape[-1]
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        f0 = self._expand_f0(f0, f0_size * self.upp)
        # ``SineGenerator`` works in (batch, time, dim), where dim is the
        # harmonic axis; the trunk is channel-first throughout.
        har_source = self.m_source(f0.transpose(1, 2)).transpose(1, 2)
        har_source = self._apply_source_gain(har_source, mel)
        x = self.pre_conv(har_source)
        downs = []
        for index, (block, (old_size, new_size)) in enumerate(
            zip(self.downsample_blocks, self.df0)
        ):
            if index == 0:
                x = F.leaky_relu(x, self.leaky_relu_slope)
            downs.append(x)
            x = self._decimate(x, int(f0_size * old_size), int(f0_size * new_size))
            x = block(x)

        mel = self.mel_conv(mel)
        if g is not None:
            mel = mel + self.cond(g)

        x = torch.cat([mel, x], dim=1)

        for ups, res, down in zip(
            self.upsample_blocks,
            self.upsample_conv_blocks,
            reversed(downs),
        ):
            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
                x = F.leaky_relu(x, self.leaky_relu_slope)
                x = torch.cat([x, down], dim=1)
                x = checkpoint(res, x, use_reentrant=False)
            else:
                x = ups(x)
                x = F.leaky_relu(x, self.leaky_relu_slope)
                x = torch.cat([x, down], dim=1)
                x = res(x)

        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.conv_post(x)
        x = self.out_tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        """
        Fold every weight norm back into its weight, by walking the modules
        rather than listing them by name.
        """

        for module in list(self.modules()):
            if hasattr(module, "parametrizations") and hasattr(
                module.parametrizations, "weight"
            ):
                remove_parametrizations(module, "weight", leave_parametrized=True)