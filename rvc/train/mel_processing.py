import numpy as np
import torch
import torch.utils.data
from librosa import mel_frequencies
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, hop_size, win_size, center=False):
    """
    Compute the spectrogram of a signal using STFT.

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT window size.
        hop_size (int): Hop size between frames.
        win_size (int): Window size.
        center (bool, optional): Whether to center the window. Defaults to False.
    """
    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Convert a spectrogram to a mel-spectrogram.

    Args:
        spec (torch.Tensor): Magnitude spectrogram.
        n_fft (int): FFT window size.
        num_mels (int): Number of mel frequency bins.
        sample_rate (int): Sampling rate of the audio signal.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    melspec = torch.log(melspec.clamp(min=1e-5) * 1)
    return melspec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False
):
    """
    Compute the mel-spectrogram of a signal.

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT window size.
        num_mels (int): Number of mel frequency bins.
        sample_rate (int): Sampling rate of the audio signal.
        hop_size (int): Hop size between frames.
        win_size (int): Window size.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        center (bool, optional): Whether to center the window. Defaults to False.
    """
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)

    melspec = spec_to_mel_torch(spec, n_fft, num_mels, sample_rate, fmin, fmax)

    return melspec


def compute_window_length(n_mels: int, sample_rate: int):
    f_min = 0
    f_max = sample_rate / 2
    window_length_seconds = 8 * n_mels / (f_max - f_min)
    window_length = int(window_length_seconds * sample_rate)
    return 2 ** (window_length.bit_length() - 1)


class MultiScaleMelSpectrogramLoss(torch.nn.Module):
    """
    Mel L1 evaluated at several STFT resolutions at once and summed.

    Short windows resolve timing and blur the spectrum; long windows do the
    reverse. Scoring both ends charges a defect that only one of them can see.

    The defaults are RefineGAN's scale set. RefineGAN2 uses a different one --
    see ``build_refinegan2_mel_loss`` -- so the two are not interchangeable and
    a scale set carries its own ``output_scale``.

    Args:
        sample_rate (int, optional): Sample rate of the waveforms, in Hz. Defaults to 24000.
        n_mels (list[int], optional): Mel bands per scale. Defaults to [5, 10, 20, 40, 80, 160, 320].
        window_lengths (list[int], optional): STFT window per scale, paired with ``n_mels``. Defaults to [32, 64, 128, 256, 512, 1024, 2048].
        loss_fn (optional): Distance between the two log-mels. Defaults to ``torch.nn.L1Loss()``.
        output_scale (float, optional): Applied to the summed loss, so a scale set
            carries its own normalisation instead of leaving each call site to
            remember a divisor. Defaults to 1.0.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: list[int] = [5, 10, 20, 40, 80, 160, 320],  # , 480],
        window_lengths: list[int] = [32, 64, 128, 256, 512, 1024, 2048],  # , 4096],
        loss_fn=None,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        # Defaulted here rather than in the signature: a module instance as a
        # default argument is built once and shared by every caller that omits it.
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.L1Loss()
        self.output_scale = float(output_scale)
        self.log_base = torch.log(torch.tensor(10.0))
        self.stft_params: list[tuple] = []
        self.hann_window: dict[int, torch.Tensor] = {}
        self.mel_banks: dict[int, torch.Tensor] = {}

        self.stft_params = [(mel, win) for mel, win in zip(n_mels, window_lengths)]

    def mel_spectrogram(
        self,
        wav: torch.Tensor,
        n_mels: int,
        window_length: int,
    ):
        # IDs for caching
        dtype_device = str(wav.dtype) + "_" + str(wav.device)
        win_dtype_device = str(window_length) + "_" + dtype_device
        mel_dtype_device = str(n_mels) + "_" + dtype_device
        # caching hann window
        if win_dtype_device not in self.hann_window:
            self.hann_window[win_dtype_device] = torch.hann_window(
                window_length, device=wav.device, dtype=torch.float32
            )

        wav = wav.squeeze(1)  # -> torch(B, T)

        stft = torch.stft(
            wav.float(),
            n_fft=window_length,
            hop_length=window_length // 4,
            window=self.hann_window[win_dtype_device],
            return_complex=True,
        )  # -> torch (B, window_length // 2 + 1, (T - window_length)/hop_length + 1)

        magnitude = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-6)

        # caching mel filter
        if mel_dtype_device not in self.mel_banks:
            self.mel_banks[mel_dtype_device] = torch.from_numpy(
                librosa_mel_fn(
                    sr=self.sample_rate,
                    n_mels=n_mels,
                    n_fft=window_length,
                    fmin=0,
                    fmax=None,
                )
            ).to(device=wav.device, dtype=torch.float32)

        mel_spectrogram = torch.matmul(
            self.mel_banks[mel_dtype_device], magnitude
        )  # torch(B, n_mels, stft.frames)
        return mel_spectrogram

    def forward(
        self, real: torch.Tensor, fake: torch.Tensor
    ):  # real: torch(B, 1, T) , fake: torch(B, 1, T)
        loss = 0.0
        for p in self.stft_params:
            real_mels = self.mel_spectrogram(real, *p)
            fake_mels = self.mel_spectrogram(fake, *p)
            real_logmels = torch.log(real_mels.clamp(min=1e-5)) / self.log_base
            fake_logmels = torch.log(fake_mels.clamp(min=1e-5)) / self.log_base
            loss += self.loss_fn(real_logmels, fake_logmels)
        return loss * self.output_scale


def mel_frequency_tilt_weights(
    num_mels: int,
    sample_rate: int,
    tilt: float = 0.0,
    max_ratio: float = 8.0,
):
    """
    Per-bin weights that undo part of the mel scale's own bin density.

    An L1 over log-mel gives every bin the same gradient magnitude, so a region
    receives the share of the objective that it holds in *bins*, not in
    spectrum and not in how wrong it is. The mel scale puts those bins at the
    bottom: over 0-16 kHz, 0-2 kHz is 12.5% of the spectrum and 45% of the
    bins, 10-16 kHz is 37.5% and 12%. The split is the same at 40, 80, 160,
    320 and 640 bands, so no choice of scale set moves it.

    Each bin is weighted by its own bandwidth raised to ``tilt``: 0.0 is the
    unweighted loss, 1.0 cancels the warping and every hertz pulls equally.
    The penalty for one and the same -6 dB shelf, normalised per column to its
    1-3 kHz value, on the RefineGAN2 set at 32 kHz:

        shelf at     tilt 0   tilt 0.5   tilt 1.0
        1-3 kHz       1.000     1.000      1.000
        6-10 kHz      0.458     0.964      1.849
        10-13 kHz     0.232     0.591      0.993
        13-16 kHz     0.155     0.443      0.666

    1.0 overshoots -- 6-10 kHz ends up charging more than 1-3 kHz, its bands
    being the wide ones -- so 0.5 is what ``build_refinegan2_mel_loss`` uses.

    Normalised to a mean of 1, so the weighting changes which bins are heard
    and not the scale of the term.

    Args:
        num_mels (int): Mel bands the weights are built for.
        sample_rate (int): Sample rate of the waveforms, in Hz.
        tilt (float, optional): 0.0 is off, 1.0 is one weight per hertz. Defaults to 0.0.
        max_ratio (float, optional): Cap on the spread between the extremes, without
            which the bottom bins are tens of hertz wide and take weights near zero.
            Defaults to 8.0.
    """

    if tilt == 0.0:
        return torch.ones(int(num_mels), dtype=torch.float32)

    # ``+2`` and the trim are how ``librosa.filters.mel`` places centres, and
    # the untrimmed array gives the end bins a neighbour to measure against.
    edges = mel_frequencies(
        n_mels=int(num_mels) + 2, fmin=0.0, fmax=sample_rate / 2, htk=False
    )
    # A triangular mel filter spans its two neighbouring centres, so this is
    # the filter's own width.
    weights = np.maximum(edges[2:] - edges[:-2], 1e-6) ** float(tilt)
    # Clipped around the geometric mean, the weights living in a log domain.
    centre = float(np.exp(np.log(weights).mean()))
    limit = float(max_ratio) ** 0.5
    weights = np.clip(weights, centre / limit, centre * limit)
    return torch.from_numpy((weights / weights.mean()).astype(np.float32))


class BandWeightedL1Loss(torch.nn.Module):
    """
    Mel L1 whose reduction is weighted per bin rather than uniform.

    The weights are rebuilt per resolution, since the multi-scale loss hands
    the same distance 40 to 640 bands and the weighting is defined by
    frequency rather than by bin count.

    Args:
        sample_rate (int): Sample rate of the waveforms, in Hz.
        tilt (float): Passed to ``mel_frequency_tilt_weights``.
        max_ratio (float, optional): Passed to ``mel_frequency_tilt_weights``. Defaults to 8.0.
    """

    def __init__(self, sample_rate: int, tilt: float, max_ratio: float = 8.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.tilt = float(tilt)
        self.max_ratio = float(max_ratio)
        self.weights: dict[str, torch.Tensor] = {}

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        bins = real.shape[-2]
        key = f"{bins}_{real.dtype}_{real.device}"
        if key not in self.weights:
            self.weights[key] = (
                mel_frequency_tilt_weights(
                    bins, self.sample_rate, self.tilt, self.max_ratio
                )
                .reshape(1, -1, 1)
                .to(device=real.device, dtype=real.dtype)
            )
        return ((real - fake).abs() * self.weights[key]).mean()


# RefineGAN2's scale set, which is not RefineGAN's.
#
# The coarse end of the default set (32/64/128) charges almost nothing for a
# defect above 6.5 kHz while charging as much as the fine scales at 1-3 kHz, so
# summing it tilts the loss toward low frequency. Dropping it and adding 4096
# flattens that tilt. Each set normalised to parity at 1-3 kHz, destroying the
# harmonic comb one band at a time:
#
#     config              1-3k   3-6k   6-9k  9-12k  12-15k   span
#     32..2048,  /3.25    1.00   0.97   0.98   0.92    0.73   0.27
#     256..4096, /2.20    1.00   0.98   0.97   0.96    0.92   0.08
#
# 4096 is close to free: 13 frames per 400 ms segment against 26 for 2048.
REFINEGAN2_MEL_N_MELS = [40, 80, 160, 320, 640]
REFINEGAN2_MEL_WINDOWS = [256, 512, 1024, 2048, 4096]

# Chosen for parity with the single-scale L1 on a 1-3 kHz comb, which is where
# RefineGAN's ``/ 3.0`` already sat: what the constants above change is the
# shape across frequency and time, not the overall weight. It rides inside the
# module as ``output_scale`` rather than being applied by callers, since a call
# site that forgot it would train at more than twice the intended mel weight.
REFINEGAN2_MEL_DIVISOR = 2.20

# How far the band weighting undoes the mel scale's bin density. 0.5 roughly
# triples what the bands above 10 kHz charge without reordering them; the loss
# value on a real generated/reference pair moves about 10%, which is the part
# that reaches anything reading the mel term's scale.
REFINEGAN2_MEL_TILT = 0.5


def build_refinegan2_mel_loss(sample_rate: int, loss_fn=None):
    """
    RefineGAN2's multi-scale mel loss: the scale set above, already normalised.

    Weighted with the same ``c_mel`` the single-scale L1 uses, so unlike
    RefineGAN's set it needs no divisor at the call site.

    The distance is band-weighted by default. The scale set flattens the loss
    across frequency only relative to the single-scale L1; what is left is the
    mel warping itself, which the scale set cannot reach -- see
    ``mel_frequency_tilt_weights``.

    Args:
        sample_rate (int): Sample rate of the waveforms, in Hz.
        loss_fn (optional): Distance between the two log-mels. Defaults to a
            ``BandWeightedL1Loss`` at ``REFINEGAN2_MEL_TILT``.
    """

    if loss_fn is None:
        loss_fn = BandWeightedL1Loss(sample_rate, REFINEGAN2_MEL_TILT)
    return MultiScaleMelSpectrogramLoss(
        sample_rate=sample_rate,
        n_mels=REFINEGAN2_MEL_N_MELS,
        window_lengths=REFINEGAN2_MEL_WINDOWS,
        loss_fn=loss_fn,
        output_scale=1.0 / REFINEGAN2_MEL_DIVISOR,
    )
