import numpy as np
import astropy.units as u
import astropy.constants as ac
import corner
import emcee
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
from astropy.modeling.models import Lorentz1D
from astropy.convolution import (
    convolve_fft,
    Gaussian1DKernel,
    Box1DKernel,
    Model1DKernel,
)
from .constants import c, k_B, h, ckms

arcsec = np.pi / 180.0 / 3600.0  # in rad


########## Plotting utils ##########
def multiplot(
    npanel=None,
    ncols=None,
    nrows=None,
    figsize=(4, 3),
    xlabel=None,
    ylabel=None,
    sharex=False,
    sharey=False,
    max_figsize=(15, None),
):
    if (npanel is not None) and (ncols is None) and (nrows is None):
        nrows = int(npanel**0.5)
        ncols = int(npanel / nrows * 0.999) + 1

        _width, _height = figsize
        width = ncols * _width
        height = nrows * _height

        width_max, height_max = max_figsize

        if (width_max is not None) and width > width_max:
            ncols = int(width_max / _width)
            nrows = int(npanel / ncols * 0.999) + 1
        if (height_max is not None) and (height > height_max):
            nrows = int(height_max / _height)
            ncols = int(npanel / nrows * 0.999) + 1

    # ncols = 5
    # nrows = int(npanel/ncols*0.999) + 1
    fig, axes = plt.subplots(
        figsize=(ncols * figsize[0], nrows * figsize[1]),
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True,
    )
    axes.flatten()[int((nrows - 1) * ncols)].set(xlabel=xlabel, ylabel=ylabel)

    if npanel is not None:
        for j in range(npanel, len(axes.flatten())):
            ax = axes.flatten()[j]
            ax.set_axis_off()

    return fig, axes


def decorate_broken_axis(axes, delta=0.1):
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-d, -1), (d, 1)],
        markersize=10,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    for i, ax in enumerate(axes):
        # xmin, xmax = ax.get_xlim()
        # ymin, ymax = ax.get_ylim()
        if i != len(axes) - 1:
            ax.spines.right.set_visible(False)
            ax.tick_params(right=False)
            ax.plot(
                [1.0 + delta, 1.0 + delta], [0.0, 1.0], transform=ax.transAxes, **kwargs
            )
        if i != 0:
            ax.spines.left.set_visible(False)
            ax.tick_params(left=False)
            ax.plot(
                [0.0 - delta, 0.0 - delta], [0.0, 1.0], transform=ax.transAxes, **kwargs
            )


########## General Functions ##########
def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def nest(d, sep="_"):
    nested = {}
    for k, v in d.items():
        context = nested
        for subkey in k.split(sep)[:-1]:
            if subkey not in context:
                context[subkey] = {}
            context = context[subkey]
        context[k.split(sep)[-1]] = v
    return nested


########### Useful functions / Unit conversion ##########
def shift_freq(nu0, vsys):
    return nu0 * (1.0 - vsys / ckms)


def unshift_freq(nu, vsys):
    return nu / (1.0 - vsys / ckms)


def freq2vel(nu, nu0):
    return (1.0 - nu / nu0) * ckms


def vel2freq(v, nu0):
    return (1.0 - v / ckms) * nu0


def dv2dnu(dv, nu0):
    """convert velocity shift to frequency shift with a given reference frequency nu0

    Parameters
    ----------
    dv : float or ndarray
        velocity shift(s) in km/s
    nu0 : float or ndarray
        reference frequency(s) in Hz

    Returns
    -------
    float or ndarray
        frequency shift in Hz
    """
    return -dv / ckms * nu0


def dv2dnu_abs(dv, nu0):
    """convert velocity width to frequency width (absolute value) with a given reference frequency nu0

    Parameters
    ----------
    dv : float or ndarray
        velocity width(s) in km/s
    nu0 : float or ndarray
        reference frequency(s) in Hz

    Returns
    -------
    float or ndarray
        frequency width in Hz
    """
    return np.abs(-dv / ckms * nu0)


def dnu2dv(dnu, nu0):
    """convert frequency shift to velocity shift with a given reference frequency nu0

    Parameters
    ----------
    dnu : float or ndarray
        frequency shift(s) in Hz
    nu0 : float or ndarray
        reference frequency(s) in Hz

    Returns
    -------
    float or ndarray
        frequency shift in Hz
    """
    return -dnu / nu0 * ckms


def dnu2dv_abs(dnu, nu0):
    """convert frequency width to velocity width (absolute value) with a given reference frequency nu0

    Parameters
    ----------
    dnu : float or ndarray
        frequency shift(s) in Hz
    nu0 : float or ndarray
        reference frequency(s) in Hz

    Returns
    -------
    float or ndarray
        frequency shift in Hz
    """
    return np.abs(-dnu / nu0 * ckms)


def get_beam_solid_angle(beam):
    """Calculate the beam solid angle.

    Parameters
    ----------
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float
        Beam solid angle in steradian.
    """
    return np.multiply(*beam) * arcsec**2 * np.pi / (4 * np.log(2))


def jypb_to_jypsr(I, beam):
    """Convert intensity in Jy / beam to Jy / sr.

    Parameters
    ----------
    I : float or array_like
        Intensity in Jy / beam.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Intensity in Jy / sr.
    """
    Omega_beam = get_beam_solid_angle(beam)
    return I / Omega_beam


def jypsr_to_jypb(I, beam):
    """Convert intensity in Jy / beam to Jy / sr.

    Parameters
    ----------
    I : float or array_like
        Intensity in Jy / beam.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Intensity in Jy / sr.
    """
    Omega_beam = get_beam_solid_angle(beam)
    return I * Omega_beam


def jypsr_to_K_RJ(I, nu):
    """Convert intensity in Jy / sr to K using RJ approx.

    Parameters
    ----------
    I : float or array_like
        Intensity in Jy / sr.
    nu : float or array_like
        Observing frequency in Hz.

    Returns
    -------
    float or array_like
        Intensity in K in R-J approx.
    """
    I *= 1e-23  # in cgs
    return c**2 / (2 * k_B * nu**2) * I


def cgs_to_jypb(I, beam):
    """Convert intensity in Jy / beam to cgs unit. Jy = 1e-23 erg/s/cm2/Hz

    Parameters
    ----------
    I : float or array_like
        Intensity in Jy / beam.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Intensity in erg s-1 cm-2 Hz-1 sr-1.
    """
    return jypsr_to_jypb(I, beam) / 1e-23


def jypb_to_cgs(I, beam):
    """Convert intensity in Jy / beam to cgs unit. Jy = 1e-23 erg/s/cm2/Hz

    Parameters
    ----------
    I : float or array_like
        Intensity in Jy / beam.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Intensity in erg s-1 cm-2 Hz-1 sr-1.
    """
    return jypb_to_jypsr(I, beam) * 1e-23


def jypb_to_K_RJ(I, nu, beam):
    """Convert intensity in Jy / beam to birghtness temeprature in Kelvin using RJ approximation.

    Parameters
    ----------
    I : float or array_like
        Intensity in Jy / beam.
    nu : float or array_like
        Observing frequency in Hz.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Brightness temperature in RJ approximation.
    """
    I = jypb_to_cgs(I, beam)
    return c**2 / (2 * k_B * nu**2) * I


def jypb_to_K(I, nu, beam):
    """Convert intensity in Jy /beam to brightness temperature in Kelvin using full planck function.

    Parameters
    ----------
    I : float or array_like
        Intenisty in Jy /beam.
    nu : flaot or array_like
        Observing frequency in Hz.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Brightness temperature.
    """
    T = np.abs(jypb_to_cgs(I, beam))
    T = h * nu / k_B / np.log(1 + 2 * h * nu**3 / (c**2 * T))

    if isinstance(I, np.ndarray):
        return np.where(I >= 0.0, T, -T)
    elif isinstance(I, float):
        return T if I >= 0.0 else -T


def K_to_jypb(T, nu, beam):
    """Convert intensity in Jy /beam to brightness temperature in Kelvin using full planck function.

    Parameters
    ----------
    I : float or array_like
        Intenisty in Jy /beam.
    nu : flaot or array_like
        Observing frequency in Hz.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Brightness temperature.
    """
    I = 2 * h * nu**3 / c**2 / (np.exp(h * nu / (k_B * np.abs(T))) - 1)
    I = cgs_to_jypb(I, beam)

    if isinstance(T, np.ndarray):
        return np.where(I >= 0.0, I, -I)
    elif isinstance(T, float):
        return I if T >= 0.0 else -I


def K_to_jypb_RJ(T, nu, beam):
    """Convert intensity in Jy /beam to brightness temperature in Kelvin using full planck function.

    Parameters
    ----------
    I : float or array_like
        Intenisty in Jy /beam.
    nu : flaot or array_like
        Observing frequency in Hz.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Brightness temperature.
    """
    I = 2 * k_B * nu**2 / c**2 * T
    I = cgs_to_jypb(I, beam)

    return I


def jypb_to_K_astropy(I, nu, beam):
    """Convert intensity in Jy /beam to brightness temperature in Kelvin using RJ approximation implemented in astropy.

    Parameters
    ----------
    I : float or array_like
        Intenisty in Jy /beam.
    nu : flaot or array_like
        Observing frequency in Hz.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Brightness temperature.
    """
    I *= u.Jy / u.beam
    nu *= u.Hz
    Omega_beam = np.multiply(*beam) * u.arcsec**2 * np.pi / (4 * np.log(2))
    return I.to(
        u.K, equivalencies=u.brightness_temperature(nu, beam_area=Omega_beam)
    ).value


def sigma_to_FWHM(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def FWHM_to_sigma(FWHM):
    return FWHM / np.sqrt(8 * np.log(2))


########## scientific operation ##########
def spatially_integrate(spectrum, source_size, beam):
    FWHM = np.sqrt(beam**2 + source_size**2)
    Omega = get_beam_solid_angle((FWHM, FWHM))
    spectrum *= Omega
    return spectrum


def convolve_Lorentzian(x, y, gamma):
    dx = np.diff(x).mean()
    kernel = Model1DKernel(
        Lorentz1D(amplitude=1.0 / (np.pi * gamma), x_0=0.0, fwhm=2 * gamma / dx),
        x_size=x.size,
    )
    kernel = Model1DKernel(
        Lorentz1D(amplitude=1.0, x_0=0.0, fwhm=2 * gamma / dx), x_size=x.size
    )
    model = convolve_fft(y, kernel, normalize_kernel=True)
    return model


def convolve_Gaussian(x, y, sigma):
    dx = np.diff(x).mean()
    kernel = Gaussian1DKernel(stddev=sigma / dx, x_size=x.size)
    convolved = convolve_fft(y, kernel, normalize_kernel=True)
    return convolved


def convolve_boxcar(x, y, w):
    dx = np.diff(x).mean()
    kernel = Box1DKernel(w / dx)
    convolved = convolve_fft(y, kernel, normalize_kernel=True)
    return convolved


######## MCMC fit tools ########
class Parameter:
    def __init__(self, name, value, bound=None, free=True, label=None, sample=None):
        self.name = name
        self.value = value
        self.bound = bound
        self.free = free
        self.label = label
        self.sample = sample
        if self.free and (self.bound is None or self.label is None):
            raise ValueError("Provide both bound and label for a free parameter.")

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value


class ParameterSet:
    def __init__(self, param_list):
        self.param_list = param_list
        for param in self.param_list:
            setattr(self, param.name, param)
        self.param_name = self.get_param_name()
        self.free_param_name = self.get_free_param_name()
        self.fixed_param_name = self.get_fixed_param_name()
        self.bound = self.get_bound()
        self.p0_init = self.get_initial_value()
        self.p0_fixed = self.get_fixed_value()
        self.label = self.get_free_param_label()

    def get_values(self):
        return [getattr(self, name) for name in self.param_name]

    def get_param_name(self):
        param_name = [param.name for param in self.param_list]
        return param_name

    def get_free_param_name(self):
        free_param_name = [param.name for param in self.param_list if param.free]
        return free_param_name

    def get_fixed_param_name(self):
        fixed_param_name = [param.name for param in self.param_list if not param.free]
        return fixed_param_name

    def get_bound(self):
        bound = [param.bound for param in self.param_list if param.free]
        return bound

    def get_initial_value(self):
        p0_init = [param.value for param in self.param_list if param.free]
        return p0_init

    def get_fixed_value(self):
        p0_fixed = [param.value for param in self.param_list if not param.free]
        return p0_fixed

    def get_free_param_label(self):
        label = [param.label for param in self.param_list if param.free]
        return label


class EmceeHammer:
    def __init__(
        self,
        params=None,
        log_probability=None,
        initial_state=None,
        nwalker=200,
        nstep=500,
        initial_state_blob_mag=1e-4,
    ):

        self.params = params
        self.log_probability = log_probability
        self.initial_state = initial_state
        self.nwalker = nwalker
        self.nstep = nstep

        # set the initial position of each walker
        if self.initial_state is not None:
            self.initial_state = np.atleast_1d(self.initial_state)
            self.ndim = self.initial_state.size
            self.p0 = self.initial_state + initial_state_blob_mag * np.random.randn(
                self.nwalker, self.ndim
            )

    def load_backend(self, filename="test.h5", name="mcmc"):
        self.sampler = emcee.backends.HDFBackend(filename, name=name)

    def run(
        self,
        pool=None,
        progress=True,
        blobs_dtype=None,
        save=True,
        savefilename="test.h5",
        name="mcmc",
    ):

        backend = None

        if save:
            backend = emcee.backends.HDFBackend(savefilename, name=name)
            backend.reset(self.nwalker, self.ndim)

        # set sampler
        self.sampler = emcee.EnsembleSampler(
            self.nwalker,
            self.ndim,
            self.log_probability,
            pool=pool,
            blobs_dtype=blobs_dtype,
            backend=backend,
        )

        # run
        print(
            "starting to run the MCMC sampling with: \n \t initial state:",
            self.initial_state,
            "\n \t number of walkers:",
            self.nwalker,
            "\n \t number of steps:",
            self.nstep,
        )
        self.sampler.run_mcmc(self.p0, int(self.nstep), progress=progress)

        self.full_sample = self.get_sample(thin=1, nburnin=0)
        if self.params is not None:
            for i, name in enumerate(self.params.free_param_name):
                p = getattr(self.params, name)
                setattr(p, "sample", self.full_sample[:, :, i])

    @property
    def chain_length(self):
        return self.sampler.get_chain().shape[0]

    def get_sample(self, thin=1, nburnin=100, flat=False):
        sample = self.sampler.get_chain(thin=thin, discard=nburnin, flat=flat)
        return sample

    def get_flat_sample(self, thin=1, nburnin=100):
        return self.get_sample(thin=thin, nburnin=nburnin, flat=True)

    def get_blobs(self, thin=1, nburnin=100, flat=False):
        blobs = self.sampler.get_blobs(thin=thin, discard=nburnin, flat=flat)
        return blobs

    def get_flat_blobs(self, thin=1, nburnin=100):
        return self.get_blobs(thin=thin, nburnin=nburnin, flat=True)

    def get_log_prob(self, thin=1, nburnin=100, flat=False):
        return self.sampler.get_log_prob(thin=thin, discard=nburnin, flat=flat)

    def _concat_multiple_sample_arrays(*args, axis=0):
        return np.concatenate(args, axis=axis)

    def get_MAP_params(self, thin=1, nburnin=100, include_blobs=False):
        MAP_params = self.get_flat_sample(thin=thin, nburnin=nburnin)[
            np.argmax(self.get_log_prob(thin=thin, nburnin=nburnin, flat=True))
        ]
        return MAP_params

    def get_random_sample_params(self, thin=1, nburnin=100, nsample=50):
        # get flat sample
        full_sample = self.get_flat_sample(thin=thin, nburnin=nburnin)

        # generate random number
        rng = np.random.default_rng(20010714)
        rints = rng.integers(low=0, high=full_sample.shape[0], size=nsample)

        params = np.empty((nsample, full_sample.shape[1]))
        for i, n in enumerate(rints):
            params[i] = full_sample[n]

        return params

    def plot_corner(
        self,
        thin=1,
        nburnin=100,
        include_blobs=False,
        labels=None,
        return_fig=False,
        **kwargs
    ):
        sample = self.get_flat_sample(thin=thin, nburnin=nburnin)
        if include_blobs:
            blobs = self.get_flat_blobs(thin=thin, nburnin=nburnin)
            sample = self._concat_multiple_sample_arrays(sample, blobs, axis=1)

        quantiles = kwargs.pop("quantiles", [0.16, 0.5, 0.84])
        show_titles = kwargs.pop("show_titles", True)
        bins = kwargs.pop("bins", 20)

        fig = corner.corner(
            sample,
            labels=labels,
            quantiles=quantiles,
            show_titles=show_titles,
            bins=bins,
            **kwargs
        )
        if return_fig:
            return fig

    def plot_walker(self, nburnin=100, labels=None, histogram=True, return_fig=False):

        sample = self.get_sample(thin=1, nburnin=0, flat=False).transpose(2, 0, 1)
        # Cycle through the plots.

        fig, axes = plt.subplots(nrows=sample.shape[0], ncols=1, layout="constrained", figsize=(9, 2 * sample.shape[0]))

        for i, s in enumerate(sample):
            ax = axes[i]
            for walker in s.T:
                ax.plot(walker, alpha=0.1, color="k")
            if labels is not None:
                ax.set_ylabel(labels[i])
            if nburnin is not None:
                ax.axvline(nburnin, ls="dotted", color="tab:blue")
            ax.set_xlim(0, s.shape[0])

            # Include the histogram.

            if histogram:
                # fig.set_size_inches(
                #     1.37 * fig.get_figwidth(), fig.get_figheight(), forward=True
                # )
                ax_divider = make_axes_locatable(ax)
                bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
                hist, _ = np.histogram(s[nburnin:].flatten(), bins=bins, density=True)
                bins = np.average([bins[1:], bins[:-1]], axis=0)
                ax1 = ax_divider.append_axes("right", size="35%", pad="2%")
                ax1.fill_betweenx(
                    bins,
                    hist,
                    np.zeros(bins.size),
                    step="mid",
                    color="darkgray",
                    lw=0.0,
                )
                ax1.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
                ax1.set_xlim(0, ax1.get_xlim()[1])
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                ax1.tick_params(which="both", left=0, bottom=0, top=0, right=0)
                ax1.spines["right"].set_visible(False)
                ax1.spines["bottom"].set_visible(False)
                ax1.spines["top"].set_visible(False)

                # get percentile
                q = np.nanpercentile(s[nburnin:].flatten(), [16, 50, 84])
                for val in q:
                    ax1.axhline(val, ls="dashed", color="black")
                text = (
                    labels[i]
                    + r"$ = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                        q[1], np.diff(q)[1], np.diff(q)[0]
                    )
                    if labels is not None
                    else r"${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                        q[1], np.diff(q)[1], np.diff(q)[0]
                    )
                )
                ax1.text(0.5, 1.0, text, transform=ax1.transAxes, ha="center", va="top")

        axes[-1].set_xlabel("Step Number")

        if return_fig:
            return fig


def condition(p, b):
    if (p >= b[0]) and (p <= b[1]):
        return True
    return False


def log_prior(param, bound):
    for p, b in zip(param, bound):
        if not condition(p, b):
            return -np.inf
    return 0.0
