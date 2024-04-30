import numpy as np
import h5py
import matplotlib.pyplot as plt
from .constants import h, c, k_B, ckms, mH2

####################### Spectrum Class ###########################
class Spectrum:
    def __init__(
        self,
        filename=None,
        nu=None,
        v=None,
        channel=None,
        I=None,
        dI=None,
        unit=None,
        nu0=None,
        beam=None,
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        filename : str, optional
            hdf5 filename which contains the spectrum data, by default None
        nu : array_like, optional
            frequency axis in Hz, by default None
        v : array_like, optional
            velocity axis in km/s, by default None
        channel : array_like, optional
            channel index, by default None
        I : array_like, optional
            spectrum in an arbitrary unit, by default None
        dI : array_like, optional
            uncertainty of the spectrum in the same unit as that of spectrum, by default None
        nu0 : float, optional
            rest frequency (or representative frequency of the spectrum) in Hz, by default None
        beam : tuple, optional
            beam in (bmaj, bmin, bpa) in arcsec and deg, by default None
        """
        self.filename = filename
        self._nu = nu
        self._v = v
        self._channel = channel
        self._I = I
        self._dI = dI
        self._unit = unit
        self._nu0 = nu0
        self._beam = beam
        self._kwargs = kwargs
        for key in kwargs:
            setattr(self, "_" + key, kwargs[key])

        self.restore()

    def restore(self):
        if self.filename is not None:
            with h5py.File(self.filename, "r") as f:
                # spectrum data
                data = f["data"]
                self.v = data["velocity"][...] if "velocity" in data.keys() else None
                self.nu = data["frequency"][...] if "frequency" in data.keys() else None
                self.channel = (
                    data["channel"][...] if "channel" in data.keys() else None
                )
                self.I = data["spectrum"][...] if "spectrum" in data.keys() else None
                self.dI = data["scatter"][...] if "scatter" in data.keys() else None

                # metadata
                self.unit = data["spectrum"].attrs["unit"]
                md = f["metadata"]
                for key in md.attrs.keys():
                    setattr(self, key, md.attrs[key])
                # self.nu0 = md.attrs["nu0"]
                # self.beam = md.attrs["beam"]
        else:
            self.v = (
                (1 - self._nu / self._nu0) * ckms
                if (self._nu is not None) and (self._nu0 is not None)
                else self._v
            )
            self.nu = (
                (1 - self._v / ckms) * self._nu0
                if (self._v is not None) and (self._nu0 is not None)
                else self._nu
            )
            self.channel = (
                self._channel if self._channel is not None else np.arange(self._I.size)
            )
            self.I = self._I
            self.dI = self._dI
            self.unit = self._unit
            self.nu0 = self._nu0
            self.beam = self._beam
            for key in self._kwargs.keys():
                setattr(self, key, getattr(self, "_" + key))

    def _set_velocity_axis(self, nu0=None):
        if nu0 is None:
            pass
        else:
            self.v = (1 - self.nu / nu0) * ckms

    def split(self, nu0=None, vrange=None, nurange=None, vsys=0.0):
        if vrange is not None:
            vstart, vend = vrange
            vstart += vsys
            vend += vsys

            if nu0 is not None and isinstance(nu0, (list, np.ndarray)):
                mask = np.zeros(self.nu.shape, dtype=bool)
                for nu in nu0:
                    v = (1 - self.nu / nu) * ckms
                    m = (v >= vstart) & (v <= vend)
                    mask = mask | m
                self.nu0 = nu0[int(len(nu0) / 2)]
                self.v = (1 - self.nu / self.nu0) * ckms
            else:
                self.nu0 = nu0 if nu0 is not None else self.nu0
                self.v = (1 - self.nu / self.nu0) * ckms
                mask = (self.v >= vstart) & (self.v <= vend)

        elif nurange is not None:
            nustart, nuend = nurange
            mask = (self.nu >= nustart) & (self.nu <= nuend)

        else:
            mask = np.ones_like(self.I, dtype=bool)

        self.v = self.v[mask] if self.v is not None else self.v
        self.nu = self.nu[mask] if self.nu is not None else self.nu
        self.channel = self.channel[mask] if self.channel is not None else self.channel
        self.I = self.I[mask] if self.I is not None else self.I
        self.dI = self.dI[mask] if self.dI is not None else self.dI

    def plot(
        self,
        ax=None,
        axis="velocity",
        scaling=1.0,
        error_scaling=1.0,
        nu0=None,
        vrange=None,
        nurange=None,
        vsys=0.0,
        indicate_loc=False,
        **errorbar_kwargs,
    ):
        self.restore()
        self.split(nu0=nu0, vrange=vrange, nurange=nurange, vsys=vsys)
        self._set_velocity_axis(nu0=np.nanmedian(nu0) if nu0 is not None else self.nu0)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if "freq" in axis:
            x = self.nu * 1e-9
            if indicate_loc:
                dnu = (1.0 - vsys / ckms) * self.nu0
                x0 = (np.array(nu0) + dnu) * 1e-9
        elif "chan" in axis:
            x = self.channel
            if indicate_loc:
                raise NotImplementedError
        else:
            x = self.v
            if indicate_loc:
                dv = (1.0 - np.array(nu0) / self.nu0) * ckms
                x0 = vsys + dv

        y = self.I * scaling
        yerr = self.dI * scaling * error_scaling if self.dI is not None else self.dI

        self.errorbar = ax.errorbar(x, y, yerr=yerr, **errorbar_kwargs)
        ax.set(xlim=(x.min(), x.max()))

        if indicate_loc:
            for loc in x0:
                ax.axvline(x=loc, color="grey", lw=0.8, alpha=0.5, ls="dotted")

        return fig, ax

    def save_to_hdf5(self, savefilename):
        self.quantities_to_save = ["v", "nu", "channel", "I", "dI"]
        self.quantities_names = [
            "velocity",
            "frequency",
            "channel",
            "spectrum",
            "scatter",
        ]
        self.quantities_units = ["km/s", "Hz", "", self.unit, self.unit]
        with h5py.File(savefilename, "w") as f:
            g = f.create_group("data")
            for quantity, name, unit in zip(
                self.quantities_to_save, self.quantities_names, self.quantities_units
            ):
                q = getattr(self, quantity)
                if q is not None:
                    dataset = g.create_dataset(name, data=q)
                    dataset.attrs["unit"] = unit if unit is not None else ""
                else:
                    self.quantities_to_save.remove(quantity)
                    self.quantities_names.remove(name)
                    self.quantities_units.remove(unit)

            # dataset = g.create_dataset("velocity", data=self.v)
            # dataset.attrs["unit"] = "km/s"
            # dataset = g.create_dataset("frequency", data=self.nu)
            # dataset.attrs["unit"] = "Hz"
            # dataset = g.create_dataset("channel", data=self.channel)
            # dataset.attrs["unit"] = ""
            # dataset = g.create_dataset("spectrum", data=self.I)
            # dataset.attrs["unit"] = self.unit
            # dataset = g.create_dataset("scatter", data=self.dI)
            # dataset.attrs["unit"] = self.unit

            g_meta = f.create_group("metadata")
            if self.beam is not None:
                g_meta.attrs["beam"] = self.beam
            if self.nu0 is not None:
                g_meta.attrs["nu0"] = self.nu0
            for key in self._kwargs.keys():
                if getattr(self, key) is not None:
                    g_meta.attrs[key] = getattr(self, key)


##################### functions for spectral modeling ######################
def eta(source_size, beam):
    """calculate the beam filling factor. The unit of source_size and beam should be the same.

    Parameters
    ----------
    source_size : float or ndarray
        source size
    beam : float or ndarray
        beam size

    Returns
    -------
    float or ndarray
        beam filling factor, eta
    """
    return source_size**2 / (beam**2 + source_size**2)


def calc_dust_optical_depth(nu, N_H2, kappa, beta, g2d=100.0, nu_ref=230e9):
    """calculate dust optical depth spectrum

    Parameters
    ----------
    nu : ndarray
        frequencies in Hz at which you want to calculate the optical depth of the dust
    N_H2 : float
        column density of H2 molecule in cm-2
    kappa : float
        dust opacity in cm2 g-1 at the reference frequency given by `nu_ref`
    beta : float
        spectral index, i.e., power-law index of the frequency dependence of the dust optical depth
    g2d : float, optional
        gas-to-dust mass ratio, by default 100
    nu_ref : float, optional
        reference frequency for the dust opacity in Hz, by default 230e9

    Returns
    -------
    ndarray
        dust optical depth spectrum
    """
    tau_d = (N_H2 * kappa * mH2 / g2d) * (nu / nu_ref) ** beta
    return tau_d


def line_profile_function(nu, nu0, sigma):
    """calculate line profile function. The returned ndarray shape will be (nu.size, nu0.size).

    Parameters
    ----------
    nu : ndarray
        frequencies in Hz at which you want to calculate phi
    nu0 : ndarray
        array of the central frequencies in Hz
    sigma : ndarray
        array of the frequency line width in Hz

    Returns
    -------
    ndarray with a shape of (nu.size, nu0.size)
        line profile function of each component
    """
    return (
        1.0
        / ((2 * np.pi) ** 0.5 * sigma[None, :])
        * np.exp(-0.5 * (nu[:, None] - nu0[None, :]) ** 2 / sigma[None, :] ** 2)
    )


def calc_line_optical_depth(nu, nu0, sigma, logN, Aul, gu, gl, xu, xl):
    phi = line_profile_function(nu, nu0, sigma)
    tau = c**2 / (8 * np.pi * nu[:, None] ** 2) * Aul[None, :] * 10**logN
    tau *= (xl[None, :] * (gu[None, :] / gl[None, :]) - xu[None, :]) * phi
    return np.sum(tau, axis=1)


def calc_line_optical_depth(nu, nu0, sigma, Tex, logN, Aul, gu, Eu, Q):
    """calculate line optical depth spectrum. The returned ndarray shape will be (nu.size, nu0.size).

    Parameters
    ----------
    nu : ndarray
        Frequencies in Hz at which you want to calculate tau
    nu0 : ndarray
        array of the central frequencies in Hz
    sigma : ndarray
        array of the frequency line width in Hz
    Tex : float or ndarray
        Excitation temperature in K of the molecule. Different Tex for each transition with ndarray is acceptable.
    logN : float or ndarray
        log10 of Column density in cm-2 of the molecule. Different Ntot for each transition with ndarray is acceptable.
    Aul : ndarray
        Einstein A coefficients in s-1 of the transitions
    gu : ndarray
        Upper state degeneracies of the transitions
    Eu : ndarray
        upper state enegies in K of the transitions
    Q : function
        partition function

    Returns
    -------
    1d array
        line optical depth profile
    """
    phi = line_profile_function(nu, nu0, sigma)
    tau_l = c**2 / (8 * np.pi * nu[:, None] ** 2) * Aul[None, :] * 10**logN
    tau_l *= gu[None, :] * np.exp(-Eu[None, :] / Tex) / Q(Tex)
    # tau_l *= (1 - np.exp(- h * nu0[None, :] / (k_B * Tex))) * phi
    tau_l *= (np.exp(h * nu0[None, :] / (k_B * Tex)) - 1) * phi
    return np.sum(tau_l, axis=1)


def Inu(
    nu,
    nu0,
    sigma,
    Tex,
    logN,
    size,
    beam,
    Aul,
    gu,
    El,
    Q,
    Tdust,
    N_H2,
    kappa,
    beta,
    g2d=100.0,
    nu_ref=230e9,
):
    """Intensity

    Parameters
    ----------
    nu : ndarray
        Frequencies in Hz at which you want to calculate tau
    nu0 : ndarray
        array of the central frequencies in Hz
    sigma : ndarray
        array of the frequency line width in Hz
    Tex : float or ndarray
        Excitation temperature in K of the molecule. Different Tex for each transition with ndarray is acceptable.
    logN : float or ndarray
        Column density in cm-2 of the molecule. Different Ntot for each transition with ndarray is acceptable.
    Aul : ndarray
        Einstein A coefficients in s-1 of the transitions
    gu : ndarray
        Upper state degeneracies of the transitions
    El : ndarray
        Lower state enegies in K of the transitions
    Q : function
        partition function
    Tdust : float
        dust temperature in K
    N_H2 : float
        column density of H2 molecule in cm-2
    kappa : float
        dust opacity in cm2 g-1 at the reference frequency given by `nu_ref`
    beta : float
        spectral index, i.e., power-law index of the frequency dependence of the dust optical depth
    g2d : float, optional
        gas-to-dust mass ratio, by default 100
    nu_ref : float, optional
        reference frequency for the dust opacity in Hz, by default 230e9

    Returns
    -------
    1d array
        source function
    """
    # optical depths
    tau_d = calc_dust_optical_depth(nu, N_H2, kappa, beta, g2d=g2d, nu_ref=nu_ref)
    tau_l = calc_line_optical_depth(nu, nu0, sigma, Tex, logN, Aul, gu, El, Q)
    tau = tau_l + tau_d

    # source function
    Snu = (tau_l * Bnu(nu, Tex) + tau_d * Bnu(nu, Tdust)) / (tau_l + tau_d)

    # return peak intensity
    return eta(size, beam) * (Snu - Bnu_CMB(nu)) * (1 - np.exp(-tau))


def Bnu(nu, T):
    """Planck function for blackbody radiation

    Parameters
    ----------
    nu : float or ndarray
        frequencies in Hz at which you wnat to calculate the blackbody
    T : float
        temperature in K

    Returns
    -------
    float or ndarray
        blackbody spectrum
    """
    return (
        2 * h * nu**3 / c**2 / (np.exp(h * nu / (k_B * T)) - 1.0) * 1e23
    )  # in Jy / sr


def Jnu(nu, T):
    """R-J brightness temperature for blackbody radiation

    Parameters
    ----------
    nu : float or ndarray
        frequencies in Hz at which you want to calculate the blackbody
    T : float
        temperature in K

    Returns
    -------
    float or ndarray
        R-J brightness temperature spectrum
    """
    return h * nu / k_B / (np.exp(h * nu / (k_B * T)) - 1.0)  # in K


def Bnu_CMB(nu, T_CMB=2.73):
    """Planck function for CMB radiation"""
    return Bnu(nu, T_CMB)


def Jnu_CMB(nu, T_CMB=2.73):
    """R-J brightness temperature for CMB radiation"""
    return Jnu(nu, T_CMB)
