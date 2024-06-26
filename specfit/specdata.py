from astroquery.linelists.cdms import CDMS
from astroquery.jplspec import JPLSpec as JPL
import astropy.units as u
import astropy.constants as ac
import numpy as np
from scipy.interpolate import interp1d

h = ac.h.cgs.value
c = ac.c.cgs.value
k_B = ac.k_B.cgs.value


class PartitionFunction:
    def __init__(self, species, T, Q, database=None, ntrans=None):
        self.species = species
        self.T = T
        self.Q = Q
        self.database = database
        self.ntrans = ntrans

        self.function = self._get_function()

    def __call__(self, T):
        if T < np.nanmin(self.T) or T > np.nanmax(self.T):
            print(
                "Warning: Input temperature is smaller or larger than the original partition function data. Will be evaluated by extrapolation."
            )
        val = self.function(T)
        if val.size == 1:
            return float(val)
        else:
            return val

    def _get_function(self):
        T = self.T[~np.isnan(self.Q)]
        Q = self.Q[~np.isnan(self.Q)]
        return interp1d(T, Q, kind="cubic", fill_value="extrapolate")


def wavenumber_to_Kelvin(wavenumber):
    return wavenumber * h * c / k_B


def logint_to_EinsteinA(logint_300, nu0, gup, Elow, Q_300):
    """convert CDMS intensity at 300 K (in nm2 MHz) to Einstein A coeff. (in s-1).
    See https://cdms.astro.uni-koeln.de/classic/predictions/description.html#description for conversion equations

    Parameters
    ----------
    logint_300 : float or ndarray
        log10 CDMS intensity at 300 K in nm2 MHz
    nu0 : float
        line frequency in MHz
    gup : float
        upper state degeneracy
    Elow : float
        lower state energy in cm-1
    Q_300 : float
        partition function at 300 K

    Returns
    -------
    float or ndarray
        Einstein A coeff.
    """
    Elow = wavenumber_to_Kelvin(Elow)
    Eup = Elow + h * nu0 * 1e6 / k_B  # in K
    Smu2 = (
        2.40251e4
        * 10**logint_300
        * Q_300
        / nu0
        / (np.exp(-Elow / 300) - np.exp(-Eup / 300))
    )
    A = 1.16395e-20 * nu0**3 * Smu2 / gup
    return A


class SpectroscopicData:

    def __init__(self) -> None:
        pass

    def _set_quantities(self):
        self.mu = self.table.meta["Molecular Weight"]
        self.Q = self.table.meta["Partition Function"]
        self.nu0 = self.table["Frequency"].value * 1e9
        self.Aul = self.table["A_ul"].value
        self.gup = self.table["g_up"].value
        self.Eup = self.table["E_up"].value

    @staticmethod
    def read_JPL_partition_function(species_table, tag):
        row = species_table[species_table["TAG"] == tag]

        T = np.array(
            species_table.meta["Temperature (K)"][::-1]
        )  # reverse the order to be in increasing temperature order
        Q = 10 ** np.array(
            [float(row[k]) or np.nan for k in row.keys() if "QLOG" in k][::-1]
        )

        return T, Q

    @staticmethod
    def read_CDMS_partition_function(species_table, tag):
        row = species_table[species_table["tag"] == tag]

        T = np.array(
            [float(k.split("(")[-1].split(")")[0]) for k in row.keys() if "lg" in k]
        )
        Q = 10 ** np.array(
            [float(row[k][0]) or np.nan for k in row.keys() if "lg" in k]
        )

        return T, Q

    def query_JPL(self, freq_range=(0.0, np.inf), species_id=1001, nofreqerr=False):
        # frequency range in Hz
        numin, numax = freq_range

        # species
        self.response = JPL.query_lines(
            min_frequency=numin * u.Hz,
            max_frequency=numax * u.Hz,
            molecule=int(species_id),
        )

        # copy for subsequent modification
        self.table = self.response.copy()

        # clean up resulting response
        # 1. remove masked column
        if self.table.mask is not None:
            masked_columns = [
                col for col in self.table.colnames if np.all(self.table.mask[col])
            ]
            self.table.remove_columns(masked_columns)
        # 2. metadata (including partition function) if species is specified
        ## get the specie name which are added to metadata table
        self.species_table = JPL.get_species_table()
        idx = self.species_table["TAG"].tolist().index(int(species_id))
        self.species = self.species_table["NAME"][idx]
        self.table.meta["Species"] = self.species

        # partition function
        T, Q = self.read_JPL_partition_function(
            species_table=self.species_table, tag=int(species_id)
        )
        self.table.meta["Partition Function"] = PartitionFunction(
            species=self.species, T=T, Q=Q, ntrans=self.species_table["NLINE"]
        )

        # 2. remove unnecessary columns
        self.table.remove_columns(["DR", "TAG", "QNFMT"])
        if nofreqerr:
            self.table.remove_column("ERR")
        self.table.add_column(
            col=[self.species] * len(self.table), name="Species", index=0
        )

        # 3. some calculus to make table values useful
        # 3-1. rename frequency (and error) and to GHz
        self.table.rename_column("FREQ", "Frequency")
        self.table["Frequency"] *= 1e-3
        self.table["Frequency"].unit = u.GHz
        self.table["Frequency"].format = "{:.7f}"
        if not nofreqerr:
            self.table.rename_column("ERR", "Frequency Error")
            self.table["Frequency Error"] *= 1e-3
            self.table["Frequency Error"].unit = u.GHz
            self.table["Frequency Error"].format = "{:.7f}"

        # 3-2. A coeff to not log
        self.table.rename_column("LGINT", "A_ul")
        self.table["A_ul"] = logint_to_EinsteinA(
            logint_300=self.table["A_ul"],
            nu0=self.table["Frequency"],
            gup=self.table["GUP"],
            Elow=self.table["ELO"],
            Q_300=self.table.meta["Partition Function"](300)
        )
        self.table["A_ul"].format = "{:.4e}"

        # 3-3. E_low to E_up
        self.table.rename_column("ELO", "E_up")
        self.table["E_up"] = (
            wavenumber_to_Kelvin(self.table["E_up"])
            + h * self.table["Frequency"] * 1e9 / k_B
        )
        self.table["E_up"].unit = "K"
        self.table["E_up"].format = "{:.5f}"

        # 3-4. GUP to g_up
        self.table.rename_column("GUP", "g_up")

        # setup
        self._set_quantities()

    def query_CDMS(
        self, freq_range=(0.0, np.inf), species_id="", use_cached=False, nofreqerr=False
    ):
        # frequency range
        numin, numax = freq_range

        # clear preivous caches
        CDMS.clear_cache()

        # # species
        # if species:
        #     species_table = CDMS.get_species_table(use_cached=use_cached)
        #     tag_list, species_list = (
        #         species_table["tag"].tolist(),
        #         species_table["molecule"].tolist(),
        #     )
        #     # look up tag list
        #     try:
        #         idx = species_list.index(species)
        #     except ValueError:
        #         raise ValueError(
        #             f"No entry found for {species}. Check the species list for existing entries."
        #         )
        #     # set the species with tag (zero-padding for 6 digits)
        #     tag = tag_list[idx]
        #     species = " ".join([str(tag).zfill(6), species])

        self.response = CDMS.query_lines(
            min_frequency=numin * u.Hz,
            max_frequency=numax * u.Hz,
            molecule=str(species_id).zfill(6),
            temperature_for_intensity=0,  # hack to retrieve A coeff instead of logint
        )

        # copy for subsequent modification
        self.table = self.response.copy()

        # clean up resulting response
        # 1. remove masked column
        if self.table.mask is not None:
            masked_columns = [
                col for col in self.table.colnames if np.all(self.table.mask[col])
            ]
            self.table.remove_columns(masked_columns)
        # 2. metadata (including partition function) if species is specified
        ## get the specie name and molweight which are added to metadata table
        self.species_table = CDMS.get_species_table(use_cached=use_cached)
        idx = self.species_table["tag"].tolist().index(int(species_id))
        self.species = self.species_table["molecule"][idx]
        self.molweight = np.unique(self.table["MOLWT"].value)
        if len(self.molweight) > 1:
            raise ValueError(
                "There are multiple values of molecular weight in the table. Check your input or query result."
            )
        self.molweight = self.molweight[0]
        self.table.meta["Species"] = self.species
        self.table.meta["Molecular Weight"] = self.molweight

        # partition function
        T, Q = self.read_CDMS_partition_function(
            species_table=species_table, tag=int(self.tag)
        )
        self.table.meta["Partition Function"] = PartitionFunction(
            species=self.species, T=T, Q=Q, ntrans=species_table["#lines"]
        )

        # 2. remove unnecessary columns
        self.table.remove_columns(["DR", "TAG", "QNFMT", "MOLWT", "Lab"])
        if nofreqerr:
            self.table.remove_column("ERR")
        self.table.add_column(col=self.table["name"], name="Species", index=0)
        self.table.remove_column("name")

        # 3. some calculus to make table values useful
        # 3-1. rename frequency (and error) and to GHz
        self.table.rename_column("FREQ", "Frequency")
        self.table["Frequency"] *= 1e-3
        self.table["Frequency"].unit = u.GHz
        self.table["Frequency"].format = "{:.7f}"
        if not nofreqerr:
            self.table.rename_column("ERR", "Frequency Error")
            self.table["Frequency Error"] *= 1e-3
            self.table["Frequency Error"].unit = u.GHz
            self.table["Frequency Error"].format = "{:.7f}"

        # 3-2. A coeff to not log
        self.table.rename_column("LGAIJ", "A_ul")
        self.table["A_ul"] = 10 ** self.table["A_ul"]
        self.table["A_ul"].format = "{:.4e}"

        # 3-3. E_low to E_up
        self.table.rename_column("ELO", "E_up")
        self.table["E_up"] = (
            wavenumber_to_Kelvin(self.table["E_up"])
            + h * self.table["Frequency"] * 1e9 / k_B
        )
        self.table["E_up"].unit = "K"
        self.table["E_up"].format = "{:.5f}"

        # 3-4. GUP to g_up
        self.table.rename_column("GUP", "g_up")

        # setup
        self._set_quantities()
