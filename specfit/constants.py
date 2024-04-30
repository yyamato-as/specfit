import astropy.constants as ac
import astropy.units as u

# constants
m_p = ac.m_p.cgs.value
mH2 = 2.016 * ac.u.to(u.g).value
h = ac.h.cgs.value
k_B = ac.k_B.cgs.value
c = ac.c.cgs.value
ckms = ac.c.to(u.km / u.s).value
