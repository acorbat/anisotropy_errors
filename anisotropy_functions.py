# Define functions for anisotropy and crossed intensities calculated from
# parameters or data


def anisotropy_from_intensity(Ipar, Iper):
    """
    Calculate anisotropy from crossed intensities values.
    """
    return (Ipar - Iper)/(Ipar + 2*Iper)


def anisotropy_from_monomer(m, Am, Ad, b):
    """
    Calculate anisotropy from monomer fraction, monomer and dimer anisotropy
    and brightness relation.
    """
    return (Ad*b+(Am-b*Ad)*m)/(b+(1-b)*m)


def monomer_from_anisotropy(a, Am, Ad, b):
    """
    Calculate monomer fraction from anisotropy, monomer and dimer anisotropy
    and brightness relation.
    """
    return (b * a - Ad*b) / ((Am - b * Ad) - (1-b) * a)



def total_fluorescence_from_intensity(I_par, I_per):
    """
    Calculate total fluorescence from crossed intensities.
    """
    return I_par + 2 * I_per


def total_fluorescence_from_monomer(m, b, c_b_1):
    """
    Calculate fluorescence from monomer fraction, brightness relation and
    monomer brightness.
    """
    return c_b_1*(m+(1-m)*b)


def intensity_parallel_from_monomer(M, A_1, A_2, b, c_b_1):
    """
    Calculate parallel intesity from monomer fraction, monomer and dimer
    anisotropy, brightness relation and monomer brightness.
    """
    return (2/3)*(((A_1+0.5)-(A_2+0.5)*b) *M + (A_2+0.5)*b)*c_b_1


def intensity_perpendicular_from_monomer(M, A_1, A_2, b, c_b_1):
    """
    Calculate perpendicular intesity from monomer fraction, monomer and dimer
    anisotropy, brightness relation and monomer brightness.
    """
    return (1/3)*(((1-A_1) - (1-A_2)*b) *M + (1-A_2)*b)*c_b_1


def intensity_parallel_from_monomer_n(M, A_1, A_2, b):
    """
    Calculate parallel intesity nominator from monomer fraction, 
    monomer and dimer anisotropy, brightness relation and monomer brightness.
    """
    return (2)*(((A_1+0.5)-(A_2+0.5)*b) *M + (A_2+0.5)*b)


def intensity_perpendicular_from_monomer_n(M, A_1, A_2, b):
    """
    Calculate perpendicular intesity nominator from monomer fraction,
    monomer and dimer anisotropy, brightness relation and monomer brightness.
    """
    return (((1-A_1) - (1-A_2)*b) *M + (1-A_2)*b)


def intensity_parallel_from_monomer_r(M, A_1, A_2, b):
    """
    Calculate parallel intesity normalized by total fluorescence from monomer
    fraction, monomer and dimer anisotropy, brightness relation and monomer
    brightness.
    """
    num = intensity_parallel_from_monomer_n(M, A_1, A_2, b)
    denom = intensity_parallel_from_monomer_n(M, A_1, A_2, b)
    denom += 2 * intensity_perpendicular_from_monomer_n(M, A_1, A_2, b)
    return num / denom


def intensity_perpendicular_from_monomer_r(M, A_1, A_2, b):
    """
    Calculate perpendicular intesity normalized by total fluorescence from
    monomer fraction, monomer and dimer anisotropy, brightness relation and
    monomer brightness.
    """
    num = intensity_parallel_from_monomer_n(M, A_1, A_2, b)
    denom = intensity_parallel_from_monomer_n(M, A_1, A_2, b)
    denom += 2 * intensity_perpendicular_from_monomer_n(M, A_1, A_2, b)
    return num / denom


def intensity_parallel_from_anisotropy(Aniso, Fluo):
    """
    Calculate parallel intensity from anisotropy and total fluorescence.
    """
    return (2/3)*Fluo*(Aniso+0.5)


def intensity_perpendicular_from_anisotropy(Aniso, Fluo):
    """
    Calculate perpendicular intensity from anisotropy and total fluorescence.
    """
    return (1/3)*Fluo*(1-Aniso)