import numpy as np

# Defines some constants and formulas
kB_kcal = 0.00198657
fMaxBoltzDist = lambda x, T: (2 / np.sqrt(np.pi * (T * kB_kcal)**3)
                              * np.sqrt(x) * np.exp(-x / (kB_kcal * T)))
fgaussianModel = lambda x, a, b, c: (a / (np.sqrt(2 * np.pi) * c)
                                     * np.exp(-(x - b)**2 / (2 * c**2)))

# Ideal resolution for SPHERES instrument, FWHM of 0.66e-6 eV (single gaussian)
FTresFuncSPHERES  = lambda x: (np.exp(-x**2 * np.pi**2 
                              * (0.66e-6 / 4.135e-15)**2))

FTresFuncGaussian = lambda x, width: np.exp(-x**2 * np.pi**2 * width**2)


def getPseudoVoigtFTFunc(a, width):
    """Fourier transform of a pseudo-Voigt profile

    This returns a lambda function that takes time vector as input.

    Parameters
    ----------
    a : float
        A float between 0 and 1 giving the relative contribution of Lorentzian
        and Gaussian.
    width : float
        A float gibing the width of the lineshapes in eV.

    """
    out = lambda x: (
            a * np.exp(-x**2 * np.pi**2 * (width / 4.135e-15)**2)
            + (1 - a) * np.exp(-x * (width / 4.135e-15))
    )

    return out


def getRandomVec(q):
    """ Computes a random vector components of given magnitude q. """

    qi = np.random.rand(3) - 0.5

    qi *= q / np.sqrt(np.sum(qi**2))

    return qi.astype('f')
