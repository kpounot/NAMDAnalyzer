import numpy as np

#_Defines some constants and formulas
kB_kcal = 0.00198657
fMaxBoltzDist = lambda x, T: ( 2 / np.sqrt(np.pi * (T * kB_kcal)**3) 
                                            * np.sqrt(x) * np.exp(-x/(kB_kcal * T)) ) 
fgaussianModel = lambda x, a, b, c: a / (np.sqrt(2*np.pi) * c) * np.exp(-(x-b)**2 / (2*c**2))

#_Ideal resolution for SHPERES instrument, FWHM of 0.66e-6 eV (single gaussian)
FTresFuncSPHERES    = lambda x: np.exp(-x**2 * np.pi**2 * 0.66e-6 / 2) * np.sqrt(np.pi * 0.66e-6)


FTvoigtSPHERES      = lambda x: ( 0.05 * np.pi * np.exp(-x*0.28e-6**2) + 0.95 
                                                                       * np.exp(-x**2*np.pi**2*0.28e6**2 / 2) 
                                                                       * np.sqrt(np.pi * 0.28e-6) ) 




def getRandomVec(q):
    """ Computes a random vector components of given magnitude q. """
 
    qi = np.random.rand(3) - 0.5

    qi *= q / np.sqrt(np.sum(qi**2))

    return qi.astype('f')

