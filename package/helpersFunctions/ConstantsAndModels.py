#_Defines some constants and formulas
kB_kcal = 0.00198657
fMaxBoltzDist = lambda x, T: ( 2 / np.sqrt(np.pi * (T * self.kB_kcal)**3) 
                                            * np.sqrt(x) * np.exp(-x/(self.kB_kcal * T)) ) 
fgaussianModel = lambda x, a, b, c: a / (np.sqrt(2*np.pi) * c) * np.exp(-(x-b)**2 / (2*c**2))

#_Ideal resolution for SHPERES instrument, FWHM of 0.65e-6 eV (single gaussian)
resFuncSPHERES = lambda x: np.exp(-x**2/(2*0.276e-6**2))
 
