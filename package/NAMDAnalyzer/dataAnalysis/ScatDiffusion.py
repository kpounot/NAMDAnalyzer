"""

Classes
^^^^^^^

"""

import sys
import numpy as np

from collections import defaultdict

from scipy.spatial import ConvexHull, Voronoi, Delaunay
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from scipy.optimize import root


from NAMDAnalyzer.helpersFunctions import ConstantsAndModels as CM 
from NAMDAnalyzer.dataAnalysis.RadialDensity import COMRadialNumberDensity

try:
    from NAMDAnalyzer.lib.pylibFuncs import py_compIntScatFunc
except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
            + "Please compile it before using it.\n")


from NAMDAnalyzer.dataParsers.HydroproReader import HydroproReader



class ScatDiffusion(HydroproReader):
    """ This class defines methods to compute corrected diffusion coefficients.
        Starting from coefficients Dt0, for translational diffusion and Dr0 for rotational diffusion
        in the dilute limit obtained with HYDROPRO for example, corrections on the volume fraction
        and coefficient themselves can be performed to obtain the apparent diffusion coefficient 
        for crowded solution that can be used to compare with experimental scattering data.
        
        :arg dataset:       a self.dataset class instance containing trajectories data 
        :arg hydroproFile:  output .res file from hydropro containing computes parameters 

    """


    def __init__(self, dataset, hydroproFile=None):

        if hydroproFile:
            super().__init__(hydroproFile)

        self.dataset = dataset

        self.effPhi = None

        self.roots  = None




#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compEffVolFrac(self, massP, solventVol=1, compSpecVol=False, selection='protein', frame=0):
        """ Computes the hydrodynamic radius based on translational diffusion coefficient in the dilute limit
            and uses the relation, effPhi = phi * (Rh/R)**3 to get the effective volume fraction. 
            
            :arg massP:       mass of protein used in the experiment (in grams)
            :arg solventVol:  volume of solvent used (in cm^3), to compute initial volume fraction 
            :arg compSpecVol: if True, computes the specific volume based in protein sequence instead
                              of using the one from HYDROPRO.
            :arg selection:   if compSpecVol is True, use the given selection to compute the specific volume
            :arg frame:       if compSpecVol is True, use the given frame to compute the specific volume
            
        """

        if compSpecVol:
            vol     = self.dataset.getProtVolume(selection, frame) * 1e-24
            specVol = self.dataset.getSpecVolume(selection, frame)
        else:
            vol     = self.params.vol
            specVol = self.params.specVol


        R = ( 3/(4*np.pi) * vol )**(1/3)

        phi = specVol * massP / (solventVol + specVol * massP)

        self.effPhi = phi * (self.params.rh / R)**3




    def _compCorrectedDt(self):
        """ Computes the translational diffusion coefficient in crowded solution based on protein's 
            effective volume fraction phi and translational diffusion coefficient Ds0 in the dilute limit. 
        
            See M.Tokuyama, I.Oppenheim, J. Korean Phys. 28, (1995) 

        """

        b = np.sqrt( 9 / 8 * self.effPhi )
        c = 11 / 16 * self.effPhi
        
        H = 2*b**2 / (1 - b) - c / (1 + 2*c) - b*c*(2+c) / (1+c)*(1-b+c)

        self.Dt = self.params.dt0 / ( 1 + H )




    def _compCorrectedDr(self):
        """ Computes the rotational diffusion coefficient in crowded solution based on protein's 
            effective volume fraction phi and rotational diffusion coefficient in the dilute limit. 

        """

        self.Dr = self.params.dr0 * (1 - 0.631*self.effPhi - 0.726*self.effPhi**2)






    def compAppDiffCoeff(self, qVals, maxN=100, density_dr=1, maxDensityR=60, frame=-1, minMethod='lm'):
        """ Computes apparent diffusion coefficient D based on corrected diffusion coefficients Dt and Dr. 
        
            As Dt and Dr are in [cm] and [s] units, the obtained result is given in cm^2/s 

            :arg qVals:         a list of scattering vector amplitude values to compute D (in [angström])
            :arg maxN:          maximum number of spherical bessel function to be used  
            :arg density_dr:    increment value for the radius r used to compute the radial density from the 
                                protein center-of-mass
            :arg maxDensityR:   maximum radius to be used to compute the radial density
            :arg frame:         frame to be used on the loaded trajectory to compute the radial density 

        """ 


        if self.effPhi is None:
            print("The effective volume fraction was not computed.\n"
                    + "Use self.compEffVolFrac() method before calling this method.\n")
            return


        #_Pre-process q-values and diffusion coefficients
        qVals = np.array( qVals ).astype('f') * 1e8 #_Conversion to cm^(-1)

        self._compCorrectedDt()
        self._compCorrectedDr()

        #_Computes the density
        density = COMRadialNumberDensity(self.dataset, 'protH', dr=density_dr, 
                                         maxR=maxDensityR, frame=frame)
        density.compDensity()
        density = ( density.radii, density.density )


        solRoot = root(self._fitFuncAppDiffCoeff, x0=self.Dt, args=(qVals, density, maxN), method=minMethod) 

        self.roots = solRoot
                





    def _fitFuncAppDiffCoeff(self, D, qVals, density, maxN=100):
        """ Fitting function to compute apparent diffusion coefficient. """


        dist, density = density
        dist = dist[:,np.newaxis] * 1e-8
        density = density[:,np.newaxis] 

        res = np.zeros(qVals.size)
        for n in range(maxN):
            Bn = (2*n+1) * np.sum( density * spherical_jn(n, qVals*dist)**2, axis=0 )
            res += ( Bn * (self.Dr * n*(n+1) + (self.Dt - D) * qVals**2) 
                            / (self.Dr * n*(n+1) + (self.Dt + D) * qVals**2)**2 )


        return res





    def compDt(self, qVals, D, Dr, maxN=100, density_dr=1, maxDensityR=60, frame=-1, 
                init_Dt=None, minMethod='lm'):
        """ Computes coefficients Dt based on the apparent diffusion coefficient D and theoretical Dr. 
        
            Dt is given in [cm^2/s]. 

            :arg qVals:         a list of scattering vector amplitude values to compute D (in [angström])
            :arg D:             the apparent self-diffusion coefficient from neutron scattering
            :arg Dr:            the rotational diffusion coefficient from neutron scattering
            :arg maxN:          maximum number of spherical bessel function to be used  
            :arg density_dr:    increment value for the radius r used to compute the radial density from the 
                                protein center-of-mass
            :arg maxDensityR:   maximum radius to be used to compute the radial density
            :arg frame:         frame to be used on the loaded trajectory to compute the radial density 
            :arg init_Dt:       initial guess for Dt
            :arg Dr_scaling:    scaling factor for Dr


        """ 


        if self.effPhi is None:
            print("The effective volume fraction was not computed.\n"
                    + "Use self.compEffVolFrac() method before calling this method.\n")
            return


        if init_Dt is None:
            self._compCorrectedDt()
            init_Dt = self.Dt


        #_Pre-process q-values and diffusion coefficients
        qVals = np.array( qVals ).astype('f') * 1e8 #_Conversion to cm^(-1)


        #_Computes the density
        density = COMRadialNumberDensity(self.dataset, 'protH', dr=density_dr, 
                                         maxR=maxDensityR, frame=frame)
        density.compDensity()
        density = ( density.radii, density.density )


        solRoot = root(self._fitFunc_Dt, x0=init_Dt, 
                       args=(D, Dr, qVals, density, maxN), method=minMethod) 

        self.Dt_estimate = solRoot.x[0]
                



    def _fitFunc_Dt(self, Dt, D, Dr, qVals, density, maxN=100):
        """ Fitting function to compute apparent diffusion coefficient. """

        dist, density = density
        dist = dist[:,np.newaxis] * 1e-8
        density = density[:,np.newaxis] 

        res = np.zeros(qVals.size)
        for n in range(maxN):
            Bn = (2*n+1) * np.sum( density * spherical_jn(n, qVals*dist)**2, axis=0 )
            res += ( Bn * (Dr * n*(n+1) + (Dt - D) * qVals**2) 
                            / (Dr * n*(n+1) + (Dt + D) * qVals**2)**2 )


        return res







#---------------------------------------------
#_Other physical parameters (D2O density and viscosity)
#---------------------------------------------
    @classmethod
    def getViscosityD2O(self, T, dT=None):
        """ Calculates the viscosity of D2O in units of [Pa*s] for given temperature in units of [K]
            Dependence of viscosity on temperature was determined by Cho et al. (1999)
            "Thermal offset viscosities of liquid H2O, D2O and T2O", J.Phys. Chem. B
            103(11):1991-1994

            eta(T) is valid from 280k up to 400K 

        """

        T = float(T)

        if dT:
            dT = float(dT)

        C  = 885.60402
        a  = 2.799e-3
        b  = -1.6342e-5
        c  = 2.9067e-8
        g  = 1.55255
        T0 = 231.832
        t = T - T0
        
        eta  = 1e-3 * C * ( t + a * t**2 + b * t**3 + c * t**4 )**(-g);
        detadT = ( -1e-3 * C * g * ( 1 + 2 * a * t + 3 * b * t**2 + 4 * c * t**3 ) *
                                                    ( t + a * t**2 + b * t**3 + c * t**4 )**(-1-g) )  
        if dT:
            deta = abs( detadT ) * dT
            return eta, deta

        else:
            return eta



    @classmethod
    def getDensityD2O(self, T):
        """ Calculates the density of D2O in units of [g/cm^3] for given temperature T in [K]
            
            References:

                - Handbook of Chemistry and Physics, 73rd Edition, 
                  Lide, D.R. Ed.; CRC Press: Boca Raton 1992; Chapter 6, pg. 13
                  http://physchem.uni-graz.ac.at/sm/Service/Water/D2Odens.htm 

        """

        T = float(T)

        table = np.array( [ [3.82, 1.1053],
                            [5, 1.1055],
                            [10, 1.1057],
                            [15, 1.1056],
                            [20, 1.1050],
                            [25, 1.1044],
                            [30, 1.1034],
                            [35, 1.1019],
                            [40, 1.1001],
                            [45, 1.0979],
                            [50, 1.0957],
                            [55, 1.0931],
                            [60, 1.0905],
                            [65, 1.0875],
                            [70, 1.0847],
                            [75, 1.0815],
                            [80, 1.0783],
                            [85, 1.0748],
                            [90, 1.0712],
                            [95, 1.0673],
                            [100, 1.0635], 
                            [105, 1.0598] ] )


        table[:,0] = table[:,0] + 273.15
        densityD2O = interp1d(table[:,0], table[:,1])

        return densityD2O(T)


