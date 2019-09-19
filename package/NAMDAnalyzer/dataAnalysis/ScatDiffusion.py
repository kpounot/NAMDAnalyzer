import sys
import numpy as np

from scipy.spatial import ConvexHull, Voronoi
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from scipy.optimize import root


from NAMDAnalyzer.helpersFunctions import ConstantsAndModels as CM 

from NAMDAnalyzer.lib.pylibFuncs import py_compIntScatFunc
from NAMDAnalyzer.dataParsers.HydroproReader import HydroproReader



class ScatDiffusion(HydroproReader):

    def __init__(self, dataset, hydroproFile=None):
        """ This class defines methods to compute corrected diffusion coefficients.
            Starting from coefficients Dt0, for translational diffusion and Dr0 for rotational diffusion
            in the dilute limit obtained with HYDROPRO for example, corrections on the volume fraction
            and coefficient themselves can be performed to obtain the apparent diffusion coefficient 
            for crowded solution that can be used to compare with experimental scattering data.
            
            Input:  dataset         -> a self.dataset class instance containing trajectories data 
                    hydroproFile    -> output .res file from hydropro containing computes parameters """


        if hydroproFile:
            super().__init__(hydroproFile)

        self.dataset = dataset

        self.effPhi = None

        self.roots  = None


#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compEffVolFrac(self, massP, solventVol=1):
        """ Computes the hydrodynamic radius based on translational diffusion coefficient in the dilute limit
            and uses the relation, effPhi = phi * (Rh/R)**3 to get the effective volume fraction. 
            
            Input:  massP       -> mass of protein used in the experiment (in grams)
                    solventVol  -> volume of solvent used (in cm^3), to compute initial volume fraction """


        R = ( 3/(4*np.pi) * self.params.vol )**(1/3)

        phi = self.params.specVol * massP / (solventVol + self.params.specVol * massP)

        self.effPhi = phi * (self.params.rh / R)**3




    def compCorrectedDt(self):
        """ Computes the translational diffusion coefficient in crowded solution based on protein's 
            effective volume fraction phi and translational diffusion coefficient Ds0 in the dilute limit. 
        
            See M.Tokuyama, I.Oppenheim, J. Korean Phys. 28, (1995) """

        b = np.sqrt( 9 / 8 * self.effPhi )
        c = 11 / 16 * self.effPhi
        
        H = 2*b**2 / (1 - b) - c / (1 + 2*c) - b*c*(2+c) / (1+c)*(1-b+c)

        self.Dt = self.params.dt0 / ( 1 + H )




    def compCorrectedDs(self):
        """ Computes the rotational diffusion coefficient in crowded solution based on protein's 
            effective volume fraction phi and rotational diffusion coefficient in the dilute limit. """

        self.Dr = self.params.dr0 * (1 - 1.3 * self.effPhi**2) 






    def compAppDiffCoeff(self, qVals, maxN=100, density_dr=1, maxDensityR=60, frame=-1):
        """ Computes apparent diffusion coefficient based on corrected diffusion coefficients Dt and Dr. 
        
            As Dt and Dr are in [cm] and [s] units, the obtained result is given in cm^2/s """ 


        if self.effPhi is None:
            print("The effective volume fraction was not computed.\n"
                    + "Use self.compEffVolFrac() method before calling this method.\n")
            return

        qVals = np.array( qVals ).astype('f') * 1e8 #_Conversion to cm^(-1)

        self.compCorrectedDt()
        self.compCorrectedDs()

        density = self.dataset.getCOMRadialNumberDensity('protH', dr=density_dr, maxR=maxDensityR, frame=frame)

        solRoot = root(self.fitFuncAppDiffCoeff, x0=self.Dt, args=(qVals, density, maxN), method='lm') 

        self.roots = solRoot
                





    def fitFuncAppDiffCoeff(self, D, qVals, density, maxN=100):
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




#---------------------------------------------
#_Other physical parameters (solute volume, D2O density,...)
#---------------------------------------------
    def getProtVolume(self, frame=-1):
        """ Estimates protein volume using ConvexHull for each residue.
            A scaling of 10 is used for each residue to take into account 
            an average atom volume of 10 Angströms.
            
            There should be only one protein in the trajectories. 
            
            Returns volume in cm**3 """

        selProt = self.dataset.getSelection('protein')

        nbrResid = int(self.dataset.psfData.atoms[selProt,2][-1])
        
        vol = 0
        for i in range(nbrResid):
            residRange = np.argwhere( self.dataset.psfData.atoms[selProt,2] == str(i+1) )[:,0]
            vol += ConvexHull(self.dataset.dcdData[residRange, frame]).volume * 10


        return vol * 1e-24





    def getSpecVolume(self, frame=-1):
        """ Estimates protein specific volume based on volume estimation.

            There should be only one protein in the trajectories. 
            
            Returns specific volume in cm**3 / g """

        massP = np.sum( self.dataset.getAtomsMasses('protein') )

        protVol = self.getProtVolume(frame)

        specVol = protVol / massP * 6.02214076e23

        return specVol






    def getViscosityD2O(self, T, dT=None):
        """ Calculates the viscosity of D2O in units of [Pa*s] for given temperature in units of [K]
            Dependence of viscosity on temperature was determined by Cho et al. (1999)
            "Thermal offset viscosities of liquid H2O, D2O and T2O", J.Phys. Chem. B
            103(11):1991-1994

            eta(T) is valid from 280k up to 400K """

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



    def getDensityD2O(self, T):
        """ Calculates the density of D2O in units of [g/cm^3] for given temperature T in [K]
            Handbook of Chemistry and Physics, 73rd Edition, 
            Lide, D.R. Ed.; CRC Press: Boca Raton 1992; Chapter 6, pg. 13
            http://physchem.uni-graz.ac.at/sm/Service/Water/D2Odens.htm """

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


