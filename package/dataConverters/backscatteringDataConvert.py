import numpy as np

from scipy.fftpack import fft, ifft, fftfreq, fftshift


class BackScatData:

    def __init__(self):

        self.EISF       = None
        self.interFunc  = None
        self.scatFunc   = None
        self.qVals      = None


#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compIntermediateFunc(self, qValList, minFrames, maxFrames, nbrBins=100, selection='protNonExchH', 
                                                                                    begin=0, end=None):
        """ This method computes intermediate function for all q-value (related to scattering angle)

            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    selection   -> atom selection
                    begin       -> first frame to be used
                    end         -> last frame to be used 
                    
            Returns an (nbr of q-values, timesteps) shaped array. """

        #_Get the indices corresponding to the selection
        if type(selection) == str:
            selection = self.psfData.getSelection(selection)

        self.qVals = qValList
        qValList = np.array(qValList).reshape(len(qValList), 1) #_For convenience

        #_Computes atoms positions
        atomPos = np.sqrt(np.sum(self.parent.alignCenterOfMass(selection, begin, end)**2, axis=2))

        intermediateFunc = np.zeros(qValList.size) #_Initial array of shape (number of q-values,)
        timestep = []
        for it in range(nbrBins):
            nbrFrames = minFrames + int(it * (maxFrames - minFrames) / nbrBins )
            
            #_Computes intermediate scattering function for one timestep, averaged over time origins
            corr = np.array( [(atomPos[:,i + nbrFrames] - atomPos[:,i]) 
                                                    for i in range(0, atomPos.shape[1] - nbrFrames)] )
            
            corr = qValList * np.mean(corr, axis=0) #_Get the exponent

            corr = np.exp( 1j * corr ) #_Exponentiation

            corr = np.mean( corr , axis=1 ) #_Averaging over atoms
            
            #_Add the intermediate function to the result array
            intermediateFunc = np.row_stack( (intermediateFunc, corr) )

            #_Compute time step
            timestep.append(self.parent.timestep * nbrFrames * self.parent.dcdFreq[0])


        self.interFunc = intermediateFunc[1:].T, timestep




    def compEISF(self, qValList, minFrames, maxFrames, nbrBins=100, resFunc=None, 
                                                selection='protNonExchH', begin=0, end=None):
        """ This method performs a multiplication of the inverse Fourier transform given resolution 
            function with the computed intermediate function to get the convoluted signal, 
            which can be used to compute MSD. 
            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    resFunc     -> resolution function to be used (optional, default resFuncSPHERES)
                    selection   -> atom selection (optional, default 'protein')
                    begin       -> first frame to be used (optional, default 0)
                    end         -> last frame to be used (optional, default None) """

        if resFunc == None:
            resFunc = self.parent.parent.resFuncSPHERES

        #_Gets intermediate scattering function        
        self.compIntermediateFunc(qValList, minFrames, maxFrames, nbrBins, selection, begin, end)

        intFunc, timesteps = self.interFunc 

        #_Gets resolution function 
        resolution = resFunc(fftfreq(intFunc.shape[1]))

        #_Inverse Fourier transform of resolution
        resolution = ifft(resolution)

        eisf = resolution * intFunc #_Computes the EISF

        self.EISF = eisf / eisf[:,0][:,np.newaxis], timesteps #_Returns the normalized EISF and time 




    def compScatteringFunc(self, qValList, minFrames, maxFrames, nbrBins=100, resFunc=None,
                                                        selection='protNonExchH', begin=0, end=None):
        """ This method calls getIntermediateFunc several times for different time steps, given by
            the number of frames, which will start from minFrames and by incremented to reach maxFrames
            in the given number of bins.

            Then, a Fourier transform is performed to compute the scattering function.

            Input:  qValList    -> list of q-values to be used 
                    minFrames   -> minimum number of frames to be used (lower limit of time integration)
                    maxFrames   -> maximum number of frames to be used (upper limit of time integration)
                    nbrBins     -> number of desired data points in the energy dimension (optional, default 50)
                    selection   -> atom selection (optional, default 'protein')
                    begin       -> first frame to be used (optional, default 0)
                    end         -> last frame to be used (optional, default None) """

        self.compEISF(qValList, minFrames, maxFrames, nbrBins, resFunc, selection, begin, end)

        scatFunc, timesteps = self.EISF 

        #_Performs the Fourier transform
        scatFunc = fftshift( fft(scatFunc, axis=1), axes=1 ) / scatFunc.shape[1]

        #_Convert time to energies
        energies = fftshift( 6.582119514e-2 * 2 * np.pi * fftfreq(scatFunc.shape[1], d=1/nbrBins) )  

        self.scatFunc = scatFunc, energies


#---------------------------------------------
#_Data accession methods
#---------------------------------------------

    def getEISF(self):
        """ Accession method for self.EISF attribute. """

        if not self.EISF:
            return "No EISF was computed yet. Use compEISF method before using this."
        else:
            return self.EISF


    def getIntermediateFunc(self):
        """ Accession method for self.interFunc attribute. """

        if not self.interFunc:
            return ("No intermediate function was computed yet. Use compIntermedateFunc method " 
                    + "before using this.")
        else:
            return self.interFunc


    def getScatFunc(self):
        """ Accession method for self.scatFunc attribute. """

        if not self.scatFunc:
            return ("No scattering function was computed yet. Use compScatFunc method "
                    + "before using this.")
        else:
            return self.scatFunc


#---------------------------------------------
#_Conversion methods (for nPDyn)
#---------------------------------------------
    def convertScatFunctoEISF(self, scatFunc):
        """ First finds the index corresponding to the 0 energy transfer on x axis.
            Then returns an array containing the intensity for each q-value. """

        #_Get the zero energy transfer index
        elasticIdx = np.argwhere(self.scatFunc[0] == np.max(self.scatFunc[0]))[0][1]

        return self.scatFunc[:,elaticIdx]
