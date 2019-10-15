"""

Classes
^^^^^^^

"""

import sys

import numpy as np

import matplotlib.pyplot as plt

try:
    from NAMDAnalyzer.lib.pylibFuncs import (py_cdf, 
                                             py_waterOrientAtSurface, 
                                             py_setWaterDistPBC,
                                             py_getWaterOrientVolMap)
except ImportError:
    print("NAMDAnalyzer C code was not compiled, several methods won't work.\n"
            + "Please compile it before using it.\n")





class Rotations:
    """ This class defines methods to compute rotational relaxation and orientation probabilities. 
        
        Some plotting methods are also available to quicly check the results. 
        
        :arg data:        a Dataset class instance containing trajectories data 
        :arg sel1:        first selection corresponding to one end of each vector
        :arg sel2:        second selection for vectors, should be of same size as sel1
        :arg tMax:        maximum number of frames to be used 
        :arg step:        time interval between each computed vectors
        :arg dPhi:        angular bin size for orientational probability (in degrees)
        :arg axis:        reference axis for orientation probabilities
        :arg nbrTimeOri:  number of time origins to be averaged over (optional, default 25) """


    def __init__(self, data, sel1, sel2, tMax=100, step=1, dPhi=0.5, axis='z', nbrTimeOri=20):
                                                                                                            
        self.data       = data
        self.sel1       = sel1
        self.sel2       = sel2
        self.tMax       = tMax
        self.step       = step
        self.dPhi       = dPhi / 180 * np.pi
        self.axis       = axis
        self.nbrTimeOri = nbrTimeOri

        self.rotCorr    = None
        self.rotDensity = None
        self.times      = None
        self.angles     = None



#---------------------------------------------
#_Computation methods
#---------------------------------------------
    def compRotationalRelaxation(self):
        """ For each frame in the range of tMax with gieven step, computes the distance vector between 
            sel1 and sel2 and performs scalar product between the first one and all other one. 
            This is averaged over multiple time origins and the obtained correlation is stored
            in self.rotRelax variable. 
            
            References: 
                
                - Yu-ling Yeh and Chung-Yuan Mou (1999). Orientational Relaxation 
                  Dynamics of Liquid Water Studied by Molecular Dynamics Simulation, 
                  J. Phys. Chem. B 1999, 103, 3699-3705. 

        """


        self.times  = ( np.arange(0, self.tMax, self.step, dtype=int) 
                        * self.data.dcdFreq[0] * self.data.timestep * 1e12 )
        corr        = np.zeros_like(self.times)

        oriList = ( (self.data.nbrFrames - self.tMax) * np.random.random(self.nbrTimeOri) ).astype(int)
        for idx, frame in enumerate(oriList):

            sel1 = self.sel1 + ' frame %i' % frame
            sel2 = self.sel2 + ' frame %i' % frame

            sel1 = self.data.dcdData[self.data.selection(sel1), frame:frame+self.tMax:self.step]
            sel2 = self.data.dcdData[self.data.selection(sel2), frame:frame+self.tMax:self.step]

            angles  = sel2 - sel1
            angles  = angles / np.sqrt( np.sum( angles**2, axis=2 ) )[:,:,np.newaxis]

            angles  = np.sum(angles[:,[0]] * angles, axis=2)

            corr += np.mean( (3*angles**2 - 1) / 2, axis=0) / self.nbrTimeOri


        self.rotCorr    = corr


        



    def compOrientationalProb(self):
        """ Compute the probability for the vector between sel1 and sel2 to be in a 
            particular orientation.
            The angle is computed with respect to a given axis. 
            Averaging is performed for each frame between 0 and tMax with given step. 

        """

        if isinstance(self.axis, str):
            if self.axis=='x':
                ref = np.array( [[[1, 0, 0]]] )
            elif self.axis=='y':
                ref = np.array( [[[0, 1, 0]]] )
            elif self.axis=='z':
                ref = np.array( [[[0, 0, 1]]] )
        else:
            ref = self.axis / np.sqrt( np.sum( self.axis**2 ) )


        self.angles     = np.arange(0, np.pi, self.dPhi, dtype='float32')
        self.rotDensity = np.zeros_like( self.angles, dtype='float32' )
        

        sel1 = self.data.dcdData[self.data.selection(self.sel1)]
        sel2 = self.data.dcdData[self.data.selection(self.sel2)]

        angles  = sel2 - sel1
        angles  = angles / np.sqrt( np.sum( angles**2, axis=2 ) )[:,:,np.newaxis]

        angles = np.arccos( np.sum(ref * angles, axis=2) )
        angles = angles.flatten().astype('float32')

        normF = angles.size

        py_cdf(angles, self.rotDensity, self.angles[-1], self.dPhi, normF)




#---------------------------------------------
#_Plotting methods
#---------------------------------------------
    def plotRotationalRelaxation(self):
        """ Used to quickly plot rotational relaxation function """


        fig, ax = plt.subplots()

        ax.plot(self.times, self.rotCorr, marker='o')
        ax.set_xlabel('Time [ps]')
        ax.set_ylabel('$C_{\hat{u}}(t)$')

        fig.show()



    def plotOrientationalProb(self):
        """ Used to quickly plot orientational probability """


        fig, ax = plt.subplots()

        ax.plot(self.angles, self.rotDensity)
        ax.set_xlabel('$\\theta \ [rad]$')
        ax.set_ylabel('$P(\\theta)$')

        fig.show()













class WaterAtProtSurface:
    """ This class provides way to compute water molecules orientation relative to protein surface.

        Basically, for each selected water molecule, approximately 3-7 nearest atoms from protein
        are selected, their geometric center is computed and the vector between water oxygen and this
        center defines the normal to protein surface.
        The dipole moment vector of water molecule is dotted with this normal vector 
        to determine the orientation.
        
        :arg data:    a Dataset class instance containing trajectories data 
        :arg maxR:    maximum distance from protein surface for water molecules selection
        :arg maxN:    maximum number of protein atoms to be used to compute normal to surface
        :arg frames:  frames to be used to average orientations
        :arg watVec:  vector on water molecule to be used to compute orientation, 
                      can be 'D' for electric dipole moment, 'H1' for O-H1 vector or 'H2' for O-H2 vetor.
        :arg nbrVox:  number of voxels to be used in each dimensions to generate the volumetric map.

    """
                          


    def __init__(self, data, protSel='protein', maxR=5, maxN=5, frames=None, watVec='D', nbrVox=256):

        self.data = data

        if isinstance(protSel, str):
            self.protSel = self.data.selection(protSel) 
        else:
            self.protSel = protSel

        self.maxR = maxR
        self.maxN = maxN

        if frames is None:
            self.frames = np.arange(0, self.data.nbrFrames)
        else:
            self.frames = frames

        
        self.watVec = watVec

        self.nbrVox = nbrVox


        self.orientations = None



    def compOrientations(self):
        """ Computes, for all selected water oxygens, the orientation of the dipole moment vector relative to
            protein surface. 

            This generates a volumetric dataset of size 256*256*256, in which each voxel carries a value
            corresponding to the angle.

        """

        waterO = self.data.selection('name OH2')
        waterH = self.data.selection('name H1 H2')

        waterO = self.data.dcdData[waterO, self.frames]

        waterH = self.data.dcdData[waterH, self.frames]

        if self.watVec == "D":
            watVec = ( (waterO - waterH[::2]) + (waterO - waterH[1::2]) ) / 2 
        elif self.watVec == "H1":
            watVec =  waterH[0] - waterO
        elif self.watVec == "H2":
            watVec =  waterH[1] - waterO

        prot = self.data.dcdData[self.protSel,self.frames]

        out = np.zeros( (waterO.shape[0], waterO.shape[1]), dtype='float32' )

        cellDims = self.data.cellDims[self.frames]
    

        py_waterOrientAtSurface(waterO, watVec, prot, out, cellDims, self.maxR, self.maxN) 


        self.orientations = np.ascontiguousarray(waterO[:,:,0], dtype='float32')
        self.keepWat      = np.ascontiguousarray(waterO[:,:,1], dtype=int)




    def generateVolMap(self):
        """ Generates a volumetric map with water orientations relative to protein surface.
            
            First, protein structures are aligned for all frames. 
            Then water molecules are moved based on periodic boundary conditions applied on
            distance between water oxygen and closest protein atom, such that the volumetric map
            corresponds to the positions computed with :py:meth:compOrientations .

            Eventually, for each voxel, orientation of water molecules present in the voxel limits 
            is averaged over all frames.

            Using :py:meth:writeVolMap, a .pdb file and a .dx file are created and can be
            directly imported into VMD to visualize average orientation for each voxel. 

        """

        
        prot = self.data.dcdData[self.protSel,self.frames]


        cellDims = self.data.cellDims[self.frames]


        water = self.data.selection('water')
        nbrWAtoms = water.getUniqueAtomName().size
        water = self.data.dcdData[water,self.frames]

        #_Moves water molecules to their nearest atom
        py_setWaterDistPBC(water, prot, cellDims, nbrWAtoms)


        #_Getting minimum and maximum of all coordinates
        min_x = np.concatenate( (water, prot) )[:,:,0].min()
        min_y = np.concatenate( (water, prot) )[:,:,1].min()
        min_z = np.concatenate( (water, prot) )[:,:,2].min()
        minCoor = np.array( [min_x, min_y, min_z] )

        #_Moving coordinates close to origin
        prot  -= minCoor
        water -= minCoor

        max_x = np.concatenate( (water, prot) )[:,:,0].max() 
        max_y = np.concatenate( (water, prot) )[:,:,1].max()
        max_z = np.concatenate( (water, prot) )[:,:,2].max()
        maxCoor = np.array( [max_x, max_y, max_z] ) * 1.001

        
        self.volMap = np.zeros( (self.nbrVox, self.nbrVox, self.nbrVox), dtype='float32')
    
        #_Get water indices on volMap based on coordinates
        indices = (water[::nbrWAtoms,:,:] / maxCoor * self.nbrVox).astype('int32')

        counts = np.zeros_like(self.volMap, dtype='int32')


        py_getWaterOrientVolMap(indices, self.orientations, counts, self.volMap)


        counts[counts == 0] = 1

        self.volMap     /= counts
        self.volOri      = np.array([0.0, 0.0, 0.0])
        self.volDeltas   = maxCoor / self.nbrVox
        self.pCoor       = prot
        self.wCoor       = water






    def writeVolMap(self, fileName=None, frame=0):
        """ Write the volumetric map containing averaged water orientations relative to protein surface.
            
            The file is in the APBS .dx format style, so that it can be imported directly into VMD.
            Moreover, a pdb file is also generated containing frame averaged coordinates for aligned protein.

            :arg fileName: file name for .pdb and .dx files. If none, the name of the loaded .psf file
                           is used.
            :arg frame:    frame to be used to generate the pdb file

        """

        if fileName is None:
            fileName = self.data.psfFile[:-4]

        volMap = self.volMap.flatten()

        wSel = self.data.selection('water')

        #_Find indices to keep
        nbrWat = wSel.getUniqueAtomName().size
        toKeep = np.zeros(nbrWat * self.keepWat.shape[0])
        for i in range(nbrWat):
            toKeep[i::nbrWat] += self.keepWat[:,frame]


        #_Gets water and protein coordinates for selected frame
        wCoor = self.wCoor[toKeep.astype(bool), frame]
        pCoor = self.pCoor[:,frame]

        coor  = np.concatenate( (pCoor, wCoor) ).squeeze()


        #_Write the volumetric map file
        with open(fileName + '.dx', 'w') as f:
            f.write('object 1 class gridpositions counts ' + 3*'%i ' % ( self.nbrVox,
                                                                        self.nbrVox,
                                                                        self.nbrVox ) + '\n')
            f.write('origin %f %f %f\n' % (self.volOri[0], self.volOri[1], self.volOri[2]))
            f.write('delta %f 0.000000 0.000000\n' % self.volDeltas[0])
            f.write('delta 0.000000 %f 0.000000\n' % self.volDeltas[1])
            f.write('delta 0.000000 0.000000 %f\n' % self.volDeltas[2])
            f.write('object 2 class gridconnections counts ' + 3*'%i ' % ( self.nbrVox,
                                                                          self.nbrVox,
                                                                          self.nbrVox) + '\n')
            f.write('object 3 class array type float rank 0 items %i data follows\n' % self.nbrVox**3)

            for idx in range( int(np.ceil(volMap.size / 3)) ):
                batch = volMap[3*idx:3*idx+3]
                for val in batch:
                    f.write('%f ' % val)

                f.write('\n')


            f.write('attribute "dep" string "positions"\n')
            f.write('object "regular positions regular connections" class field\n')
            f.write('component "positions" value 1\n')
            f.write('component "connections" value 2\n')
            f.write('component "data" value 3\n')



        #_Write the PDB file
        wSel = self.data.selection( wSel._indices[np.argwhere(toKeep.astype(bool))[:,0]] )

        sel = self.protSel + wSel

        sel.writePDB(fileName, coor=coor)

