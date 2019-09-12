# NAMDAnalyzer
Python based routines for molecular dynamics simulation analysis from NAMD


## Installation:

### Unix and Windows
To install it within your python distribution, use 

    make [build] (for openMP version) or [build_cuda] (for CUDA accelerated version - recommended) 
    make install

or
    
    python setup.py [build, build_cuda]
    python setup.py install


## Start the interpreter:
Initialization of the ipython console can be done using the following command:

    ipython -i <path to NAMDAnalyzer.py> -- <list of files to be loaded> [-s stride]

### Options: 

- -s stride -> use to skip frames when loading a .dcd file. For instance if "-s 5" is provided, 
only the frames that are multiples of 5 will be loaded.

## Usage:
The program is organized on a master class contained in NAMDAnalyzer.Dataset.

#### Open log file and plot data
To analyze log file, the following can be used:

``` python
import NAMDAnalyzer as nda

data = nda.Dataset('20190326_fss_tip4p_prelax.out') #_Import log file

#_Another log file can be append to the imported one using data.logData.appendLOG() method  

#_To plot data series using keywords given in data.logData.etitle, removing first 500 frames of minimization
data.logData.plotDataSeries('TEMP KINETIC TOTAL', begin=501)

#_To plot data distribution
data.logData.plotDataDistribution('KINETIC', binSize=20)

#_Data can be fitted to any model using 
data.logData.plotDataDistribution('KINETIC', fit=True, model=your_model_function, p0=init_parameters)

```

Outputs of previous code:

<table>
    <tr>
        <td>
            <img src="/doc/fig/log_dataSeries.png" width="250">
        </td>
        <td>
            <img src="/doc/fig/log_dataDist.png" width="250">
        </td>
    </tr>
</table>



#### Load trajectories, selection, and analysis

``` python

import NAMDAnalyzer as nda

d = nda.Dataset('psfFile.psf', 'dcdFile.dcd')

#_Trajectories can be append to already loaded ones using either d.appendDCD('dcdFile.dcd')
#_or d.appendCoordinates('pdbFile.pdb').


#_To compute RMSD per atom for molecules aligned in all frames
d.getRMSDperAtom(selection='protein and segname V1', align=True, frames=slice(0, None))

#_To compute and plot RMSD per atom for molecules aligned in all frames
d.plotRMSDperAtom(selection='protein and segname V1', align=True, frames=slice(0, None))



#_To compute radial pair distribution function for water within 3 angstrom of a protein region
r, pdf = d.getRadialNumberDensity( 'name OH2 and within 3 of protein and resid 40:80',
                                   'name OH2 and within 3 of protein and resid 40:80',
                                   dr=0.1, maxR=15, frames=range(0,1000,5) )

import matplotlib.pyplot as plt

plt.plot(r, pdf)
plt.xlabel('radius r [$\AA$]')
plt.ylabel('$\\rho (r)$')
plt.show()



#_To plot averaged distances between a residue and the rest of the protein using a parallel plot
d.plotAveragedDistances_parallelPlot('protein and resid 53', 'protein', maxDist=10, step=2)

#_To plot the same distances but using a chord diagram
cd = d.plotAveragedDistances_chordDiagram('protein and resid 53', 'protein', maxDist=10, step=2)
cd.show()

```


Outputs of previous code:

<table>
    <tr>
        <td>
            <img src="/doc/fig/ubq_rmsdPerAtom.png" width="200">
        </td>
        <td>
            <img src="/doc/fig/radialDistWater.png" width="200">
        </td>
        <td>
            <img src="/doc/fig/averagedDistances_parallel.png" width="200">
        </td>
        <td>
            <img src="/doc/fig/averagedDistances_chord.png" width="200">
        </td>
    </tr>
</table>


