# NAMDAnalyzer
Python based routines for molecular dynamics simulation analysis from NAMD


## Installation:

### Unix
Change compiler to unix in the setup.cfg file.

To install it within your python distribution, use 

    make 
    make install

or
    
    python3 setup.py build
    python3 setup.py install


### Windows
MinGW is necessary to compile C and cython routines.
Change compiler to mingw32 i nthe setup.cfg file.

To install it within your python distribution, use 

    mingw32-make.exe 
    mingw32-make.exe install


## Start the interpreter:
Initialization of the ipython console can be done using the following command:

    ipython -i <path to NAMDAnalyzer.py> -- <list of files to be loaded> [-s stride]

### Options: 

- -s stride -> use to skip frames when loading a .dcd file. For instance if "-s 5" is provided, only the frames that are multiples of 5 will be loaded.

## Usage:
The program is organized on a master class contained in NAMDAnalyzer.

New files can be loaded using the importFile method.

Each class in dataParser contains an __init__ routine, in which the file is read and data extracted.
They contain also methods to get access to data in numpy array format,
and methods to plot data in different ways.
In addition, plotting methods accept a 'fit=True' and 'model=<modelToUse>' arguments to fit the data, which are extracted 
from the plot figure and fitted using scipy's curve_fit method using the given model (typically a lambda function).

The user can also define its own selection of atoms to work with. For this, use the getSelection method of the master class.
The selections can then be used with several other methods like dcdData.getSTDperAtom.

NAMDAnalyzer contains also methods to convert trajectories to Quasi-Elastic Neutron Scattering spectra (in NAMDAnalyzer.dcdData class). 
These can be used for simulations validation with experimental data.
