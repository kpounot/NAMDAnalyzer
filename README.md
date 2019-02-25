# NAMDAnalyzer
Python based routines for molecular dynamics simulation analysis from NAMD

The  programm is organized on a master class contained in NAMDAnalyzer.
New files can be loaded using the importFile method.


Installation:

Windows
MinGW is necessary to compile C and cython routines.
Then simply use nmake to build the API inplace, which can be use directly from powershell console (see 'Start the interpreter')

To install it within your python distribution, use 
    python setup.py build_ext bdist
Then use 
    pip install ./dist/<name of .zip file created>


Start the interpreter:
Initialization of the ipython console can be done using the following command:
ipython -i <path to NAMDAnalyzer.py> -- <list of files to be loaded> [-s stride]

    -s stride -> use to skip frames when loading a .dcd file. For instance if "-s 5" is provided, only the frames that are multiples of 5 will be loaded.

Usage:
Each class in dataParser contains an __init__ routine, in which the file is read and data extracted.
They contain also methods to get access to data in numpy array format,
and methods to plot data in different ways.
In addition, plotting methods accept a 'fit=True' and 'model=<modelToUse>' arguments to fit the data, which are extracted 
from the plot figure and fitted using scipy's curve_fit method using the given model (typically a lambda function).

The user can also define its own selection of atoms to work with. For this, use the getSelection method of the master class.
The selections can then be used with several other methods like dcdData.getSTDperAtom.

NAMDAnalyzer contains also methods to convert trajectories to Quasi-Elastic Neutron Scattering spectra (in NAMDAnalyzer.dcdData class). 
These can be used for simulations validation with experimental data.
