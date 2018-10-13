# NAMDAnalyzer
Python based routines for molecular dynamics simulation analysis from NAMD

The  programm is organized on a master class contained in NAMDAnalyzer.
New files can be loaded using the importFile method.

Start the interpreter:
Initialization of the ipython console can be done using the following command:
ipython <path to NAMDAnalyzer.py> <list of files to be loaded>

Usage:
Each class in dataParser contains an __init__ routine, in which the file is read and data extracted.
They contain also methods to get access to data in numpy array format,
and methods to plot data in different ways.
In addition, plottin methods accept a 'fit=True' and 'model=<modelToUse>' arguments to fit the data, which are extracted 
from the plot figure and fitted using scipy's curve_fit method using the given model (typically a lambda function).

The user can also define its own selection of atoms to work with. For this, use the newSelection method of the master class.
The selections can then be used with several other methods like dcdData.getSTDperAtom.
