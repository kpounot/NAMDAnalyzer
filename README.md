# NAMDAnalyzer
Python based routines for molecular dynamics simulation analysis from NAMD

The  programm is organized on a master class contained in NAMDAnalyzer.
New files can be loaded using the _importFile method.

Start the interpreter:

Initialization of the ipython console can be done using the following command:

ipython {path to NAMDAnalyzer.py} {list of files to be loaded}

Usage:

Each class in dataParser contains an __init__ routine, in which the file is read and data extracted.
They contain also methods to get access to data in numpy array format,
and methods to plot data in different ways.
In addition, plottin methods accept a 'fit=True', 'model=<modelToUse>', 'p0=<starting parameters>' arguments to fit the data, 
which are extracted from the plot figure and fitted using scipy's curve_fit method, and using the given model 
(typically a lambda function) with the given starting parameters p0 (python list).
