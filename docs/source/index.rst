.. NAMDAnalyzer documentation master file, created by
   sphinx-quickstart on Wed Sep 18 17:51:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NAMDAnalyzer
============

NAMDAnalyzer is a python based API providing several analysis routines
for NAMD generated files.

Installation:
-------------

Unix and Windows
^^^^^^^^^^^^^^^^

To install it within your python distribution, use 

::

    make [build] (for openMP version) or [build_cuda] (for CUDA accelerated version - recommended) 
    make install

or

::
    
    python [setup.py, cuda_setup.py]
    python setup.py install


Start the interpreter:
----------------------

Initialization of the ipython console can be done using the following command:

:: 

    ipython -i <path to NAMDAnalyzer.py> -- <list of files to be loaded> [-s stride]

Options: 

- -s --stride 
    use to skip frames when loading a .dcd file. For instance if "-s 5" is provided, 
    only the frames that are multiples of 5 will be loaded.



Reference
---------

.. toctree::
   :maxdepth: 2

   dataset
   dataAnalysis/index
   dataManipulation/index
   dataParsers/index
   helpersFunctions/index
   kdTree/index
   license
   help


Quick start
-----------

.. include:: quickstart.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
