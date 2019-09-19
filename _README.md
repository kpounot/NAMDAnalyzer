NAMDAnalyzer
============

Python based routines for molecular dynamics simulation analysis from NAMD


Installation:
-------------

Unix and Windows
^^^^^^^^^^^^^^^^

To install it within your python distribution, use 

    make [build] (for openMP version) or [build_cuda] (for CUDA accelerated version - recommended) 
    make install

or
    
    python setup.py [build, build_cuda]
    python setup.py install


Start the interpreter:
----------------------

Initialization of the ipython console can be done using the following command:

    ipython -i <path to NAMDAnalyzer.py> -- <list of files to be loaded> [-s stride]

Options: 
^^^^^^^^

- -s stride -> use to skip frames when loading a .dcd file. For instance if "-s 5" is provided, 
only the frames that are multiples of 5 will be loaded.

Usage:
---------
The program is organized on a master class contained in NAMDAnalyzer.Dataset.

Open log file and plot data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To analyze log file, the following can be used:

.. code-block:: python
    import NAMDAnalyzer as nda

    data = nda.Dataset('20190326_fss_tip4p_prelax.out') #_Import log file

    #_Another log file can be append to the imported one using data.logData.appendLOG() method  

    #_To plot data series using keywords given in data.logData.etitle, removing first 500 frames of minimization
    data.logData.plotDataSeries('TEMP KINETIC TOTAL', begin=501)

    #_To plot data distribution
    data.logData.plotDataDistribution('KINETIC', binSize=20)

    #_Data can be fitted with any model using 
    data.logData.plotDataDistribution('KINETIC', fit=True, model=your_model_function, p0=init_parameters)

Outputs of previous code:

.. image:: /docs/fig/log_dataSeries.png
   :scale: 25 %

.. image:: /docs/fig/log_dataDist.png
   :scale: 25 %
