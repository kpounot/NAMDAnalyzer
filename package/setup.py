from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

import os


with open('../README.md', 'r') as f:
    description = f.read()




packagesList = [    'NAMDAnalyzer.dataManipulation',
                    'NAMDAnalyzer.dataParsers',
                    'NAMDAnalyzer.dataAnalysis',
                    'NAMDAnalyzer.lib',
                    'NAMDAnalyzer.helpersFunctions' ]



pycompIntScatFunc_ext   = Extension( "NAMDAnalyzer.lib.pycompIntScatFunc", 
                                   ["NAMDAnalyzer/lib/src/compIntScatFunc.c", 
                                    "NAMDAnalyzer/lib/pycompIntScatFunc.pyx"],
                                   include_dirs=["NAMDAnalyzer/lib/src", np.get_include()],
                                   extra_compile_args=['-fopenmp'],
                                   extra_link_args=['-fopenmp'] )



pygetWithin_ext         = Extension( "NAMDAnalyzer.lib.pygetWithin", 
                                   ["NAMDAnalyzer/lib/pygetWithin.pyx", "NAMDAnalyzer/lib/src/getWithin.cpp"],
                                   include_dirs=["NAMDAnalyzer/lib/src",  np.get_include()],
                                   language='c++',
                                   extra_compile_args=['-fopenmp'],
                                   extra_link_args=['-fopenmp'] )


pygetCOM_ext            = Extension( "NAMDAnalyzer.lib.pygetCenterOfMass", 
                                    ["NAMDAnalyzer/lib/pygetCenterOfMass.pyx"],
                                    include_dirs=[np.get_include()] ) 


pysetCOMAligned_ext     = Extension( "NAMDAnalyzer.lib.pysetCenterOfMassAligned", 
                                    ["NAMDAnalyzer/lib/pysetCenterOfMassAligned.pyx"],
                                    include_dirs=[np.get_include()] ) 


setup(  name='NAMDAnalyzer',
        version='alpha',
        description=description,
        author='Kevin Pounot',
        author_email='kpounot@hotmail.fr',
        url='github.com/kpounot/NAMDAnalyzer',
        py_modules=['NAMDAnalyzer.Dataset'],
        packages=packagesList,
        ext_modules=cythonize( [pycompIntScatFunc_ext, 
                                pygetWithin_ext,
                                pygetCOM_ext,
                                pysetCOMAligned_ext] )  )

