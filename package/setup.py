from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

import os


with open('../README.md', 'r') as f:
    description = f.read()


packagesList = [    'NAMDAnalyzer',
                    'NAMDAnalyzer.dataManipulation',
                    'NAMDAnalyzer.dataParsers',
                    'NAMDAnalyzer.dataConverters',
                    'NAMDAnalyzer.lib',
                    'NAMDAnalyzer.helpersFunctions' ]



pycompIntScatFunc_ext = Extension( "NAMDAnalyzer.lib.pycompIntScatFunc", 
                                   ["NAMDAnalyzer/lib/pycompIntScatFunc.pyx", 
                                    "NAMDAnalyzer/lib/src/compIntScatFunc.c"],
                                   include_dirs=["NAMDAnalyzer/lib/src", np.get_include()],
                                   extra_compile_args=['-Ofast'],
                                   extra_link_args=['-Ofast'] )



pygetWithin_ext       = Extension( "NAMDAnalyzer.lib.pygetWithin", 
                                   ["NAMDAnalyzer/lib/pygetWithin.pyx", "NAMDAnalyzer/lib/src/getWithin.cpp"],
                                   include_dirs=["NAMDAnalyzer/lib/src",  np.get_include()],
                                   language='c++',
                                   extra_compile_args=['-Ofast'],
                                   extra_link_args=['-Ofast'])


pygetCOM_ext = Extension( "NAMDAnalyzer.lib.pygetCenterOfMass", 
                          ["NAMDAnalyzer/lib/pygetCenterOfMass.pyx"],
                          include_dirs=[np.get_include()] ) 


pysetCOMAligned_ext = Extension( "NAMDAnalyzer.lib.pysetCenterOfMassAligned", 
                                 ["NAMDAnalyzer/lib/pysetCenterOfMassAligned.pyx"],
                                 include_dirs=[np.get_include()] ) 


setup(  name='NAMDAnalyzer',
        version='alpha',
        description=description,
        author='Kevin Pounot',
        author_email='kpounot@hotmail.fr',
        url='github.com/kpounot/NAMDAnalyzer',
        packages=packagesList,
        ext_modules=cythonize( [pycompIntScatFunc_ext, 
                                pygetWithin_ext,
                                pygetCOM_ext,
                                pysetCOMAligned_ext] )  )

