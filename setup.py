from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


with open('README.md', 'r') as f:
    description = f.read()


packagesList = [    'package',
                    'package.dataManipulation',
                    'package.dataParsers',
                    'package.dataConverters',
                    'package.helpersFunctions',
                    'package.test'  ]


pycompIntScatFunc_ext = Extension( "pycompIntScatFunc", 
                                   ["package/lib/pycompIntScatFunc.pyx", "package/lib/src/compIntScatFunc.c"],
                                   include_dirs=["package/lib/src", np.get_include()],
                                   extra_compile_args=['-Ofast'],
                                   extra_link_args=['-Ofast'])



pygetWithin_ext       = Extension( "pygetWithin", 
                                   ["package/lib/pygetWithin.pyx", "package/lib/src/getWithin.cpp"],
                                   include_dirs=["package/lib/src",  np.get_include()],
                                   language='c++',
                                   extra_compile_args=['-Ofast'],
                                   extra_link_args=['-Ofast'])


pygetCOM_ext = Extension( "pygetCenterOfMass", ["package/lib/pygetCenterOfMass.pyx"],
                                   include_dirs=[np.get_include()] ) 


pysetCOMAligned_ext = Extension( "pysetCenterOfMassAligned", ["package/lib/pysetCenterOfMassAligned.pyx"],
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

