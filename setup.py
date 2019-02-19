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
                                   include_dirs=["package/lib/src",  np.get_include()] )

pygetWithin_ext       = Extension( "pygetWithin", 
                                   ["package/lib/pygetWithin.pyx", "package/lib/src/getWithin.cpp"],
                                   include_dirs=["package/lib/src",  np.get_include()],
                                   language='c++' )


setup(  name='NAMDAnalyzer',
        version='alpha',
        description=description,
        author='Kevin Pounot',
        author_email='kpounot@hotmail.fr',
        url='github.com/kpounot/NAMDAnalyzer',
        packages=packagesList,
        ext_modules=cythonize( [pycompIntScatFunc_ext, pygetWithin_ext] )  )

