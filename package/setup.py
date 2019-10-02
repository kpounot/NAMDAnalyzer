import os

from setuptools import setup
from setuptools import Extension
from setuptools import Command
from setuptools.command.build_ext import build_ext

from Cython.Build import cythonize
from Cython.Compiler import Options
Options._directive_defaults['language_level'] = 3

import numpy as np

import site


with open('../README.rst', 'r') as f:
    description = f.read()


pyxPath = "NAMDAnalyzer/lib/"
srcPath = "NAMDAnalyzer/lib/openmp/src/" 

#_The following is used to compile with openmp with both mingGW and msvc
copt =  {'msvc': ['/openmp', '/Ox', '/fp:fast']  ,
         'mingw32'  : ['-fopenmp','-ffast-math','-march=native'] ,
         'unix'     : ['-fopenmp','-ffast-math','-march=native'] }
lopt =  {'mingw32'  : ['-fopenmp'],
         'unix'     : ['-fopenmp'] }



#_Used by setup function to define compile and link extra arguments
class build_ext_subclass( build_ext ):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt.keys():
           for e in self.extensions:
               e.extra_compile_args = copt[ c ]
        if c in lopt.keys():
            for e in self.extensions:
                e.extra_link_args = lopt[ c ]
        build_ext.build_extensions(self)




packagesList = [    'NAMDAnalyzer.dataManipulation',
                    'NAMDAnalyzer.dataParsers',
                    'NAMDAnalyzer.dataAnalysis',
                    'NAMDAnalyzer.selection',
                    'NAMDAnalyzer.lib',
                    'NAMDAnalyzer.kdTree',
                    'NAMDAnalyzer.helpersFunctions' ]


#_Defines extensions
pylibFuncs_ext   = Extension( "NAMDAnalyzer.lib.pylibFuncs", 
                                   [srcPath + "compIntScatFunc.cpp", 
                                    srcPath + "getDistances.cpp", 
                                    srcPath + "getRadialNbrDensity.cpp", 
                                    srcPath + "getHydrogenBonds.cpp", 
                                    srcPath + "getWithin.cpp", 
                                    srcPath + "getParallelBackend.cpp", 
                                    "NAMDAnalyzer/lib/common/src/" + "dcdReader.cpp", 
                                    "NAMDAnalyzer/lib/common/src/" + "dcdCellReader.cpp", 
                                    "NAMDAnalyzer/lib/" + "libFunc.pyx"],
                                   language='c++',
                                   include_dirs=[srcPath, np.get_include(),
                                                 "NAMDAnalyzer/lib/common/src"])







setup(  name='NAMDAnalyzer',
        version='1.0',
        description=description,
        author='Kevin Pounot',
        author_email='kpounot@hotmail.fr',
        url='github.com/kpounot/NAMDAnalyzer',
        py_modules=['NAMDAnalyzer.Dataset'],
        packages=packagesList,
        ext_modules=cythonize( [pylibFuncs_ext]),
        cmdclass={'build_ext': build_ext_subclass})

