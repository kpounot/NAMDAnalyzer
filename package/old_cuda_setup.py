import os

from setuptools import setup
from setuptools import Extension
from setuptools import Command
from setuptools.command.build_ext import build_ext
import distutils

from Cython.Build import cythonize
from Cython.Compiler import Options
Options._directive_defaults['language_level'] = 3

import numpy as np


with open('../README.md', 'r') as f:
    description = f.read()


pyxPath = "NAMDAnalyzer/lib/cython_pyx/"
srcPath = "NAMDAnalyzer/lib/openmp/src/" 
cudaSrcPath = "NAMDAnalyzer/lib/cuda/src/"


try:
    if os.environ['OS'] == 'Windows_NT':
        cudaPath = os.environ['CUDA_PATH'] #_Using the default installation key in Path variable
        cudaInclude = cudaPath + "\\include"
        cudaLib     = cudaPath + "\\lib\\x64"
        ext         = "obj"
    else:
        cudaPath    = "usr/" 
        cudaInclude = cudaPath + "/include"
        cudaLib     = cudaPath + "/lib64"
        ext         = "o"
except KeyError:
    print("\n\nError: Couldn't locate CUDA path, please intall it or add it to PATH variable\n\n")
    


#_The following is used to compile with openmp with both mingGW and msvc
copt =  {'msvc'     : ['/openmp', '/Ox', '/fp:fast'],
         'mingw32'  : ['-fopenmp','-O3','-ffast-math','-march=native'] }
lopt =  {'mingw32'  : ['-fopenmp']}




def preprocessNVCC(path):
    #_Used to process .cu file and create static libraries for GPU part of the program

    for f in os.listdir(path):
       if f[-3:] == '.cu':
            os.system("nvcc -lib -o %s.lib %s" % ('NAMDAnalyzer/lib/cuda/' + f[:-3], path + f))



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

        #_Simply uses execute function to make sure .cu files are processed first with nvcc
        self.execute(preprocessNVCC, [cudaSrcPath])

        build_ext.build_extensions(self)




packagesList = [    'NAMDAnalyzer.dataManipulation',
                    'NAMDAnalyzer.dataParsers',
                    'NAMDAnalyzer.dataAnalysis',
                    'NAMDAnalyzer.lib',
                    'NAMDAnalyzer.helpersFunctions' ]


#_Defines extensions
pylibFunc_ext   = Extension( "NAMDAnalyzer.lib.pycompIntScatFunc", 
                                   [cudaSrcPath + "compIntScatFunc.cpp", pyxPath + "pycompIntScatFunc.pyx"],
                                   library_dirs=["NAMDAnalyzer/lib/cuda", cudaLib],
                                   libraries=['compIntScatFunc', 'cuda', 'cudart'],
                                   language='c++',
                                   include_dirs=[cudaSrcPath, np.get_include(), cudaInclude])


pygetDistances_ext      = Extension( "NAMDAnalyzer.lib.pygetDistances", 
                                   [pyxPath + "pygetDistances.pyx", cudaSrcPath + "getDistances.cpp"],
                                   include_dirs=[cudaSrcPath, np.get_include(), cudaInclude],
                                   library_dirs=["NAMDAnalyzer/lib/cuda", cudaLib],
                                   libraries=['getDistances', 'cuda', 'cudart'],
                                   language='c++')


pygetHydrogenBonds_ext      = Extension( "NAMDAnalyzer.lib.pygetHydrogenBonds", 
                                   [pyxPath + "pygetHydrogenBonds.pyx", cudaSrcPath + "getHydrogenBonds.cpp"],
                                   include_dirs=[cudaSrcPath, np.get_include(), cudaInclude],
                                   library_dirs=["NAMDAnalyzer/lib/cuda", cudaLib],
                                   libraries=['getHydrogenBonds', 'cuda', 'cudart', 'cudadevrt'],
                                   language='c++')



pygetWithin_ext         = Extension( "NAMDAnalyzer.lib.pygetWithin", 
                                   [pyxPath + "pygetWithin.pyx", cudaSrcPath + "getWithin.cpp"],
                                   include_dirs=[cudaSrcPath, np.get_include(), cudaInclude],
                                   library_dirs=["NAMDAnalyzer/lib/cuda", cudaLib],
                                   libraries=['getWithin', 'cuda', 'cudart'],
                                   language='c++')

pygetCOM_ext            = Extension( "NAMDAnalyzer.lib.pygetCenterOfMass", 
                                    [pyxPath + "pygetCenterOfMass.pyx"],
                                    include_dirs=[np.get_include()] ) 


pysetCOMAligned_ext     = Extension( "NAMDAnalyzer.lib.pysetCenterOfMassAligned", 
                                    [pyxPath + "pysetCenterOfMassAligned.pyx"],
                                    include_dirs=[np.get_include()] ) 






setup(  name='NAMDAnalyzer',
        version='alpha',
        description=description,
        author='Kevin Pounot',
        author_email='kpounot@hotmail.fr',
        url='github.com/kpounot/NAMDAnalyzer',
        py_modules=['NAMDAnalyzer.Dataset'],
        packages=packagesList,
        ext_modules=cythonize( [pygetWithin_ext,
                                pygetDistances_ext,
                                pygetHydrogenBonds_ext,
                                pygetCOM_ext,
                                pysetCOMAligned_ext,
                                pycompIntScatFunc_ext]),
        cmdclass={'build_ext': build_ext_subclass})

