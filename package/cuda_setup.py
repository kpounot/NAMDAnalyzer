import os, sys

from setuptools import setup
from setuptools import Extension
from setuptools import Command
from setuptools.command.build_ext import build_ext
import distutils

from Cython.Build import cythonize
from Cython.Compiler import Options
Options._directive_defaults['language_level'] = 3

import numpy as np


with open('../README.rst', 'r') as f:
    description = f.read()


pyxPath = "NAMDAnalyzer/lib/cython_pyx/"
srcPath = "NAMDAnalyzer/lib/openmp/src/" 
cudaSrcPath = "NAMDAnalyzer/lib/cuda/src/"


try:
    if 'win32' in sys.platform:
        cudaPath    = os.environ['CUDA_PATH'] #_Using the default installation key in Path variable
        cudaInclude = cudaPath + "\\include"
        cudaLib     = cudaPath + "\\lib\\x64"
        libPrefix   = ''
        libExt      = '.lib'
        Xcompiler   = ''
    else:
        nvccPath    = os.popen('which nvcc').read().strip()
        cudaPath    = nvccPath[:nvccPath.find('/bin/nvcc')] 
        cudaInclude = cudaPath + "/include"
        cudaLib     = os.popen('locate libcuda.so').read().split()
        for libfile in cudaLib:
            if '64'in libfile:
                cudaLib = libfile[:libfile.find('/libcuda.so')]
        libPrefix   = 'lib'
        libExt      = '.a'
        Xcompiler   = "-Xcompiler '-fPIC'"
except KeyError:
    print("\n\nError: Couldn't locate CUDA path, please intall it or add it to PATH variable\n\n")
    


#_The following is used to compile with openmp with both mingGW and msvc
copt =  {'msvc'     : ['/openmp', '/Ox', '/fp:fast'],
         'mingw32'  : ['-fopenmp','-ffast-math','-march=native'], 
         'unix'     : ['-fopenmp','-ffast-math','-march=native'] }
lopt =  {'mingw32'  : ['-fopenmp'],
         'unix'     : ['-fopenmp']}




def preprocessNVCC(path):
    #_Used to process .cu file and create static libraries for GPU part of the program

    for f in os.listdir(path):
       if f[-3:] == '.cu':
            os.system("nvcc %s -lib -o %s%s%s %s" % (  Xcompiler,
                                                        'NAMDAnalyzer/lib/cuda/' + libPrefix, 
                                                        f[:-3],
                                                        libExt, 
                                                        path + f))



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
                    'NAMDAnalyzer.kdTree',
                    'NAMDAnalyzer.helpersFunctions' ]


#_Defines extensions
pylibFuncs_ext   = Extension( "NAMDAnalyzer.lib.pylibFuncs", 
                                   [cudaSrcPath + "compIntScatFunc.cpp", 
                                    cudaSrcPath + "getDistances.cpp", 
                                    cudaSrcPath + "getHydrogenBonds.cpp", 
                                    cudaSrcPath + "getWithin.cpp", 
                                    cudaSrcPath + "getParallelBackend.cpp", 
                                    "NAMDAnalyzer/lib/" + "libFunc.pyx"],
                                   library_dirs=["NAMDAnalyzer/lib/cuda", cudaLib],
                                   extra_objects=[
                                        'NAMDAnalyzer/lib/cuda/%scompIntScatFunc%s' % (libPrefix, libExt),
                                        'NAMDAnalyzer/lib/cuda/%sgetDistances%s' % (libPrefix, libExt),
                                        'NAMDAnalyzer/lib/cuda/%sgetHydrogenBonds%s' % (libPrefix, libExt),
                                        'NAMDAnalyzer/lib/cuda/%sgetWithin%s' % (libPrefix, libExt) ],
                                   libraries=['cuda', 'cudart'],
                                   language='c++',
                                   include_dirs=[cudaSrcPath, np.get_include(), cudaInclude])



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

