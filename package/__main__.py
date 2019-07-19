import sys, os

import argparse

from NAMDAnalyzer.Dataset import Dataset
from NAMDAnalyzer.dataAnalysis.backscatteringDataConvert import BackScatData
from NAMDAnalyzer.dataAnalysis.ScatDiffusion import ScatDiffusion
from NAMDAnalyzer.dataAnalysis.HydrogenBonds import HydrogenBonds



parser = argparse.ArgumentParser()
parser.add_argument("fileList", nargs="*", help="List of data files to be loaded on startup.")
parser.add_argument("-s", "--stride", type=int, nargs='?', default=1,
                help="Stride parameter to extract only a subset of the dcd file.")


args = parser.parse_args()

data = Dataset(args.fileList, args.stride)

