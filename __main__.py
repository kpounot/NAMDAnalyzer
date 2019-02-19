import sys, os

import argparse

from package.NAMDAnalyzer import NAMDAnalyzer



parser = argparse.ArgumentParser()
parser.add_argument("fileList", nargs="*", help="List of data files to be loaded on startup.")
parser.add_argument("-s", "--stride", type=int, nargs='?', default=1,
                help="Stride parameter to extract only a subset of the dcd file.")


args = parser.parse_args()

data = NAMDAnalyzer(args.fileList, args.stride)

