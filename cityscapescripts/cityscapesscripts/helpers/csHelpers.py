#!/usr/bin/python

import os, sys, getopt
import glob
import math
import json
from collections import namedtuple

try:
    from PIL import PILLOW_VERSION
except:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)

try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

try:
    from annotation  import Annotation
    from labels      import labels, name2label, id2label, trainId2label, category2labels
except:
    print("Failed to find all Cityscapes modules")
    sys.exit(-1)

def printError(message):
    print('ERROR: ' + str(message))
    sys.exit(-1)

class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

def getColorEntry(val, args):
    if not args.colorized:
        return ""
    if not isinstance(val, float) or math.isnan(val):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

CsFile = namedtuple('csFile', ['city', 'sequenceNb', 'frameNb', 'type', 'type2', 'ext'])

def getCsFileInfo(fileName):
    baseName = os.path.basename(fileName)
    parts = baseName.split('_')
    parts = parts[:-1] + parts[-1].split('.')
    if not parts:
        printError('Cannot parse given filename ({}). Does not seem to be a valid Cityscapes file.'.format(fileName))
    if len(parts) == 5:
        csFile = CsFile(*parts[:-1], type2 = "", ext = parts[-1])
    elif len(parts) == 6:
        csFile = CsFile(*parts)
    else:
        printError('Found {} part(s) in given filename ({}). Expected 5 or 6.'.format(len(parts), fileName))

    return csFile

def getCoreImageFileName(filename):
    csFile = getCsFileInfo(filename)
    return "{}_{}_{}".format(csFile.city, csFile.sequenceNb, csFile.frameNb)

def getDirectory(fileName):
    dirName = os.path.dirname(fileName)
    return os.path.basename(dirName)

def ensurePath(path):
    if not path:
        return
    if not os.path.isdir(path):
        os.makedirs(path)

def writeDict2JSON(dictName, fileName):
    with open(fileName, 'w') as f:
        f.write(json.dumps(dictName, default = lambda o: o.__dict__, sort_keys = True, indent = 4))

if __name__ == "__main__":
    printError("Only for include, not executable on its own.")
