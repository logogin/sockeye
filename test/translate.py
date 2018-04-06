import sockeye
import sockeye.utils
import sockeye.translate
import sockeye.inference

import mxnet as mx
import numpy as np

import os
import sys


def translate():
    sys.argv.extend(['-m', '../sockeye/tutorials/seqcopy/seqcopy_model',
                     #'--use-cpu',
                     '--input', 'in.txt'])
    sockeye.translate.main()


if __name__ == '__main__':
    print(os.path.abspath(sockeye.__file__))
    print(os.path.abspath(mx.__file__))

    translate()
