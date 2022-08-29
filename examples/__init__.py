import sys, os
add_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(add_path)

from .simple_stgcn.STGCN_example import experiment