"""Import bootstrap for the recovered semantic engine; no server run tree."""
import os,sys
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
from paper_paths import REPO,WEIGHTS
os.environ.setdefault('PIERS_WEIGHTS_ROOT',str(WEIGHTS))
sys.path.insert(0,str(REPO/'third_party/effective_omnisam'))
import utils.sam2_bootstrap
import torch,numpy as np,cv2
import validation as v
from utils.tools import merge_width_scatter_np
from i1_dense_runtime import install_fixed_memory as dense_memory,verify_i2,I2
from seam_core import choose,base_coverage
from seam_cuda import risk_cuda,fuse_cuda
