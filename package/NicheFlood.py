import os
import numpy as np
import pandas as pd
from itertools import product

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (30,30)

import scipy.ndimage as nd
from skimage.morphology import watershed, square
from skimage import io
from ctypes import *
from _ctypes import FreeLibrary

def fill(df, colname, outdir, runs, X='x:x', Y='y:y', mask_dist=100, max_dist=1000):
  libSpaCE = CDLL("N:/Will's CODEX pipeline/SpaCE.dll")
  
  c_neighborhood_floodfill = libSpaCE.neighborhood_floodfill
  c_neighborhood_floodfill.restype = None
  c_neighborhood_floodfill.argtypes = [POINTER(c_byte), c_int, c_int, c_int, c_int]
  
  outdir_flood = os.path.join(outdir, 'NicheFlood')
  if not os.path.exists(outdir_flood): os.makedirs(outdir_flood)
  
  regs = np.unique(df.index.get_level_values('RegionID').values)
  for run, reg in product(runs, regs):
    positions = df.loc[pd.IndexSlice[:,run,reg],[X,Y,colname]].astype(np.int)
    if len(positions) > 0:
      w = np.max(positions[X])+1
      h = np.max(positions[Y])+1
      
      seed = np.ascontiguousarray(np.zeros([2, h, w], dtype=np.uint8))
      seed[0][positions[Y],positions[X]] = positions[colname].values.astype(np.int16)+1
      
      distance = nd.distance_transform_edt(seed[0] == 0)
      mask = distance
      mask[mask<mask_dist]=0
      mask[mask>0]=128

      seed[0] = mask
      seed[0][positions[Y],positions[X]] = positions[colname].values.astype(np.int16)+1
      
      nc = np.max(positions[colname].values.astype(np.int16))+1
      maxdist = max_dist
      c_neighborhood_floodfill(seed.ctypes.data_as(POINTER(c_byte)), w, h, nc, maxdist)
      
      #plt.imshow(seed[0], 'Spectral')
      
      print('Saving flooded tissue neighborhoods for '+run+' Region '+str(reg))
      io.imsave(os.path.join(outdir_flood, run+'_'+str(reg)+'nicheflood.tif'), seed[0].astype(np.uint8), check_contrast=False)
  FreeLibrary(libSpaCE._handle)
