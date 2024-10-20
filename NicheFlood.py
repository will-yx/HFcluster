import os
import numpy as np
import pandas as pd
from itertools import product

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (30,30)

import scipy.ndimage as nd
from skimage.morphology import square
from skimage.segmentation import watershed
from skimage import io, measure, filters
from ctypes import *
from _ctypes import FreeLibrary

from skimage import graph
import networkx as nx
import seaborn as sns

libSpaCE = cdll.msvcrt
dll_path = os.path.join(os.path.dirname(__file__),'flood.dll')
if os.path.isfile(dll_path):
    if hasattr(os, 'add_dll_directory'):
      for p in os.getenv('PATH').split(';'):
        if p not in ['','.'] and os.path.isdir(p): os.add_dll_directory(p)
    libSpaCE = CDLL(dll_path)
else: print('Unable to find flood.dll')

def fill(df, colname, outdir, runs, X='x:x', Y='y:y', NNmask=False, mask_dist=100, max_dist=1000):  
  c_neighborhood_floodfill = libSpaCE.neighborhood_floodfill
  c_neighborhood_floodfill.restype = None
  c_neighborhood_floodfill.argtypes = [POINTER(c_byte), c_int, c_int, c_int, c_int]
  
  outdir_flood = os.path.join(outdir, 'NicheFlood')
  if not os.path.exists(outdir_flood): os.makedirs(outdir_flood)
  
  tissue_ids = sorted(list(set([(run,reg) for run, reg in zip(df.index.get_level_values('RunID'),df.index.get_level_values('RegionID'))])))
  for run, reg in tissue_ids:
    positions = df.loc[pd.IndexSlice[:,run,reg],[X,Y,colname]].astype(np.int)

    if len(positions) > 0:
      w = np.max(positions[X])+1
      h = np.max(positions[Y])+1

      seed = np.ascontiguousarray(np.zeros([2, h, w], dtype=np.uint8))
    
      if NNmask:
        NNdir = 'N:/CODEX processed/{}/NNout_2'.format(run)
        mask = io.imread(os.path.join(NNdir, 'region{}_NN_class.png'.format(reg)))[0:h,0:w]
        mask[mask != 1] = 0
      else:
        seed[0][positions[Y],positions[X]] = positions[colname].values.astype(np.int16)+1
        distance = nd.distance_transform_edt(seed[0] == 0)
        mask = distance
        mask[mask<mask_dist] = 0
      
      mask[mask>0] = 128
      
      seed[0] = mask
      seed[0][positions[Y],positions[X]] = positions[colname].values.astype(np.int16)+1
	  
      nc = np.max(positions[colname].values.astype(np.int16))+1
      maxdist = max_dist
      c_neighborhood_floodfill(seed.ctypes.data_as(POINTER(c_byte)), w, h, nc, maxdist)
      
      #plt.imshow(seed[0], 'Spectral')
      
      print('Saving flooded tissue neighborhoods for '+run+' Region '+str(reg))
      io.imsave(os.path.join(outdir_flood, run+'_'+str(reg)+'nicheflood.png'), seed[0].astype(np.uint8), check_contrast=False)
  FreeLibrary(libSpaCE._handle)

def neighborhood_adjacency(flood_file, background=0, connectivity=2, show=True):
    flood_img = io.imread(flood_file)
    label_img = measure.label(flood_img, background=background, connectivity=connectivity)

    nicheID, area = np.unique(label_img, return_counts=True)
    area_dict=dict(zip(nicheID, area))
    area_dict.pop(0, None)

    edge_img = filters.sobel(flood_img)
    #segmentation.find_boundaries(label_img, background=0, connectivity=2).astype(np.uint8)

    L = graph.rag_boundary(label_img, edge_img)
    L.remove_node(0)
    if show: #must be True to populate 'centroid' parameter in nodes
        graph.show_rag(label_img, L, flood_img)

    n = len(L.nodes)
    adjacency = np.zeros((n,n))

    obj_class = []
    obj_size = []
    for node in L.nodes():
        for neighbor in nx.neighbors(L, node):
            adjacency[node-1, neighbor-1] = 1
        L.nodes[node]['class'] = flood_img[L.nodes[node]['centroid']]
        L.nodes[node]['area'] = area_dict[node]
        obj_class += [L.nodes[node]['class']-1]
        obj_size += [L.nodes[node]['area']]
    class_adjacency = pd.DataFrame(adjacency, index=L.nodes(), columns=L.nodes(), dtype=np.int16)
    class_adjacency['index_class']=obj_class
    class_adjacency = class_adjacency.groupby('index_class').sum().T
    class_adjacency['index_class']=obj_class
    class_adjacency['area']=obj_size
    del [obj_class, obj_size, adjacency]
    return L, class_adjacency

def batch_adj(tissues, flood_path, pattern='nicheflood.png', show=True):
	adjacency_graphs = {}
	neighborhood_adjacencies = pd.DataFrame()

	for tissue_id in tissues:
		print(tissue_id)
		flood_file = os.path.join(flood_path, tissue_id+pattern)
		print(flood_file)
		assert(os.path.isfile(flood_file))
		g, class_adjacency = neighborhood_adjacency(flood_file, show=show)
		class_adjacency['tissue']=str(tissue_id)
		adjacency_graphs[tissue_id]=g
		neighborhood_adjacencies = neighborhood_adjacencies.append(class_adjacency)
		
	return adjacency_graphs, neighborhood_adjacencies