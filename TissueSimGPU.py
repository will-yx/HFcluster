from ctypes import *
from _ctypes import FreeLibrary

from timeit import default_timer as timer
from glob import glob

import sys
import os
import csv

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.ndimage.filters import minimum_filter1d, maximum_filter1d
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes

libSpaCE = cdll.msvcrt
dll_path = os.path.join(os.path.dirname(__file__),'flood.dll')
if os.path.isfile(dll_path):
    if hasattr(os, 'add_dll_directory'):
      for p in os.getenv('PATH').split(';'):
        if p not in ['','.'] and os.path.isdir(p): os.add_dll_directory(p)
    libSpaCE = CDLL(dll_path)
else: print('Unable to find flood.dll')

def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib

def f2i(x):
  return int(float(x))

def showpair(a,b, lo=0, hi=255):
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
  ax[0].imshow(a, vmin=lo, vmax=hi)
  ax[0].axis('off')
  
  ax[1].imshow(b, vmin=lo, vmax=hi)
  ax[1].axis('off')
  
  fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.95, bottom=0.05, left=0, right=1)
  plt.show()

def loadfile(fname, drop_clusters, celltype_col='celltypeid', neighborhood_col='neighborhood'):
  with open(fname) as csvfile:
    rdr = csv.reader(csvfile, dialect='excel')
    header = next(rdr)
    cols = np.full(5, -1, np.int32)
    for i,h in enumerate(header):
      h = (h.split(':')[0]).lower()
      if h==celltype_col.lower(): 
        cols[0] = i
        print(h)
      elif h=='cellid': cols[1] = i
      elif h=='x': cols[2] = i
      elif h=='y': cols[3] = i
      elif h==neighborhood_col.lower(): cols[4] = i
      #elif h=='FileID': cols[4] = i
      #elif h=='Index in File' or h=='FileIDX': cols[5] = i
    
    if np.any(cols == -1):
      print('Error, some required parameters were not found!')
      print(header)
      print(cols)
      return []
    
    text = [[line[c] for c in cols] for line in rdr if line[0][0] is not '#']
    data = np.array([[0, f2i(l[0]), f2i(l[1]), f2i(l[2]), f2i(l[3]), f2i(l[4]), 0] for l in text if not f2i(l[0]) in drop_clusters], dtype=np.int32)
    
    return data

################################################################################
if __name__ == '__main__':
  drop_clusters = []
  with open(os.path.join(indir, 'celltypename2id.csv')) as f:
    drop_clusters.extend([int(r[1]) for r in csv.reader(f) if r[1].isnumeric() and r[0].lower() in ['unassigned','none','drop']])

  print('Dropping clusters: ', drop_clusters)

niche_radius = 133 # pixels

#iterations = 10
min_events = 1000000 # The simulator will loop over each cluster type until at least min_event positions have been simulated

cell_diameter = 18 # pixels, must be a multiple of 3
ds = 24 # downsample factor for determining valid tissue area

################################################################################
def simulate_region(inputfile, use_cached_data=1, celltype_col='celltypeid', neighborhood_col='neighborhood', drop_clusters=[], niche_radius=133, iterations=10, min_events=1000000, cell_diameter=18, ds=24):
  cell_diameter = 3 * int(cell_diameter//3)
  
  c_fixed_vs_fixed_count = libSpaCE.fixed_vs_fixed_count
  c_fixed_vs_fixed_count.restype = None
  c_fixed_vs_fixed_count.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_int), POINTER(c_int), c_int, c_int, c_int]

  G_fixed_origin_random_neighbors = libSpaCE.fixed_origin_random_neighbors_gpu
  G_fixed_origin_random_neighbors.restype = None
  G_fixed_origin_random_neighbors.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_int), POINTER(c_int), POINTER(c_ubyte), c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
  
  basename = os.path.splitext(inputfile)[0]
  data_cache_file = basename + '_cache.npz'
  
  print(inputfile)
  if use_cached_data and os.path.isfile(data_cache_file):
    data_cached = np.load(data_cache_file)
    data = data_cached['data'];
    data_cached.close()
  else:
    data = loadfile(inputfile, drop_clusters, celltype_col=celltype_col, neighborhood_col=neighborhood_col)
    np.savez_compressed(data_cache_file, data=data)
  
  if len(data) < 1:
    print('ERROR')
    return 1
  
  n_events = len(data)
  
  outdir = basename + '_out_FvR'
  if not os.path.exists(outdir): os.makedirs(outdir)
  
  data = sorted(data, key=lambda x: (x[1], x[2])) # sort by ClusterID, and by EventID within clusters
  n_full = len(data) # number of Events
  
  w = 0
  h = 0
  
  for d in range(n_full):
    x = data[d][3]
    y = data[d][4]
    if x > w: w = x
    if y > h: h = y
  
  w+=1
  h+=1
  
  wds = 1 + (w-1) // ds
  hds = 1 + (h-1) // ds
  mask_ds = np.zeros([hds, wds], dtype=np.uint8)
  
  for d in range(n_full):
    x = data[d][3]
    y = data[d][4]
    mask_ds[y//ds, x//ds] += 1
  
  c = -1
  cid = -1
  classes = []
  firstidx = []

  min_eid = 2**31
  max_eid = 0

  n = 0
  for d in range(n_full):
    if data[d][1] in drop_clusters: continue
    
    eid = data[d][2]
    if eid < min_eid: min_eid = eid
    if eid > max_eid: max_eid = eid
    
    if data[d][1] == cid:
      data[d][0] = c
    else:
      cid = data[d][1]
      firstidx.extend([n])
      classes.extend([cid])
      c+=1
      data[d][0] = c
    n+=1
    
  firstidx.extend([n])

  firstidx = np.ascontiguousarray(firstidx, dtype=np.int32)

  counts = np.diff(firstidx)
  for i,c in enumerate(counts):
    print('  Cluster {:02d}\t{}'.format(classes[i], c))


  nc = len(classes)

  print('Number of events: ', n)
  print('EventID range: [{},{}]'.format(min_eid, max_eid))
  print('width: {}, height: {}'.format(w, h))
  print('Number of clusters: ', nc)

  print('ClusterIDs:')
  print(classes)

  # index in data, cluster index, X, Y
  #cdata = np.ascontiguousarray([[i, d[0], d[3], d[4]] for i,d in enumerate(data) if not d[1] in drop_clusters], dtype=np.int32)
  cdata = np.ascontiguousarray([[d[5], d[0], d[3], d[4]] for i,d in enumerate(data) if not d[1] in drop_clusters], dtype=np.int32)

  niche       = np.ascontiguousarray(np.zeros([n, nc], dtype=np.float32))
  niche_sim   = np.ascontiguousarray(np.zeros([n, nc], dtype=np.float32))

  mindist     = np.ascontiguousarray(np.zeros([n, nc], dtype=np.float32))
  mindist_sim = np.ascontiguousarray(np.zeros([n, nc], dtype=np.float32))

  eids = [data[cdata[d,0]][2] for d in range(n)]
  
  mask_ds_processed = binary_dilation(mask_ds, iterations=2)
  mask_ds_processed = binary_fill_holes(mask_ds_processed)
  mask_ds_processed = binary_erosion(mask_ds_processed, border_value=1, iterations=2)
  
  mask_ds_processed = np.ascontiguousarray(mask_ds_processed, dtype=np.uint8)
  
  #showpair(mask_ds, mask_ds_processed, 0, 2)
  
  t0 = timer()
  c_fixed_vs_fixed_count(niche.ctypes.data_as(POINTER(c_float)), mindist.ctypes.data_as(POINTER(c_float)), cdata.ctypes.data_as(POINTER(c_int)), firstidx.ctypes.data_as(POINTER(c_int)), nc, n, niche_radius)
  t1 = timer(); print("computed niche for event data: %.2fs" % (t1 - t0)); t0=t1;
  
  for it in range(iterations):
    G_fixed_origin_random_neighbors(niche_sim.ctypes.data_as(POINTER(c_float)), mindist_sim.ctypes.data_as(POINTER(c_float)), cdata.ctypes.data_as(POINTER(c_int)), firstidx.ctypes.data_as(POINTER(c_int)), mask_ds_processed.ctypes.data_as(POINTER(c_ubyte)), ds, nc, n, w, h, niche_radius, cell_diameter, min_events, it)
    t1 = timer(); print("computed random niche (%d of %d): %.2fs" % (it+1, iterations, t1 - t0)); t0=t1;

  free_libs([libSpaCE])
    
  niche_representative   = np.zeros([nc, nc], dtype=np.float32)
  mindist_representative = np.zeros([nc, nc], dtype=np.float32)

  for c1 in range(nc):
    for c2 in range(nc):
      niche_representative[  c1,c2] = np.mean([niche_sim[e,c2] for e in range(firstidx[c1],firstidx[c1+1])]) / iterations
      mindist_representative[c1,c2] = np.mean([mindist_sim[e,c2] for e in range(firstidx[c1],firstidx[c1+1])]) / iterations
  
  niche_expected   = np.zeros([n, nc], dtype=np.float32)
  mindist_expected = np.zeros([n, nc], dtype=np.float32)
  niche_relative   = np.zeros([n, nc], dtype=np.float32)
  mindist_relative = np.zeros([n, nc], dtype=np.float32)
  
  for c1 in range(nc):
    for e in range(firstidx[c1],firstidx[c1+1]):
      for c2 in range(nc):
        niche_expected[e][c2] = float(niche_sim[e][c2]) / iterations
        rep = niche_representative[c1,c2]
        niche_relative[e][c2] = float(niche[e][c2]) / rep if rep > 0 else -niche[e][c2]
        
  for c1 in range(nc):
    for e in range(firstidx[c1],firstidx[c1+1]):
      for c2 in range(nc):
        mindist_expected[e][c2] = float(mindist_sim[e][c2]) / iterations
        rep = mindist_representative[c1,c2]
        mindist_relative[e][c2] = mindist[e][c2] / rep if rep > 0 else -mindist[e][c2]
  
  
  out_individual_niche_file = outdir + '/individual_niche.csv'
  out_individual_dist_file  = outdir + '/individual_dist.csv'
  
  out_individual_relative_niche_file = outdir + '/individual_relative_niche.csv'
  out_individual_relative_dist_file  = outdir + '/individual_relative_dist.csv'
  
  out_average_niche_file = outdir + '/average_niche.csv'
  out_average_dist_file  = outdir + '/average_dist.csv'
  
  out_average_expected_niche_file = outdir + '/average_expected_niche.csv'
  out_average_expected_dist_file  = outdir + '/average_expected_dist.csv'
  
  out_average_relative_niche_file = outdir + '/average_relative_niche.csv'
  out_average_relative_dist_file  = outdir + '/average_relative_dist.csv'
  
  with open(out_individual_niche_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['CellID', 'X', 'Y'] + classes[:])
    for r,row in enumerate(niche): writer.writerow([eids[r], cdata[r,2], cdata[r,3]] + list(row))

  with open(out_individual_dist_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['CellID', 'X', 'Y'] + classes[:])
    for r,row in enumerate(mindist): writer.writerow([eids[r], cdata[r,2], cdata[r,3]] + list(row))

  with open(out_individual_relative_niche_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['CellID', 'X', 'Y'] + classes[:])
    for r,row in enumerate(niche_relative): writer.writerow([eids[r], cdata[r,2], cdata[r,3]] + list(row))

  with open(out_individual_relative_dist_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['CellID', 'X', 'Y'] + classes[:])
    for r,row in enumerate(mindist_relative): writer.writerow([eids[r], cdata[r,2], cdata[r,3]] + list(row))
  
  niche_avg            = np.zeros([nc, 1+nc], dtype=np.float32)
  mindist_avg          = np.zeros([nc, 1+nc], dtype=np.float32)
  niche_expected_avg   = np.zeros([nc, 1+nc], dtype=np.float32)
  mindist_expected_avg = np.zeros([nc, 1+nc], dtype=np.float32)
  niche_relative_avg   = np.zeros([nc, 1+nc], dtype=np.float32)
  mindist_relative_avg = np.zeros([nc, 1+nc], dtype=np.float32)
  
  for j in range(nc):
    niche_relative_avg[j,0] = niche_expected_avg[j,0] = niche_avg[j,0] = classes[j]
    mindist_relative_avg[j,0] = mindist_expected_avg[j,0] = mindist_avg[j,0] = classes[j]
  
  for c1 in range(nc):
    for d in range(firstidx[c1],firstidx[c1+1]):
      for c2 in range(nc):
        niche_avg[  c1,1+c2] += niche[d, c2]
        mindist_avg[c1,1+c2] += mindist[d, c2]

        niche_expected_avg[  c1,1+c2] += niche_expected[d,c2]
        mindist_expected_avg[c1,1+c2] += mindist_expected[d,c2]
        
        niche_relative_avg[  c1,1+c2] += niche_relative[d,c2]
        mindist_relative_avg[c1,1+c2] += mindist_relative[d,c2]
    
    n1 = firstidx[c1+1] - firstidx[c1]
    for c2 in range(nc):
      niche_avg[  c1,1+c2] /= n1
      mindist_avg[c1,1+c2] /= n1
      
      niche_expected_avg[  c1,1+c2] /= n1
      mindist_expected_avg[c1,1+c2] /= n1
      
      niche_relative_avg[  c1,1+c2] /= n1
      mindist_relative_avg[c1,1+c2] /= n1
  
  header = ['cellTypeID'] + classes[:]
  with open(out_average_niche_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(header)
    for row in niche_avg:
      writer.writerow(row)
  
  with open(out_average_dist_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(header)
    for row in mindist_avg:
      writer.writerow(row)
  
  with open(out_average_expected_niche_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(header)
    for row in niche_expected_avg:
      writer.writerow(row)
  
  with open(out_average_expected_dist_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(header)
    for row in mindist_expected_avg:
      writer.writerow(row)
  
  with open(out_average_relative_niche_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(header)
    for row in niche_relative_avg:
      writer.writerow(row)
  
  with open(out_average_relative_dist_file, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(header)
    for row in mindist_relative_avg:
      writer.writerow(row)
  
  return 0

def main():
  #run = sys.argv[1] if len(sys.argv) > 1 else 'run12'
  inputfiles = []
  for run in ['20190517_run09_postclo', '20190523_run10_postveh', '20190610_run11_postclo', '20190802_run12_preveh', '20190816_run13_postclo', '20190820_run14_postveh', '20191018_run17_postveh3', '20191028_run18_preveh2', '20191104_run19_postclo']:
    inputfiles=inputfiles+sorted(glob(indir+'/'+run+'/*run*compensated.csv'))
  print(inputfiles)

  for inputfile in inputfiles:
    status = simulate_region(inputfile)
    if status != 0: break

if __name__ == '__main__':
    main()








