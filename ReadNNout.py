#NN inference mask postprocess and instance segmentation
from ctypes import *
from _ctypes import FreeLibrary

import os
from glob import glob
import shutil
import numpy as np
import pandas as pd

from PIL import Image
from skimage import io
from skimage import measure
from skimage import morphology
import scipy

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer as timer

from numba import jit

ncls = 8

def free_libs(libs):
  for lib in libs:
    if os.name=='nt': FreeLibrary(lib._handle)
    else: lib.dlclose(lib._handle)
    del lib

def showpair(a,b, lo=0, hi=255, title=None):
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)
  ax[0].imshow(a, vmin=lo, vmax=hi)
  ax[0].axis('off')
  
  ax[1].imshow(b, vmin=lo, vmax=hi)
  ax[1].axis('off')
  
  fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.95, bottom=0.05, left=0, right=1)
  if title: plt.title(title)
  plt.show()

  
def obj_area_analysis(mask, show=False):
  nicheID, area = np.unique(mask, return_counts=True)
  area[0] = 0
  #area_dict = dict(zip(nicheID, area))
  
  area_mask = area[mask].astype(np.uint32)
  
  return nicheID, area, area_mask

def label_mode(mask, labels):
  from scipy.stats import mode
  mask_mode, mask_count = mode(labels[mask])
  if len(mask_mode) == 1: return mask_mode[0]
  return -1

def cleanup_classification(img):
  h, w = img.shape
  maxdist = 100
  flooded = np.ascontiguousarray(np.zeros([2, h, w], dtype=np.uint8))
  flooded[0] = img
  
  libFlood = CDLL('flood.dll')
  c_neighborhood_floodfill = libFlood.neighborhood_floodfill
  c_neighborhood_floodfill.restype = None
  c_neighborhood_floodfill.argtypes = [POINTER(c_byte), c_int, c_int, c_int, c_int]
  
  t0 = timer()
  c_neighborhood_floodfill(flooded.ctypes.data_as(POINTER(c_byte)), w, h, ncls, maxdist)
  t1 = timer(); print("GPU floodfill: %.1fs" % (t1 - t0)); t0=t1;
  
  free_libs([libFlood])
  
  out = flooded[0]
  
  return out

def postprocess(img_file, background_class=1, fiber_classes=[2,3,4,5], mask_opening=10, fill_threshold=500, area_min=1000, area_max=30000, homogenize=True, show=False, save_output=True):
  if not os.path.isfile(img_file): raise NameError('Input image {} not found'.format(img_file))
  
  dir = os.path.dirname(img_file)
  if 'reg' in os.path.basename(img_file).lower():
    filename_root = [s for s in os.path.basename(img_file).split('_') if 'reg' in s.lower()][0].split('.')[0]
  else:
    filename_root = os.path.basename(img_file).split('.')[:-1].split('_')[0].split('.')[0]
  
  NN_raw = io.imread(img_file)
  print('{} loaded'.format(filename_root))

  NN_result = cleanup_classification(NN_raw)
  
  if show: showpair(NN_raw, NN_result, lo=0, hi=8)

  t0 = timer()
  NN_labeled = measure.label(NN_result, background=background_class)
  
  #measure object morphological properties
  NN_df = pd.DataFrame.from_dict(measure.regionprops_table(NN_labeled, intensity_image=NN_result, properties=('label', 'area', 'centroid', 'mean_intensity', 'major_axis_length', 'minor_axis_length')))
  NN_df = NN_df.set_index('label')
  print('Measure all objects: {:.1f}s'.format(timer()-t0))

  if show:
    hi = np.max(NN_labeled)
    showpair(NN_result.astype(np.uint32) * hi/ncls, NN_labeled, hi=hi)
  
  #isolate fiber classes from mask
  thresholded = np.isin(NN_result, list(fiber_classes))
  
  #perform binary morphological smoothing and fill holes
  if mask_opening:
    t0 = timer()
    thresholded = scipy.ndimage.morphology.binary_erosion(thresholded, iterations=mask_opening)
    thresholded = scipy.ndimage.morphology.binary_dilation(thresholded, iterations=mask_opening)
    print('Morphological opening on masks: {:.1f}s'.format(timer()-t0))
  
  t0 = timer()
  thresholded = morphology.remove_small_holes(thresholded, area_threshold=fill_threshold, connectivity=1)
  thresholded = scipy.ndimage.binary_fill_holes(thresholded)
  print('Fill holes in masks: {:.1f}s'.format(timer()-t0))
  
  if show: showpair(NN_result, thresholded, lo=0, hi=ncls)
  
  #generate labeled object image
  objmask, _ = scipy.ndimage.label(thresholded)
  print('Find objects: {:.1f}s'.format(timer()-t0))
  t0 = timer()
  
  #analyze object sizes
  objIDs, area, area_mask = obj_area_analysis(objmask, show)
  
  print('Filtering objects by size [{}-{}] px'.format(area_min, area_max))
  filtered = (area_min <= area) & (area <= area_max)
  filtered_objs = objIDs[filtered]
  filtered_area = area[filtered]
  filtered_object_lookup = objIDs * filtered
  filtered_objmask = filtered_object_lookup[objmask]
  print('{} objects passed filter'.format(len(filtered_objs)))
  
  #measure object morphological properties
  fiber_df = pd.DataFrame.from_dict(measure.regionprops_table(filtered_objmask, intensity_image=NN_result, properties=('area', 'centroid', 'major_axis_length', 'minor_axis_length')))
  fiber_df.index = filtered_objs
  #fiber_df.groupby('class').count()
  
  print('Measure filtered objects: {:.1f}s'.format(timer()-t0))
  
  #find mode class of each object
  if homogenize:
    print('Homogenizing object class labels')
    t0 = timer()
    class_table = np.zeros([len(objIDs), ncls], dtype=np.uint16)
    
    @jit(nopython=True)
    def count_object_classes(table, mask_in, labels_in):
      for ID, label in zip(mask_in.flat, labels_in.flat):
        table[ID, label] += 1
    
    count_object_classes(class_table, filtered_objmask, NN_result)
    
    obj_class_all = np.array([np.argmax(row) for row in class_table], dtype=np.uint8)
    obj_class_all[0] = 0
    
    obj_class = obj_class_all[filtered]
    fiber_df['class'] = obj_class
    
    print('Assign labels: {:.1f}s'.format(timer()-t0))
    
    #generate new class image
    new_class_img = obj_class_all[filtered_objmask]
    if show: showpair(NN_result, new_class_img, 0, ncls)
  
  if save_output:
    print('Saving outputs')
    io.imsave(os.path.join(dir, str(filename_root)+'_NN_class.png'), NN_result, check_contrast=False)
    io.imsave(os.path.join(dir, str(filename_root)+'_NN_labeled.png'), NN_labeled.astype(np.uint16), check_contrast=False)
    io.imsave(os.path.join(dir, str(filename_root)+'_fiber_area.png'), np.clip(area_mask // 10, 0, 65535).astype(np.uint16), check_contrast=False)
    io.imsave(os.path.join(dir, str(filename_root)+'_fiber_class.png'), new_class_img, check_contrast=False)
    io.imsave(os.path.join(dir, str(filename_root)+'_fiber_labeled.png'), filtered_objmask.astype(np.uint16), check_contrast=False)
    NN_df.to_csv(os.path.join(dir, str(filename_root)+'_NN_summary.csv'), index_label = 'objID')
    fiber_df.to_csv(os.path.join(dir, str(filename_root)+'_fiber_summary.csv'), index_label = 'fiberID')
    
    
  print('Finished processing {}'.format(os.path.basename(img_file)))

def getmaskvalues(cells_df, mask, x='x:x', y='y:y'):
  xs = np.clip(np.rint(cells_df.loc[:,x].to_numpy()).astype(np.int32), 0, mask.shape[1]-1)
  ys = np.clip(np.rint(cells_df.loc[:,y].to_numpy()).astype(np.int32), 0, mask.shape[0]-1)
  
  return mask[(ys, xs)]

def run_readNN(indir, csv_pattern, NN_img_pattern, x='x:x', y='y:y', NNdir='NNout', background_class=1, fiber_classes=[2,3,4,5], mask_opening=10, fill_threshold=500, area_min=1000, area_max=30000, show=False):
  if not os.path.exists(indir): raise NameError('Directory {} does not exist'.format(indir))
  
  nn_dir = os.path.join(indir, NNdir)
  if not os.path.exists(nn_dir):
    #raise NameError('Directory {} does not exist'.format(NNdir))
    os.makedirs(nn_dir)

    run = indir.find('_run')+1
    run = indir[run:run+5]

    src_images = sorted(glob('N:/Fiona 2020/NN Fiber Segmentation/out/{}/region*_out1_F322DRSAb01W95v3_31s2v3vW0ab-RG9a.png'.format(run)))
    for src in src_images:
      shutil.copyfile(src, os.path.join(nn_dir, os.path.basename(src)))
  
  fcs_dir = os.path.join(indir, 'processed', 'segm', 'segm-1', 'fcs', 'compensated')
  if not os.path.exists(fcs_dir): raise NameError('Processed FCS directory not found')

  csv_files = sorted(glob(os.path.join(fcs_dir, csv_pattern)))
  NN_imgs = sorted(glob(os.path.join(nn_dir, NN_img_pattern)))
  
  assert(len(NN_imgs) == len(csv_files))
  
  for file in NN_imgs:
    print('Generating object masks for {}'.format(file))
    postprocess(file, background_class=background_class, fiber_classes=fiber_classes, mask_opening=mask_opening, fill_threshold=fill_threshold, area_min=area_min, area_max=area_max, homogenize=True, show=show, save_output=True)

  for i, csv_file in enumerate(csv_files):
    csv_file = csv_files[i]
    region = 'region{}'.format(i+1)
    reg = 'reg{:03}'.format(i+1)
    
    print('Reading {}'.format(os.path.basename(csv_file)))
    nuclei_df = pd.read_csv(csv_file)

    NN_class_img = os.path.join(nn_dir, region+'_NN_class.png')
    if not os.path.isfile(NN_class_img): raise NameError('Object mask file {} not found'.format(NN_class_img))
    print('Loading NN clasification image {}'.format(os.path.basename(NN_class_img)))
    NN_class = io.imread(NN_class_img)
    
    obj_labeled_img = os.path.join(nn_dir, region+'_NN_labeled.png')
    if not os.path.isfile(obj_labeled_img): raise NameError('Object mask file {} not found'.format(obj_labeled_img))
    print('Loading object labels {}'.format(os.path.basename(obj_labeled_img)))
    obj_labeled = io.imread(obj_labeled_img)
    
    fiber_labeled_img = os.path.join(nn_dir, region+'_fiber_labeled.png')
    if not os.path.isfile(fiber_labeled_img): raise NameError('Fiber mask image {} not found'.format(fiber_labeled_img))
    print('Loading fiber labels {}'.format(os.path.basename(fiber_labeled_img)))
    fiber_labeled = io.imread(fiber_labeled_img)
    
    nuclei_df['objID'] = getmaskvalues(nuclei_df, obj_labeled, x=x, y=y)
    nuclei_df['fiberID'] = getmaskvalues(nuclei_df, fiber_labeled, x=x, y=y)
    nuclei_df = nuclei_df.join(pd.get_dummies(getmaskvalues(nuclei_df, NN_class, x=x, y=y), prefix='NN_class'))
    
    if 1: # save output
      idx = csv_file.rfind('_reg')+1
      idx += csv_file[idx:].find('_')
      outfile = csv_file[:idx]+'_NN_class.csv'
      
      print('Saving csv file:\n  {}'.format(outfile))
      nuclei_df.to_csv(outfile)
      print('Done!')

if __name__ == '__main__':
  tstart = timer()

  processed_dir = 'N:/CODEX processed' #Processed folder parent directory

  runs = ['Run_name']

  for run in runs:
    indir = os.path.join(processed_dir, run)
    csv_pattern = '*loose.csv'
    NN_img_pattern = '*out*.png'
    
    run_readNN(indir, csv_pattern, NN_img_pattern, NNdir='NNout10', mask_opening=10, show=False)
  
  print('Total time: {:.1f}s'.format(timer()-tstart))

'''

Example single run call:

indir = r'N:\CODEX processed\20190513_run08_20200416'
csv_pattern = '*loose.csv'
NN_img_pattern = '*out*.png'
x='x:x'
y='y:y'

run_readNN(indir, csv_pattern, NN_img_pattern, x, y)

Example batch call:

runs = ['20190513_run08_20200416', '20190517_run09_postclo_20200416',
       '20190523_run10_postveh_20200416', '20190610_run11_postclo_20200416',
       '20190802_run12_preveh_20200416', '20190816_run13_postclo_20200416',
       '20190820_run14_postveh_20200416', '20190905_run15_28monTA_20200416',
       '20191018_run17_postveh3_20200416', '20191028_run18_preveh2_20200416',
       '20191104_run19_postclo_20200416']

for run in runs:
    indir = r'N:/CODEX processed/'+run
    csv_pattern = '*loose.csv'
    NN_img_pattern = '*out*.png'
    x='x:x'
    y='y:y'
    num_classes = 7
    run_readNN(indir, csv_pattern, NN_img_pattern, x, y, NNdir)
'''
