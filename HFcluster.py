"""Read and Cluster CODEX data"""
import os
import glob
import pandas as pd
import numpy as np
import csv
from itertools import chain 
from itertools import product
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (30,30)
import seaborn as sns
import plotly.express as px

import scanpy as sc

from kneed import KneeLocator
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.semi_supervised import LabelSpreading

sc.settings.verbosity = 2             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80)

import warnings
warnings.filterwarnings('ignore', category = Warning)

import dill
from my_fcswrite import write_fcs

import anndata as ad

def preprocess(path, run, subdir, filename_pattern, DNAfilter, DNAChannel, ManualDNAthreshold, DRAQ5Channel, Zremoval, Zfilter, Z): 
    #load and filter data
	dir = os.path.join(path,run,subdir)

    #load data in dir
	csvs = [i for i in sorted(glob.glob(os.path.join(dir,filename_pattern)))]
	print("Loading "+str(run))
	print(csvs)
	if filename_pattern.endswith('tsv'):
		q_data = pd.concat([pd.read_csv(f, sep="\t") for f in csvs], keys=[run]*len(csvs), names=['run', 'reg_cellID'])
	else:
		q_data = pd.concat([pd.read_csv(f) for f in csvs], keys=[run]*len(csvs), names=['run', 'reg_cellID'])

	cellcount = len(q_data)
	print(str(cellcount)+" events found in run "+str(run))

	# Cell Filters
	if Zremoval:
		q_data = q_data.loc[np.all([q_data[Z] <=Zfilter[1], q_data[Z] >=Zfilter[0]], axis=0),:]

	if DNAfilter:
		if ManualDNAthreshold:
			threshold = ManualDNAthreshold #CAUTION: manual overide for DNA threshold
			print('Manual DNA threshold applied: ' + str(threshold))
		else:
			data = q_data.loc[:,DNAChannel].values
			data = sorted(data, reverse=False)
			x = range(1,len(data)+1)
			kn = KneeLocator(x, data, curve='convex', direction='increasing', S=200, interp_method='interp1d')
			threshold = data[kn.knee]
			print('Automatic DNA threshold: ' + str(threshold))
		q_data = q_data.loc[q_data[DNAChannel] > threshold,:]

	rm_evts = cellcount-len(q_data)
	return q_data, rm_evts

def rescale(df, stains=None, noise=0, q=0.9, center=0.5, r=10000):
        if stains == None:
                stains = [i for i in df.columns if i.startswith('cyc')]
        rescaled = df.copy()
        if noise: 
            print(f'Suppressing low-end noise: {noise}')
            rescaled.loc[:,stains]=df.loc[:,stains].apply(lambda x: x-noise, axis=0)
            rescaled[rescaled<0]=0
        rescaled.loc[:,stains]=rescaled.loc[:,stains].apply(lambda x: (r*(x-x.quantile(q=center))/(x.quantile(q=q)-x.quantile(q=center))), axis=0)
        return rescaled

def loadData(runs, path, subdir, filename_pattern, Z_col='z:z', noise=0, scale=False, q=0.9, center=0.5, scale_val=10000, DNAfilter=True, ManualDNAthreshold=1, DNAChannel=None, DRAQ5Channel=None, Zremoval=False, Zfilter=[]):
	data=[]
	for run in runs:
		tmp, filtercount = preprocess(path, run, subdir, filename_pattern, DNAfilter, DNAChannel, ManualDNAthreshold, DRAQ5Channel, Zremoval, Zfilter, Z_col) 
		if scale:
			tmp = rescale(tmp, q=q, center=center, r=scale_val, noise=noise)
		data.append(tmp)
		print(filtercount,' events were removed in ',run,' ',len(tmp),' events remain')  
	data_all = pd.concat(data)
	data_all = data_all.reset_index()
	data_all['cellID'] = range(len(data_all))
	data_all['run_region'] = data_all['run'].astype(str)+'_'+data_all['region:region'].astype(int).astype(str)
	data_all=data_all.set_index('cellID')
	print(str(len(data_all))+" events for clustering")
	return data, data_all

def conv_adata(df):
    channels = [c for c in df.columns if c.startswith('cyc')]
    meta_data_cols = [c for c in df.columns if c not in channels]
    
    adata = ad.AnnData(df[channels])
    
    ch_marker = adata.var.index.str.split(':',expand=True).to_frame().reset_index(drop=True).loc[:,1]
    ch_cyc_ch = adata.var.index.str.split(':',expand=True).to_frame().reset_index(drop=True).loc[:,0].str[:-1]
    
    adata.var['cycle'] = ch_cyc_ch.str.split('_',expand=True).loc[:,0].str[-3:].astype(int).values
    adata.var['fl_channel'] = ch_cyc_ch.str.split('_',expand=True).loc[:,1].str[-3:].astype(int).values
    adata.var['marker'] = [i[:-len(i.split('_')[-1])-1] if i.split('_')[-1] in ['interior','border'] else i for i in ch_marker]
    adata.var['quant_type'] = [i.split('_')[-1]  if i.split('_')[-1] in ['interior','border'] else 'full' for i in ch_marker]
    adata.var['q_t'] = [i.split('_')[-1][0]  if i.split('_')[-1] in ['interior','border'] else 'f' for i in ch_marker]
    adata.var['marker_q'] = adata.var['marker'] + '_' + adata.var['q_t']
    adata.var = adata.var.reset_index().set_index('marker_q')
    adata.var_names_make_unique()
    
    meta_data = df[meta_data_cols]
    meta_data.columns = meta_data.columns.str.split(':',expand=True).to_frame()[0].values
    adata.obs = meta_data

    return adata

def transfer_meta(adata, adata_s):
    for i in adata_s.obs.keys():
        if i not in adata.obs.keys():
            adata.obs[i] = adata_s.obs[i].values
    adata.uns = adata_s.uns
    adata.obsm = adata_s.obsm
    #adata.varm = adata_s.varm
    #adata.varp = adata_s.varp
    return adata

def generateTissueArray(adata, X='x', Y='y', shift = 10000):
    run2idx = {r:i for i,r in enumerate(sorted(np.unique(adata.obs['run'])))}
    
    xshift = np.asarray([(n)*shift for n in adata.obs['region'].values])
    x = adata.obs[X].astype(int).values + xshift
    
    yshift = np.asarray([run2idx[n]*shift for n in adata.obs['run'].values])
    y = adata.obs[Y].astype(int).values + yshift

    adata.obs['array_x'] = x
    adata.obs['array_y'] = y

    adata.obsm['spatial'] = np.array(list(map(list,zip(adata.obs['x'].astype(int),adata.obs['y'].astype(int)))))
    adata.obsm['tissue_array'] = np.array(list(map(list, (zip(adata.obs['array_x'].astype(int),adata.obs['array_y'].astype(int))))))

    return adata

import plotly.express as px
def plotlyTissueArray(adata, color, palette='hsv', template='plotly_dark', log=False, width=900, height=600, inverty=True, **kwargs):
    if color in adata.obs.columns:
        c_data=adata.obs[color]
    elif color in adata.var_names:
        c_data=adata[:,color].X.flatten()
        if log:
            c_data=np.log(c_data+1)
    else:
        raise NameError(f'parameter {color} not found in .obs or .var') 
    fig = px.scatter(x=adata.obs['array_x'], y=adata.obs['array_y'], color=c_data,
                     color_discrete_sequence=palette,
                     width=width, height=height, template=template, render_mode='webgl', **kwargs)
    if inverty:
        fig.update_yaxes(autorange="reversed")
    fig.show()

def loadCelltypeNames(adata, file, colname='leiden'):
    name_dict = pd.read_csv(file,header=None, index_col=0).to_dict()[1]
    adata.obs['celltypename'] = [name_dict[int(i)] for i in adata.obs[colname]]
    adata.obs['celltype_id'] = adata.obs['celltypename'].astype('category').cat.codes
    adata.obs.groupby('celltypename')['celltype_id'].mean().astype(int).to_csv('celltypename2id.csv')
    return adata

def get_cluster_proportions(adata,
                            cluster_key="cluster_final",
                            sample_key="replicate",
                            drop_values=None, prop=True):
    """
    Input
    =====
    adata : AnnData object
    cluster_key : key of `adata.obs` storing cluster info
    sample_key : key of `adata.obs` storing sample/replicate info
    drop_values : list/iterable of possible values of `sample_key` that you don't want
    
    Returns
    =======
    pd.DataFrame with samples as the index and clusters as the columns and 0-100 floats
    as values
    """
    
    adata_tmp = adata.copy()
    sizes = adata.obs.groupby([cluster_key, sample_key]).size()
    if prop: 
        props = sizes.unstack().apply(lambda x: 100 * x / x.sum())
    else: props = sizes.unstack()
    #props = props.pivot(columns=sample_key, index=cluster_key).T
    #props.index = props.index.droplevel(0)
    props.fillna(0, inplace=True)
    
    if drop_values is not None:
        for drop_value in drop_values:
            props.drop(drop_value, axis=0, inplace=True)
    return props


def plot_cluster_proportions(cluster_props, 
                             cluster_palette=None,
                             xlabel_rotation=0, **kwargs): 
    fig, ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor("white")
    
    cmap = None
    if cluster_palette is not None:
        cmap = sns.palettes.blend_palette(
            cluster_palette, 
            n_colors=len(cluster_palette), 
            as_cmap=True)
   
    cluster_props.plot(
        kind="bar", 
        stacked=True, 
        ax=ax, 
        legend=None, 
        colormap=cmap, **kwargs
    )
    
    ax.legend(bbox_to_anchor=(1.01, 1), frameon=False, title="Cluster")
    sns.despine(fig, ax)
    ax.tick_params(axis="x", rotation=xlabel_rotation)
    ax.set_xlabel(cluster_props.index.name.capitalize())
    ax.set_ylabel("Proportion")
    fig.tight_layout()
    
    return fig

def find_highconf(ad, markers, sensitivity=10): 
    ts = []
    for m in markers:
        vals = np.sort(ad[:,m].X.flatten())
        kn=KneeLocator(range(len(vals)), vals, curve='convex', direction='increasing', S=sensitivity, interp_method='interp1d')
        ts += [vals[kn.knee]]
    
    hf = ad[:,markers].X - np.array(ts)
    hf[hf<0]=0

    ad.layers['preHF'] = ad.X
    ad.X = hf

    ad.uns['HF_thresh']=dict(zip(markers,ts))
    return ad
    
def load_xkcd():
    return pd.read_csv(f"{os.path.dirname(__file__)}/xkcd_hexcol.csv").columns.values

from skimage.io import imread

def load_mask(adata, mask_name, path, runs, pattern='reg*png', dtype=bool):
    masks={}
    print(mask_name)
    for run in runs:
        print(run)
        maskdir = os.path.join(path,run,mask_name)
        maskfiles = sorted(glob.glob(os.path.join(maskdir, pattern)))
        print(maskfiles)
        masks[run] = [imread(img).astype(dtype) for img in maskfiles]
    
    mask_val = [masks[run][r-1][y,x] for run,r,x,y in zip(adata.obs['run'], adata.obs['region'].astype(int),adata.obs['x'].astype(int),adata.obs['y'].astype(int))]
    adata.obs[mask_name] = mask_val
    return adata
    
def newDir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)
        
def propagateLabels(adata, markers, labels, kernel='knn', n_neighbors=5, alpha=0.8, n_jobs=-1, max_iter=150, **kwargs):
	vals = adata.loc[:,markers].X

	label_spread = LabelSpreading(kernel=kernel, n_neighbors=n_neighbors, alpha=alpha, n_jobs=n_jobs, max_iter=max_iter, **kwargs)
	label_spread.fit(vals, labels)

	output_labels = label_spread.transduction_
    
	return output_label_array