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
from sklearn.semi_supervised import label_propagation

sc.settings.verbosity = 2             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80)

import warnings
warnings.filterwarnings('ignore', category = Warning)

import dill
from my_fcswrite import write_fcs

def preprocess(f, path, runs, subdir, filename_pattern, Z, RemoveHighBlankCells, DNAfilter, DNAChannel, ManualDNAthreshold, DRAQ5Channel, Zremoval, Zfilter, cols): #load and filter data
    
	run = runs[f] + '/'
	dir = path + run + subdir
	os.chdir(dir)

    #load data in dir
	all_csvs = [i for i in sorted(glob.glob(filename_pattern))]
	print("Loading "+str(runs[f]))
	print(all_csvs)
	regID = []
	for i in range(len(all_csvs)):
		regID.append(i+1)
	if filename_pattern.endswith('tsv'):
		q_data = pd.concat([pd.read_csv(f, sep="\t") for f in all_csvs], keys=regID, names=['RegionID', 'RegCellID'])
	else:
		q_data = pd.concat([pd.read_csv(f) for f in all_csvs], keys=regID, names=['RegionID', 'RegCellID'])

	allcellcount = len(q_data)
	print(str(allcellcount)+" events found in run "+str(runs[f]))
	if f==0:
		cols=q_data.columns
	else:
		q_data.columns=cols

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
		if 0:   
			data = q_data.loc[:,DRAQ5Channel].values
			data = sorted(data, reverse=True)
			x = range(1,len(data)+1)
			kn = KneeLocator(x, data, curve='convex', direction='decreasing', S=100, interp_method='interp1d')
			threshold = data[kn.knee]
			print('Automatic DRAQ5 threshold: ' + str(threshold))
			q_data = q_data.loc[q_data[DRAQ5Channel] > threshold,:]

	if RemoveHighBlankCells:
		max_blanks=[i for i in q_data.columns if 'blank' in i.lower() or 'empty' in i.lower()] 
		#print(max_blanks)
		for i in max_blanks:
			data = q_data.loc[:,i].values
			if max(data)==0:
				continue
			else:
				data = sorted(data, reverse=True)
				x = range(1,len(data)+1)
				kn = KneeLocator(x, data, curve='convex', direction='decreasing', S=1, interp_method='interp1d')
				threshold = data[kn.knee]
				#print(i, threshold)
				q_data = q_data.loc[q_data[i] <= threshold,:]
	rm_evts = allcellcount-len(q_data)
	return q_data, cols, rm_evts

def rescale(df, stains=None, q=0.9):
        if stains == None:
                stains = [i for i in df.columns if i.startswith('cyc')]
        rescaled = df.copy()
        rescaled.loc[:,stains]=df.loc[:,stains].apply(lambda x: (x-x.median())/(x.quantile(q=q)-x.median())*10000, axis=0)
        return rescaled
		

# Elbow based thresholding of high confidence cells
def find_highconf(dfs, runs, markers, sensitivity): 
	print("Automatic thresholding for high confidence cells...")
	sub_data=[]
	for df in dfs:
		x = range(1, len(df)+1)
		tmp = elbow_sub(df.copy(), markers, sensitivity, x)
		sub_data.append(tmp)
	print('High confidence cells identified')
	sub_data_all = pd.concat(sub_data, keys=runs, names=['RunID'])
	return sub_data_all

def find_highconf_normalized(df, markers, sensitivity): 
        print("Automatic thresholding for high confidence cells...")
        x = range(1, len(df)+1)
        tmp = elbow_sub(df.copy(), markers, sensitivity, x)
        print('High confidence cells identified')
        return tmp

# Elbow based background removal
def elbow_sub(df, markers, sensitivity, x): 
    for i in markers:
        marker_data = df.loc[:,i].values
        marker_data = sorted(marker_data, reverse=True)
        kn = KneeLocator(x, marker_data, curve='convex', direction='decreasing', S=sensitivity, interp_method='interp1d')
        threshold = marker_data[kn.knee]
        print(i, threshold)
        df[i]=df.loc[:,i].transform(lambda x: x-threshold)
        df[i]=[value if value > 0 else 0 for value in df[i].values]
    #df[df<0]=0
    return df

# Scanpy clustering
def scluster(df, markers, log = True, var_maxmean = 5, var_mindisp = -1.5, n_neighbors=25, pcs=20, metric='euclidean', louvain_res=1.0, verbose=1, **kwargs):
	var_params = kwargs.get('var_params',{})
	pca_params = kwargs.get('pca_params',{})
	umap_params = kwargs.get('umap_params',{})
	louvain_params = kwargs.get('louvain_params',{})
	adata = sc.AnnData(df.loc[:,markers])
	adata = adata[:, markers]
	#sc.pl.highest_expr_genes(adata, n_top=10)
    #the two lines below normalize the data by log tr and min-max just before clustering
    #for any optional normalization look up up the scanpy doc in particular anndata
	if log: sc.pp.log1p(adata)
	sc.pp.scale(adata, max_value = 10)
	#adata.raw = adata
	#sc.pp.highly_variable_genes(adata, min_mean=0, max_mean=var_maxmean, min_disp=var_mindisp, **var_params)
	#if verbose:
	#	sc.pl.highly_variable_genes(adata)
    # PCA, umap, Louvain clustering
	#adata = adata[:, adata.var['highly_variable']]
	sc.tl.pca(adata, n_comps=25, svd_solver='arpack', **pca_params)
	if verbose:
		sc.pl.pca_variance_ratio(adata, log=False)
	sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=pcs, metric=metric)
	sc.tl.umap(adata, **umap_params)
	sc.tl.louvain(adata, resolution=louvain_res, **louvain_params)
	louvain = adata.obs.louvain.values
	if verbose==2:
		sc.pl.umap(adata, color=['louvain'], cmap=plt.cm.get_cmap('jet'))
    # Violin plot of Louvain clusters
	if verbose==2:
		ax = sc.pl.stacked_violin(adata, markers, groupby='louvain', rotation=90)
		adata.obs.louvain.value_counts()
	return louvain, adata

# Manual cluster merging
def manual_merge(g, merge_idx, X):
    rowidx = np.array(g.dendrogram_row.reordered_ind).astype(np.int) + 1
    len(X)
    if sum(merge_idx) != len(X):
        print('Index is {} for {} clusters. Re-check your merge index!'.format(sum(merge_idx), len(X)))
    else:
        print(len(merge_idx))
        split = np.cumsum(merge_idx)
        tmp = zip(chain([0], split), chain(split, [None])) 
        res = list(rowidx[i : j] for i, j in tmp) 

        merge = {}
        merged_idx = 0
        for i,c in enumerate(rowidx):
            if i >= split[merged_idx]: merged_idx+=1
            merge.update({c: merged_idx+1})
        print(merge)
        print('merging {} -> {} groups'.format(len(merge), len(set(merge.values()))))
        merge_list = [merge[e] for e in unique_elements]
        merged_labels = [merge_list[x-1] for x in output_labels]

        merged_labels=pd.Series(np.asarray(merged_labels), dtype="category")
        merged_labels.index = data_all.index
        return merged_labels
    
# Automated Hierarchical Merging
def f_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def hierach_merge(data_all, g, label_array, axis='rows', method='distance', c=0.1):
	if axis=='rows':
		Z = g.dendrogram_row.linkage
	elif axis=='cols':
		Z = g.dendrogram_col.linkage
	else: raise NameError('axis can only be "rows" or "cols"')
		
	if method == "distance":
		plt.figure(figsize=(25, 10))
		f_dendrogram(
			Z,
			truncate_mode='lastp',
			p=200,
			leaf_label_func=lambda id:id+1,
			leaf_rotation=90.,
			leaf_font_size=12.,
			show_contracted=True,
			annotate_above=10,
			max_d=c
		)
		plt.show()
		clusters = fcluster(Z, c, criterion='distance')
	elif method == "max":
		plt.figure(figsize=(25, 10))
		f_dendrogram(
			Z,
			truncate_mode='lastp',
			p=c,
			leaf_label_func=lambda id:id+1,
			leaf_rotation=90.,
			leaf_font_size=12.,
			show_contracted=True,
			annotate_above=10,
		)
		plt.show()
		clusters = fcluster(Z, c, criterion='maxclust')
	else: raise NameError('method can only be "distance" or "max"')
	merged_labels = [clusters[i-1] for i in label_array]
	merged_labels=pd.Series(np.asarray(merged_labels), dtype="category")
	merged_labels.index = data_all.index
	return merged_labels

def setParams(**kwargs):
	return kwargs

def propagateLabels(df, markers, in_labels, **kwargs):
	kernel = kwargs.get('kernel', 'knn')
	if "kernel" in kwargs:
		del kwargs["kernel"]
	n_neighbors = kwargs.get('n_neighbors', 5)
	if "n_neighbors" in kwargs:
		del kwargs["n_neighbors"]
	alpha = kwargs.get('alpha', 0.8)
	if "alpha" in kwargs:
		del kwargs["alpha"]
	n_jobs = kwargs.get('n_jobs', -1)
	if "n_jobs" in kwargs:
		del kwargs["n_jobs"]
	max_iter = kwargs.get('max_iter', 150)
	if "max_iter" in kwargs:
		del kwargs["max_iter"]
	
	markers_all = df.loc[:,markers]

	labels = []
	for i in in_labels:
		i = int(i)
		if i == 0: 
			i = -1
		labels.append(i)

	label_spread = label_propagation.LabelSpreading(kernel=kernel, n_neighbors=n_neighbors, alpha=alpha, n_jobs=n_jobs, max_iter=max_iter, **kwargs)
	label_spread.fit(markers_all, labels)

	output_labels = label_spread.transduction_
	output_label_array = pd.Series(np.asarray(output_labels), dtype="category")
	output_label_array.index = df.index
	return output_label_array

def loadCelltypeNames(df, outdir, filename, colname):
	clusterID = sorted(np.unique(df[colname].values))

	with open(outdir+filename) as csvdict:
		clusterdict = {int(row[0]): row[1] for row in csv.reader(csvdict, dialect='excel')}

	celltypeName = [clusterdict[i] for i in df[colname].values.astype(np.int16)]
	celltypename2id = {name:i for i,name in enumerate(sorted(np.unique(list(clusterdict.values()))))}

	(pd.DataFrame.from_dict(data=celltypename2id, orient='index').to_csv(outdir+'celltypename2id.csv', header=False))

	celltypeID = [celltypename2id[name] for name in celltypeName]
	celltypeID = pd.Series(np.asarray(celltypeID), dtype="category")
	celltypeID.index = df.index

	df['CelltypeName'] = celltypeName
	df['CelltypeID'] = celltypeID
	return df

def ClusterMeansHeatmap(df, colname, markers, save = False, outdir = None, estimate_dist = 0, metric='correlation', method='single', cmap='viridis', standard_scale= 1, **kwargs):
	cluster_expression = df.groupby(colname).mean()
	cluster_expression = cluster_expression.dropna(axis=0, how='all')
	cluster_expression = cluster_expression.loc[:,markers]
	g = sns.clustermap(cluster_expression, metric=metric, method=method, cmap=cmap, standard_scale=standard_scale, **kwargs)
	plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
	if save:
		if outdir==None: 
			print('Provide output directory as keyword argument "outdir"')
		else:
			newDir(outdir)
			plt.savefig(outdir+"ClusterMeansHeatmap_"+str(colname)+".svg", format="svg")
	
	if estimate_dist:
		dendro = g.dendrogram_row.linkage.copy()
		last = dendro[:, 2]
		last = last[:np.ceil(-len(last)/2).astype(int)]
		last_rev = last[::-1]
		idxs = np.arange(1, len(last) + 1)

		kn = KneeLocator(idxs, last_rev, curve='convex', direction='decreasing', S=1, interp_method='polynomial')
		merge_cutoff = last_rev[kn.knee]

		#plt.plot(idxs, last_rev)
		#kn.plot_knee_normalized()
		print('Estimated merge distance is '+str(merge_cutoff))
		return g, merge_cutoff
	else: return g

def CellExpressionHeatmap(df, colname, markers, save = False, outdir = None, palette=None, metric="euclidean", method="complete", cmap='viridis', vmax = 0.3, **kwargs):
	sort_data = df.sort_values(by=[colname])
	mergeIDs = np.unique(df[colname].astype(np.int).values)
	if palette==None: 
		palette = sns.color_palette("cubehelix", len(mergeIDs))
	lut = dict(zip(mergeIDs, palette))
	row_colors = sort_data[colname].astype(int).map(lut)
	p = sns.clustermap(sort_data.loc[:,markers], metric=metric, method=method, cmap=cmap, row_cluster=False, row_colors=row_colors, standard_scale= 1, vmax = vmax, yticklabels=False, **kwargs)
	for label in mergeIDs:
		p.ax_col_dendrogram.bar(0, 0, color=lut[label],
								label=label, linewidth=0)
	p.ax_col_dendrogram.legend(loc="center", ncol=11)
	if save:
		if outdir==None: 
			print('Provide output directory as keyword argument "outdir"')
		else:
			newDir(outdir)
			plt.savefig(outdir+"CellExpressionHeatmap_"+str(colname)+".png", format="png")

def CelltypeCompositionHeatmap(df, runs, colname, drop = [], save = False, outdir = None, proportions =True, metric="correlation", method="average", cmap='viridis', standard_scale= 0, figsize=(10, 30), **kwargs):
	tissue_ids = ['{}_{}'.format(run,reg) for run, reg in zip(df.index.get_level_values('RunID'),df.index.get_level_values('RegionID'))]
	celltype_counts = pd.get_dummies(df[colname]).groupby(tissue_ids).sum()
	celltype_counts = celltype_counts.fillna(0)
	celltype_counts = celltype_counts.astype('int64')
	celltype_counts = celltype_counts.drop(drop, axis=1) #drop unassigned cluster
	celltype_counts = celltype_counts.transpose()
	if proportions:
		celltype_counts = celltype_counts.apply(lambda x: x/x.sum(), axis=0)

	g = sns.clustermap(celltype_counts, metric=metric, method=method, cmap=cmap, standard_scale= standard_scale, figsize=figsize, **kwargs)
	plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
	if save:
		if outdir==None: 
			print('Provide output directory as keyword argument "outdir"')
		else:
			newDir(outdir)
			plt.savefig(outdir+"tissuecorrelation_"+str(colname)+".svg", format="svg")
	
	return celltype_counts

def loadData(runs, path, subdir, filename_pattern, Z, RemoveHighBlankCells, DNAfilter, DNAChannel, ManualDNAthreshold, DRAQ5Channel, Zremoval, Zfilter):
	data=[]
	colnames=[]
	for i in range(len(runs)):
		tmp, colnames, filtercount = preprocess(i, path, runs, subdir, filename_pattern, Z, RemoveHighBlankCells, DNAfilter, DNAChannel, ManualDNAthreshold, DRAQ5Channel, Zremoval, Zfilter, colnames) 
		data.append(tmp)
		print(filtercount,' events were removed in ',runs[i],' ',len(data[i]),' events remain')  
	data_all = pd.concat(data, keys=runs, names=['RunID'])
	data_all = data_all.reset_index()
	data_all['CellID']=range(len(data_all))
	data_all=data_all.set_index(['CellID', 'RunID', 'RegionID', 'RegCellID'])
	print(str(len(data_all))+" events for clustering")
	return data, data_all

def getMarkersforClustering(df, list):
	markers=[]
	for i in df.columns:
		if any(d.lower() == i.split(':')[-1].lower() for d in list): 
			markers.append(i)
	print(str(len(markers))+' markers used for clustering')
	return markers

def newDir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

def loadState(outdir):
	if os.path.isfile(outdir+'notebook_env.cache'):
		dill.load_session(outdir+'notebook_env.cache')
		
def saveState(outdir):
	newDir(outdir)
	dill.dump_session(outdir+'notebook_env.cache')

def save_as_CSV(data_all, runs, outdir):
	# Export all csv and region split csvs
	newDir(outdir)
	filename = outdir+'MultiRun-CombinedData.csv'
	export_csv = data_all.to_csv (filename, index = True, header=True) #Don't forget to add '.csv' at the end of the path

	regs = np.unique(data_all.index.get_level_values('RegionID').values)
	for h, i in product(runs, regs):
		try:
			df = pd.DataFrame(data_all.loc[pd.IndexSlice[:,h,i],:])
			if len(df) == 0:
				continue
			else:
				outdir_run = outdir+h+'/'
				if not os.path.exists(outdir_run):
					os.mkdir(outdir_run)
				filename = outdir_run+h+'_'+str(i)+'_compensated.csv'
				export_csv = df.to_csv (filename, index=True, header=True)
				print("Wrote data to '{:}'".format(os.path.basename(filename)))
		except: continue

def save_as_FCS(data_all, runs, outdir):
	newDir(outdir)
	fcs_data = data_all.copy()
	try:
		fcs_data = fcs_data.drop('CelltypeName', axis=1)
		print('CelltypeName column will not be exported in the FCS file! Use CelltypeID and celltypename2id.csv in the output directory to match names to IDs.')
	except:	print('No CelltypeName column, continuing...')

	header = fcs_data.columns#[2:124]
	channel_cycle = [h.split(':')[ 0] for h in header]
	channel_stain = [h.split(':')[-1] for h in header]

	regs = np.unique(data_all.index.get_level_values('RegionID').values)
	for h, i in product(runs, regs):
		try:
			df = fcs_data.loc[pd.IndexSlice[:,h,i],:].values.astype(np.float32)
			if len(df) == 0:
				continue
			else:
				nanrows = np.isnan(df).any(axis=1)
				outdir_run = outdir+h+'/'
				if not os.path.exists(outdir_run):
					os.mkdir(outdir_run)
				filename = outdir_run+'reg{:03d}_compensated.fcs'.format(i)
				write_fcs(filename, df, channel_stain, channel_cycle)
				print("Wrote data to '{:}'".format(os.path.basename(filename)))
		except: continue

def generateTissueArray(df, X, Y, shift = 10000):
	run2idx = {r:i for i,r in enumerate(sorted(np.unique(df.index.get_level_values('RunID').values)))}
	xshift = np.swapaxes(np.asarray([[(n)*shift for n in df.index.get_level_values('RegionID').values]]), 0,1)
	plot_data = df.copy()
	plot_data[[X]] = plot_data[[X]].values + xshift
	yshift = np.swapaxes(np.asarray([[run2idx[n]*shift for n in df.index.get_level_values('RunID').values]]), 0,1)
	plot_data[[Y]] = plot_data[[Y]].values + yshift
	return plot_data

def plotlyTissueArray(plot_data, X, Y, colname, palette, template='plotly_dark', width=900, height=600, **kwargs):
	p=plot_data
	idx=plot_data[colname].unique()
	fig = px.scatter(p, x=X, y=Y, color=colname,
					 color_discrete_sequence=palette,
					 width=width, height=height, template=template, render_mode='webgl', **kwargs)
	fig.update_yaxes(autorange="reversed")
	fig.show()

