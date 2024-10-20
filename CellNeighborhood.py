import numpy as np
import pandas as pd

import time
import sys
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay 

def find(adata, cluster_col = 'CelltypeName', sample_col='sample', n_neighbors = 10, k_neighborhoods = 15, drop=False, plot=False):
	
	def get_windows(job,n_neighbors):
		'''
		For each region and each individual cell in dataset, return the indices of the nearest neighbors.
		
		'job:  meta data containing the start time,index of region, region name, indices of region in original dataframe
		n_neighbors:  the number of neighbors to find for each cell
		'''
		start_time,idx,tissue_name,indices = job
		job_start = time.time()
		
		print ("Processing tissues:", str(idx+1)+'/'+str(len(exps)),': ' + exps[idx])

		tissue = tissue_group.get_group(tissue_name)
		to_fit = tissue.loc[indices][['x','y']].values

	#    fit = NearestNeighbors(n_neighbors=n_neighbors+1).fit(tissue['x','y']].values)
		fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[['x','y']].values)
		m = fit.kneighbors(to_fit)
	#    m = m[0][:,1:], m[1][:,1:]
		m = m[0], m[1]
		
		#sort_neighbors
		args = m[0].argsort(axis = 1)
		add = np.arange(m[1].shape[0])*m[1].shape[1]
		sorted_indices = m[1].flatten()[args+add[:,None]]

		neighbors = tissue.index.values[sorted_indices]
		end_time = time.time()

		#print ("Finishing:", str(idx+1)+"/"+str(len(exps)),": "+ exps[idx],end_time-job_start,end_time-start_time)
		return neighbors.astype(np.int32)
	
	keep_cols = ['x','y',sample_col,cluster_col]

	#in the line below set the K for the KNN based neighbohood definition (window size)

	#the line below sets K for the K-means
	neighborhood_name = "neighborhood"+str(n_neighbors)

	neighbor_data = pd.DataFrame([adata.obsm['spatial'][:,0],adata.obsm['spatial'][:,1], adata.obs[sample_col],adata.obs[cluster_col].values], columns=adata.obs.index, index=keep_cols).T

	celltypes = neighbor_data[cluster_col].unique().tolist()
	files = neighbor_data[sample_col].unique().tolist()

	cells = pd.concat([neighbor_data.reset_index(drop = True),pd.get_dummies(neighbor_data.reset_index(drop = True)[cluster_col])],axis=1)
	sum_cols = cells[cluster_col].unique()
	values = cells[sum_cols].values

	#find windows for each cell in each tissue region

	tissue_group = cells[['x','y',sample_col]].groupby(sample_col)
	exps = list(cells[sample_col].unique())
	tissue_chunks = [(time.time(),files.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)]
	tissues = np.array([get_windows(job,n_neighbors) for job in tissue_chunks])

	#for each cell and its nearest neighbors, reshape and count the number of each cell type in those neighbors.
	out_dict = {}
	for neighbors,job in zip(tissues,tissue_chunks):
		chunk = np.arange(len(neighbors))#indices
		tissue_name = job[2]
		indices = job[3]
		window = values[neighbors[:,:n_neighbors].flatten()].reshape(len(neighbors),n_neighbors,len(sum_cols)).sum(axis = 1)
		out_dict[(tissue_name,n_neighbors)] = (window.astype(np.float16),indices)

	#concatenate the summed windows and combine into one dataframe for each window size tested.
	window = pd.concat([pd.DataFrame(out_dict[(exp,n_neighbors)][0],index = out_dict[(exp,n_neighbors)][1].astype(int),columns = sum_cols) for exp in exps],axis=0)
	window = window.loc[cells.index.values]
	window = pd.concat([cells[keep_cols],window],axis=1)
	window.index = neighbor_data.index

	km = MiniBatchKMeans(n_clusters = k_neighborhoods,random_state=0)
	labelskm = km.fit_predict(window[sum_cols].values)
	k_centroids = km.cluster_centers_
	cells[neighborhood_name] = labelskm
	cells[neighborhood_name] = cells[neighborhood_name].astype('category')

	# this plot shows the types of cells (ClusterIDs) in the different niches
	#k_to_plot = k
	if plot:
		niche_clusters = (k_centroids)
		tissue_avgs = values.mean(axis = 0)
		fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
		fc = pd.DataFrame(fc,columns = sum_cols)
		divergmap = sns.diverging_palette(250, 1, s=90, l=50, sep=10, center='light', as_cmap=True)
		s=sns.clustermap(fc.loc[:,celltypes], vmin =-2,vmax = 2,cmap = divergmap, metric="euclidean", method="ward", row_cluster = True, figsize = (10,10))
		ax = s.ax_heatmap
		ax.set_xlabel("Celltypes")
		ax.set_ylabel("kmeans_neighborhood IDs")

	neighborIDs = cells[neighborhood_name].astype('category')
	adata.obs['kmeans_neighborhood']=neighborIDs.values
	adata.uns['spatial_neighborhood'] = {'n_neighbors':n_neighbors, 'k-means':k_neighborhoods}

	return adata, window

def delaunay_neighbors(codex_df, cluster_col='CelltypeName', drop=[], X='x:x', Y='y:y', Z='z:z'):
	codex_df = codex_df[[type not in drop for type in codex_df[cluster_col]]]
	celltypes = np.unique(codex_df[cluster_col])
	lookup = dict(zip(celltypes, range(len(celltypes))))
	runs = sorted(codex_df.index.get_level_values('RunID').unique())

	delaunay_ids = pd.DataFrame()
	delaunay_annos = pd.DataFrame()
	delaunay_annos[cluster_col] = codex_df[cluster_col]
	for r in runs:
		run = codex_df[codex_df.index.get_level_values('RunID')==r]
		regions = sorted(run.index.get_level_values('RegionID').unique())

		for region in regions:
			print('Generating Delaunay graph for {} region {}'.format(r, region))
			tissue = run[run.index.get_level_values('RegionID')==region]
			cell_coords = np.array([tissue[X].values, tissue[Y].values], np.int32).T
			cellids = tissue.index.get_level_values('CellID')

			#delaunay triangulation to find neighbors
			tri = Delaunay(cell_coords)
			indptr, neighbor_indices = tri.vertex_neighbor_vertices
			
			#get CellIDs of neighbors
			neighbor_ids = np.array([cellids[idx] for idx in neighbor_indices]).T
			tissue_niche_neighbors = pd.DataFrame(index=tissue.index)
			tissue_niche_neighbors['delaunay_neighbors'] = [neighbor_ids[indptr[cell]:indptr[cell+1]] for cell in range(len(tissue))]
			#get cell type annotation of neighbors
			neighbor_anno = np.array([tissue[cluster_col].iloc[idx] for idx in neighbor_indices]).T
			dummie = np.zeros((len(tissue), len(celltypes)), dtype=np.int)
			for cell in range(len(tissue)):
				neighbors = neighbor_anno[indptr[cell]:indptr[cell+1]]
				for neighbor in neighbors:
					celltype_idx = lookup[neighbor]
					dummie[cell,celltype_idx] = dummie[cell,celltype_idx]+1
			tissue_niche_types = pd.DataFrame(dummie, index=tissue.index, columns=celltypes)
			#append
			delaunay_ids = delaunay_ids.append(tissue_niche_neighbors)
			delaunay_annos = delaunay_annos.append(tissue_niche_types)
	return delaunay_ids, delaunay_annos	

		
def leiden_cluster(adata, niches, celltype_col='CelltypeName', resolution=0.5, cluster_size_filter=1000, plot_umap=True, **kwargs):
	import scanpy as sc
	celltypes = niches[celltype_col].unique()
	niches_adata=sc.AnnData(niches.loc[:,celltypes])
	sc.pp.neighbors(niches_adata, use_rep='X')
	sc.tl.umap(niches_adata, **kwargs)
	sc.tl.leiden(niches_adata, resolution=resolution)
	print('Leiden clustering identified {} clusters'.format(len(niches_adata.obs.leiden.unique())))

	leiden_niche = niches_adata.obs.leiden
	niche_counts = pd.DataFrame(np.unique(leiden_niche, return_counts=True)).T.sort_values(1, ascending=False)
	filtered_niche_idx = niche_counts[niche_counts[1]>cluster_size_filter][0].values
	print('{} clusters left of {} cells or greater'.format(len(filtered_niche_idx), cluster_size_filter))
	
	niches['leiden']=leiden_niche.values
	niches['knn_umap_X'] = niches_adata.obsm['X_umap'][:,0]
	niches['knn_umap_Y'] = niches_adata.obsm['X_umap'][:,1]
	if cluster_size_filter>1:
		niches['filtered_leiden']=[x if x in filtered_niche_idx else -1 for x in niches['leiden']]
	
	adata.obs['knn_niche'] = niches['filtered_leiden']
	adata.obs['knn_niche'] = adata.obs['knn_niche'].fillna(-1)
	adata.obs['knn_niche'] = adata.obs['knn_niche'].astype('category')
	adata.obsm['neighborhood_umap_X'] = niches['knn_umap_X'].values
	adata.obsm['neighborhood_umap_Y'] = niches['knn_umap_Y'].values
	
	if plot_umap:
		niches_adata.obs['filtered_leiden'] = niches['filtered_leiden'].astype('category')
		sc.pl.umap(niches_adata, color=['leiden'])
		plt.show()
		sc.pl.umap(niches_adata, color=['filtered_leiden'])
	
	return adata, niches