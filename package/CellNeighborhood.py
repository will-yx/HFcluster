import numpy as np
import pandas as pd

import time
import sys
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay


def find(data_all, outdir, X='x:x', Y='y:y', cluster_col = 'CelltypeName', k = 10, n_neighborhoods = 15, drop=False, plot=True):
	
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
		to_fit = tissue.loc[indices][[X,Y]].values

	#    fit = NearestNeighbors(n_neighbors=n_neighbors+1).fit(tissue[[X,Y]].values)
		fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[X,Y]].values)
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
	
	reg = 'Sample'
	keep_cols = [X,Y,reg,cluster_col]

	#in the line below set the K for the KNN based neighbohood definition (window size)

	#the line below sets K for the K-means
	neighborhood_name = "neighborhood"+str(k)

	neighbor_data = data_all.loc[:,[X, Y, cluster_col]]
	if drop:
		for i in drop:
			neighbor_data = neighbor_data.loc[neighbor_data[cluster_col] != i]
	celltypes = neighbor_data[cluster_col].unique().tolist()
	neighbor_data[reg] = neighbor_data.index.get_level_values('RunID') + "_" + neighbor_data.index.get_level_values('RegionID').map(str)
	files = neighbor_data.Sample.unique().tolist()
	neighbor_data = neighbor_data.reset_index(drop = True)

	cells = pd.concat([neighbor_data,pd.get_dummies(neighbor_data[cluster_col])],1)
	sum_cols = cells[cluster_col].unique()
	values = cells[sum_cols].values

	#find windows for each cell in each tissue region

	tissue_group = cells[[X,Y,reg]].groupby(reg)
	exps = list(cells[reg].unique())
	tissue_chunks = [(time.time(),files.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)]
	tissues = np.array([get_windows(job,k) for job in tissue_chunks])

	#for each cell and its nearest neighbors, reshape and count the number of each cell type in those neighbors.
	out_dict = {}
	for neighbors,job in zip(tissues,tissue_chunks):
		chunk = np.arange(len(neighbors))#indices
		tissue_name = job[2]
		indices = job[3]
		window = values[neighbors[:,:k].flatten()].reshape(len(neighbors),k,len(sum_cols)).sum(axis = 1)
		out_dict[(tissue_name,k)] = (window.astype(np.float16),indices)

	#concatenate the summed windows and combine into one dataframe for each window size tested.
	window = pd.concat([pd.DataFrame(out_dict[(exp,k)][0],index = out_dict[(exp,k)][1].astype(int),columns = sum_cols) for exp in exps],0)
	window = window.loc[cells.index.values]
	window = pd.concat([cells[keep_cols],window],1)

	km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)
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
		s=sns.clustermap(fc.loc[:,celltypes], vmin =-2,vmax = 2,cmap = divergmap, metric="euclidean", method="ward", row_cluster = True, figsize = (30,30))
		ax = s.ax_heatmap
		ax.set_xlabel("Celltypes")
		ax.set_ylabel("kmeans_neighborhood IDs")
		plt.savefig(outdir+"compositions_"+neighborhood_name+".svg", format="svg")

	neighborID = cells[neighborhood_name].astype('category')
	if drop:
		for i in range(len(drop)):
			if i ==0:
				assigned = data_all.loc[data_all[cluster_col] != drop[i]]
			else:
				assigned = assigned.loc[assigned[cluster_col] != drop[i]]
		neighborID.index = assigned.index
	else:
		neighborID.index = data_all.index
	data_all['kmeans_neighborhood'] = neighborID
	data_all['kmeans_neighborhood'] = data_all['kmeans_neighborhood'].cat.add_categories([-1]).fillna(-1)
	return data_all, window

def delaunay_neighbors(codex_df, cluster_col='CelltypeName', drop=[], X='x:x', Y='y:y', Z='z:z'):
	codex_df = codex_df[[type not in drop for type in codex_df[cluster_col]]]
	celltypes = np.unique(codex_df[cluster_col])
	runs = sorted(codex_df.index.get_level_values('RunID').unique())

	delaunay_ids = pd.DataFrame()
	delaunay_annos = pd.DataFrame()
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
					dummie[cell,neighbor-1] = dummie[cell,neighbor-1]+1
			tissue_niche_types = pd.DataFrame(dummie, index=tissue.index, columns=celltypes)
			#append
			delaunay_ids = delaunay_ids.append(tissue_niche_neighbors)
			delaunay_annos = delaunay_annos.append(tissue_niche_types)
	return delaunay_ids, delaunay_annos	