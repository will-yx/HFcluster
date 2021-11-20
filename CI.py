import numpy as np, pandas as pd, seaborn as sns
import networkx as nx
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad

def neighborinteractions(df, col_name, mode='interactions', drop=False, save=False, **kwargs):
	tissue_mean = np.array(df.iloc[:,4:-2].astype(np.int64).sum()/len(df))
	celltypes = df.iloc[:,4:-2].columns
	centroids = df.iloc[:,4:-2].groupby(df[col_name]).mean().values
	clusters = sorted(df[col_name].unique())
	
	if mode=='logodd':
		fc = np.log2(((centroids+tissue_mean)/(centroids+tissue_mean).sum(axis = 1, keepdims = True))/tissue_mean)
	elif mode=='interactions':
		fc = centroids
	elif mode=='norm_interactions':
		fc = centroids/tissue_mean
	elif mode=='zscore':
		tissue_std = np.array(df.iloc[:,4:-2].astype(np.int64).std())
		fc = (centroids-tissue_mean)/tissue_std
	elif mode=='pct':
		fc = df.iloc[:,4:-2].rank(axis=0, pct = True, numeric_only=True).groupby(df[col_name]).mean()
	else: raise NameError('mode not found. Choose between "interactions", "norm_interactions", "logodd", "pct".')
	fc = pd.DataFrame(fc,columns = celltypes, index=clusters)
	if mode=='logodd':
		fc = fc.fillna(fc.min(numeric_only=True).min())
	else:
		fc = fc.fillna(0)
	if drop:
		fc = fc.drop(drop, axis=1)
	divergmap = sns.diverging_palette(250, 1, s=90, l=50, sep=10, center='light', as_cmap=True)
	s=sns.clustermap(fc, **kwargs)
	ax = s.ax_heatmap
	ax.set_xlabel('Celltypes')
	ax.set_ylabel(col_name)
	if save:
		plt.savefig(save[0]+"neighborinteractions"+col_name+'_'+save[1]+".svg", format="svg")
	return fc
		
import networkx as nx
import math
from matplotlib.patches import Arc, RegularPolygon
from numpy import radians as rad

def drawCirc(ax,radius,centX,centY,angle_,theta2_,weight,color_='black'):
    #========Line
    arc = Arc([centX,centY],radius,radius,angle=angle_, 
              theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=weight,color=color_)
    ax.add_patch(arc)


    #========Create the arrow head
    endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            weight*3,                # radius
            rad(theta2_+angle_),     # orientation
            color=color_
        )
    )
    # Make sure you keep the axes scaled or else arrow will distort

def networkmap(df, edgemax=10, layout='circle', weight=pd.DataFrame(), sort=False, filter=None, r = 500, arc=0.02, self_edge=True, figsize=(10,10), save=False):
	plt.figure(figsize=figsize)
	G = nx.MultiDiGraph()
	#Cell types as nodes
	G.add_nodes_from(df.index)
	#Add interactions as edges
	if weight.empty:
		weight = df.stack().reset_index().rename(columns={'level_0':'source','level_1':'target', 0:'weight'})
		weight['edge'] = list(zip(weight.source, weight.target))
		weight = weight.drop(['source', 'target'], axis=1)
		for i in weight.edge:
			G.add_edge(*i)
	# or reset edges and add from weight DataFrame
	else:
		G.remove_edges_from(list(G.edges()))
		for i in weight.edge:
			G.add_edge(*i)
		#G.remove_nodes_from(list(nx.isolates(G)))
    
	weights = weight['weight'].values
	weight['edge'] = [(*i, 0) for i in weight['edge']]
	weight['weight']=[{'weight': i} for i in weights]
	weight_attr = weight.set_index('edge').to_dict()

	nx.set_edge_attributes(G, weight_attr['weight'])

	#filter edges
	if filter:
		remove = []
		for s,e,k, in G.edges:
			if abs(G[s][e][k]['weight']) < filter:
				remove.append((s,e))
		for s,e in remove:
			G.remove_edge(s,e)
	
	min_weight = np.amin(list(nx.get_edge_attributes(G, 'weight').values()))
	max_weight = np.amax(list(nx.get_edge_attributes(G, 'weight').values()))
	#print(min_weight,max_weight)

	#sort nodes by rank or edges
	if not sort:
		sorted_nodes=G.nodes
	elif sort == 'edges':
		sorted_nodes = sorted([(len(list(nx.all_neighbors(G, node))),node) for node in G.nodes], reverse=1)
		sorted_nodes = [node[1] for node in sorted_nodes]
	elif sorted(sort) == sorted(list(G.nodes)):
		sorted_nodes = sort
	else:
		raise('sort list is incompatible, must contain all nodes')

	#set initial position of cell types equally distributed on a circle
	numPoints = len(G.nodes)
	points = []
	for index in range(numPoints):
		points.append([r*math.cos((index*2*math.pi)/numPoints),r*math.sin((index*2*math.pi)/numPoints)])
	initial_pos = dict(zip(sorted_nodes,points))
	 
	if layout =='circle':
		pos=initial_pos
	elif layout == 'spring':
		pos=nx.spring_layout(G, scale=100, iterations=500, weight = 'weight', pos=initial_pos)
	elif layout == 'kamada_kawai':
		distances=list(nx.get_edge_attributes(G, 'weight').values())
		pos=nx.kamada_kawai_layout(G,dist=distances,pos=initial_pos,weight='weight',scale=50,center=None,dim=2)
	elif len(layout) > 1:
		pos=layout
	else:
		pos=initial_pos

	for s,e,k, in G.edges:
		#print(weight['weight'].values[i]['weight'])
		if G[s][e][k]['weight'] > 0:
			G[s][e][k]['color']='red'
		if G[s][e][k]['weight'] <= 0:
			G[s][e][k]['color']='blue'
				
	node_size=[len(G[node])*50 for node in G.nodes]
	node_weights = [np.sum([np.abs(G[node][celltype][0]['weight']) for celltype in list(G[node])]) for node in G.nodes]
	node_color= [((node_weight/max(node_weights)),0,1-(node_weight/max(node_weights)),1,) for node_weight in node_weights]
	#node_color=[(1-(len(G[node])/(max(node_size)/50)),1,(len(G[node])/(max(node_size)/50))) for node in G.nodes]

	ax = plt.gca()
	for s,e,k in G.edges:
		rad = arc*numPoints*(k+1)
		w = (abs(G[s][e][k]['weight'])-abs(min_weight))/max_weight*edgemax
		if s==e:
			if self_edge:
				drawCirc(ax,50,pos[s][0]+20,pos[s][1]+20,-80,240,w,color_=G[s][e][k]['color'])
			else: continue
		else:
			ax.annotate("",xy=pos[s],xytext=pos[e],arrowprops=dict(
										width=w,headwidth=w*3,headlength=w*3,
										#arrowstyle="->", 
										color=G[s][e][k]['color'],
										connectionstyle=f"arc3,rad={rad}",
										alpha=0.4))#,linewidth=w))
		

	nx.draw_networkx_nodes(G, pos,
						   node_color=node_color,#'0.85',
						   node_size=node_size,
						   alpha=1)
	nx.draw_networkx_labels(G, {p:[pos[p][0],pos[p][1]-20] for p in pos}, font_size=10)

	plt.axis('off')
	ax.set_xlim(min(list(zip(*pos.values()))[0])-50,max(list(zip(*pos.values()))[0])+50)
	ax.set_ylim(min(list(zip(*pos.values()))[1])-50,max(list(zip(*pos.values()))[1])+50)
	if save:
		plt.savefig(save[0]+"interaction_network_"+save[1]+".svg", format="svg")
	plt.show()
	plt.draw()
	return G, pos
	
divergmap = sns.diverging_palette(1, 250, s=90, l=50, sep=10, center='light', as_cmap=True)
divergmap_r = sns.diverging_palette(250, 1, s=90, l=50, sep=10, center='light', as_cmap=True)
directmap = sns.light_palette((1, 90, 50), input="husl", as_cmap =True)
directmap_r = sns.light_palette((1, 90, 50), input="husl", as_cmap =True, reverse=True)

from scipy import stats
def bayes_test(data, alpha, threshold):
    mean,var,std  = stats.bayes_mvs(data, alpha=alpha)
    FC = mean[0]
    CI = mean[1]
    sig = any([all([CI[0]>threshold[1], CI[1]>threshold[1]]), all([CI[0]<threshold[0], CI[1]<threshold[0]])])
    return FC, CI, sig