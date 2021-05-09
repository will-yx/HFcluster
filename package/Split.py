import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

# Plotly Biaxial
def plotBiaxial(input, Ch_x, Ch_y, Ch_c = False, log = False):
	data = input.reset_index()
	
	for i in input.columns:
		if Ch_x.lower() == i.split(':')[-1].lower(): 
			Ch_x=i
		if Ch_y.lower() == i.split(':')[-1].lower(): 
			Ch_y=i
		if Ch_c:
			if Ch_c.lower() == i.split(':')[-1].lower(): 
				Ch_c=i

	if Ch_c:
		cell_fig = px.scatter(data, x=Ch_x, y=Ch_y, color=Ch_c,
						 color_continuous_scale='viridis',
						 log_x=log, log_y=log,
						 width=800, height=800, template='plotly_dark', render_mode='webgl')
	else:
		cell_fig = px.scatter(data, x=Ch_x, y=Ch_y,
						 log_x=log, log_y=log,
						 width=800, height=800, template='plotly_dark', render_mode='webgl')

	fig  = go.FigureWidget(cell_fig)

	sel_idx = []

	def get_points_wrapper(out):
		def retrieve_select(trace, points, state):
			out[:] = points.point_inds
		return retrieve_select    
	for f in fig.data:
		f.on_selection(get_points_wrapper(sel_idx))
	display(fig)
	return (data, sel_idx)

def getSelectCells(input, sel_idx):
	select_data = input.loc[sel_idx] 
	#print(select_data)
	idx=select_data['CellID'].values
	return idx
	
def newCluster(input, colname, idx):
	new = max(input[colname].unique())+1
	temp=input[colname].cat.remove_unused_categories()
	temp=temp.cat.add_categories([new])
	temp.loc[pd.IndexSlice[idx,:,:,:]]=new
	split=input.copy()
	split[colname]=temp
	split[colname]=split[colname].cat.remove_unused_categories()
	print('Selected cells are now labeled as {} in {}.'.format(new,colname))
	return split