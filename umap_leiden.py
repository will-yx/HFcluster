import numpy as np
import umap
import igraph as ig
import leidenalg as la

def umap_embed(data, **kwargs):
    mapper = umap.umap_.UMAP(**kwargs)
    embedding=mapper.fit_transform(data)
    return mapper, embedding

def leiden(adj_matrix, method='rbc', resolution=1, n_iters=-1):
    print('Making graph...')
    sources, targets = adj_matrix.nonzero()
    edgelist = zip(sources.tolist(), targets.tolist())
    G = ig.Graph(edgelist)
    print('Partitioning...')
    if method == 'modular':
        partition = la.find_partition(G, la.ModularityVertexPartition)
    elif method == 'rbc':
        partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter = resolution)
    elif method == 'cpm':
        partition = la.find_partition(G, la.CPMVertexPartition, resolution_parameter = resolution)
    else:
        raise NameError('Unknown method. Choice of "modular", "rbc", "cpm"')
    print('Optimizing partitions...')
    optimiser = la.Optimiser()
    diff = 1
    while diff > 0:
        diff = optimiser.optimise_partition(partition, n_iterations=n_iters)
    return partition

def umap_leiden(mapper, method='rbc', resolution=1, n_iters=-1):
    g = mapper.graph_
    g = g.tocoo().astype(np.int8)    
    return leiden(g, method=method, resolution=resolution, n_iters=n_iters)

def partition_to_id(partition, min_members=1):
    partition = sorted(partition, key=len, reverse=True)
    n = max(max(partition, key=max))
    clusters = np.zeros(n+1)
    for i, members in enumerate(partition):
        if len(members) > min_members:
            clusters[members]=i
        else: clusters[members]=-1
    return clusters