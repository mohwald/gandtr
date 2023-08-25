"""Functions for layers"""

import torch


# Generic functions

def normalize_vec_l2(vec):
    """Perform l2 normalization on each vector in a given tensor (across last dim)"""
    return vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-6)

def idx2rank_dim1(idxs):
    """Convert indexes to their rank across dim 1, e.g. after argsort"""
    ranks = torch.zeros_like(idxs)
    size = idxs.shape[1]
    for i, idx in enumerate(idxs):
        ranks[i][idx] = torch.arange(size, device=idxs.device)
    return ranks

# Clustering assignment

def assign_weights_softmax(dists, base):
    """Assignment weights of softmax soft-assignment"""
    return torch.nn.functional.softmax(-base * dists, dim=1)

def assign_weights_cmeans(dists, fuzzifier, eps):
    """Assignment weights of c-means soft-assignment"""
    dists_eps = eps**((fuzzifier - 1) / 2)
    tiled_dists = dists.unsqueeze(2).repeat_interleave(dists.shape[1], dim=2) + dists_eps
    weights = 1 / tiled_dists.div(tiled_dists.transpose(1, 2)).pow(2 / (fuzzifier - 1)).sum(-1)
    return weights

# Clustering algorithms

def init_clusters_forgy(points, n_clusters):
    """Lloyd–Forgy algorithm for initializing clusters from points"""
    return points[torch.randperm(points.shape[0])[:n_clusters]]

def iterate_kmeans(points, clusters, iterations):
    """Iterate k-means clustering algorithm"""
    for i in range(iterations):
        _, assignment = torch.min(torch.cdist(points, clusters), dim=1)
        for ctr, cluster in enumerate(clusters):
            clusters[ctr] = torch.mean(points[assignment == ctr], dim=0)
    return clusters

def iterate_cmeans(points, clusters, iterations, fuzzifier, eps):
    """Iterate fuzzy c-means clustering algorithm"""
    for i in range(iterations):
        weights = torch.pow(assign_weights_cmeans(torch.cdist(points, clusters), fuzzifier, eps), fuzzifier).T
        clusters = weights.mm(points) / (weights.sum(-1, keepdim=True) + eps)
    return clusters

def iterate_softmax(points, clusters, iterations, base, eps):
    """Iterate soft k-means clustering algorithm bug with softmax weights"""
    for i in range(iterations):
        weights = torch.pow(assign_weights_softmax(torch.cdist(points, clusters), base), base).T
        clusters = weights.mm(points) / (weights.sum(-1, keepdim=True) + eps)
    return clusters

def cluster_faiss(points, n_clusters):
    """Cluster points using faiss"""
    from asmk import index
    idx = index.FaissIndex() if points.device.type == "cpu" else index.FaissGpuIndex(0)
    clusters = idx.cluster(points.cpu().numpy(), n_clusters, "l2")
    return torch.from_numpy(clusters).to(points.device)
