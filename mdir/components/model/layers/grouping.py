"""
Layers grouping local features, returning indexes for each group at each position. The grouping
is performed for the whole tuple, not for a single image.
"""

import math
import pickle
import torch

from . import functional

#
# Generic grouping classes
#

SIZE_SHORTCUTS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768,
                  "64k": 65536, "128k": 131072, "256k": 262144, "512k": 524288}

class Grouping(torch.nn.Module):
    """General grouping base class"""

    eps = 1e-6

    # Input shapes (hard): x (features, dim), att (features, 1), c (features, dim)
    # Input shapes (soft): x (features, 1, dim), att (features, 1, 1), c (centroids, dim)
    # Return shape (features, dim) for hard and (features, centroids, dim) for soft
    feature_functions = {
        "iden": lambda x, att, c: x,
        "att": lambda x, att, c: att*x,
        "res": lambda x, att, c: x-c,
        "resatt": lambda x, att, c: att*(x-c),
        "normres": lambda x, att, c: functional.normalize_vec_l2(x-c),
        "normresatt": lambda x, att, c: att * functional.normalize_vec_l2(x-c),
        "normressoftmaxatt": lambda x, att, c: torch.nn.functional.softmax(att, dim=0) * att * functional.normalize_vec_l2(x-c),
        "normresatt2": lambda x, att, c: att**2 * functional.normalize_vec_l2(x-c),
    }

    nearest_params = {
        "all": lambda: None,
        "top": lambda ma=1: ma,
    }

    # Input and output shape: dst (features, centroids)
    assignment_functions = {
        "uniform": lambda: (lambda dst: torch.ones_like(dst)),
        "softmax": lambda base, *, detach=False: (lambda dst: functional.assign_weights_softmax(dst.detach() if detach else dst, base)),
        "softmax2": lambda base: (lambda dst: functional.assign_weights_softmax(dst**2, base)),
        "rankserie": lambda base: (lambda dst: base**(-functional.idx2rank_dim1(dst.argsort(dim=1)).type(torch.float32)-1) * (base-1)),
        "cmeans": lambda fuzzifier: (lambda dst: functional.assign_weights_cmeans(dst, fuzzifier, 1e-6))
    }

    # Input and output shapes: d (centroids, dim)
    descriptor_functions = {
        "l2norm": lambda: (lambda d: d / (torch.norm(d, dim=1, keepdim=True) + 1e-6)),
        "normsign": lambda: (lambda d: torch.sign(d) / d.shape[1]**0.5),
        "sigmoid": lambda base: (lambda d: 2*torch.sigmoid(base * d) - 1),
    }

    # Input shapes: d (centroids, dim), f (features, centroids, dim), att (features, 1),
    #               ass (features, centroids)
    # Return shape (centroids,)
    weight_functions = {
        "unif": lambda: (lambda d, f, att, ass: (ass != 0).any(dim=0).type(torch.float32)),
        "maxass": lambda: (lambda d, f, att, ass: ass.max(dim=0)[0]),
        "avgass": lambda: (lambda d, f, att, ass: ass.mean(dim=0)),
        "maxassatt": lambda *, detach=False: (lambda d, f, att, ass: ((ass*att).detach() if detach else ass*att).max(dim=0)[0]),
        "softmaxassatt": lambda: (lambda d, f, att, ass: (torch.nn.functional.softmax(ass*att, dim=0) * ass*att).sum(dim=0)),
        "avgassatt": lambda *, detach=False: (lambda d, f, att, ass: ((ass*att).detach() if detach else ass*att).mean(dim=0)),
        "avgassatt2": lambda: (lambda d, f, att, ass: (ass*att**2).mean(dim=0)),
        "descnorm3": lambda: (lambda d, f, att, ass: torch.norm(d, dim=-1)**3),
    }

    def __init__(self, centroids, features, nearest, assignment, descriptor, weights):
        super().__init__()
        centroids = self._parse_size(centroids)
        assert centroids > 0, centroids
        self.feature_function = self.feature_functions[features.lower()]
        self.nearest = self.str_func_call(nearest, self.nearest_params)
        self.assignment_function = self.str_func_call(assignment, self.assignment_functions)
        self.weight_function = self.str_func_call(weights, self.weight_functions)
        self.descriptor_function = self.str_func_call(descriptor, self.descriptor_functions)
        self.params = {"centroids": centroids, "features": features, "nearest": nearest,
                       "assignment": assignment, "descriptor": descriptor, "weights": weights}

    def forward(self, images):
        acc = []
        for feats, atts in images:
            # Convert the memory format of features from dim x size1 x size2 into size12 x dim
            acc.append(([x.reshape(x.shape[0], -1).T for x in feats], [x.reshape(-1, 1) for x in atts]))
        return self._forward(acc)

    def assign_images(self, images, centroids):
        """Assign each image (at each scale) independently"""
        grouped = torch.zeros(((len(images),) + centroids.shape), device=centroids.device)
        weights = torch.zeros((len(images), centroids.shape[0]), device=centroids.device)
        for i, (feats, atts) in enumerate(images):
            feat, att = torch.cat(feats, dim=0), torch.cat(atts, dim=0)
            if feat.shape[0]:
                desc, feat, ass = self.assign_features(feat, att, centroids)
                grouped[i] = self.descriptor_function(desc)
                weights[i] = self.weight_function(desc, feat, att, ass)
        return grouped, weights

    def assign_features(self, features, attentions, centroids):
        """Either hard-assign or soft-assign features from a single image to centroids forming
            a dense vector"""
        if self.nearest is None:
            # Soft-assign
            assignment = self.assignment_function(torch.cdist(features, centroids))
            features = self.feature_function(features.unsqueeze(1), attentions.unsqueeze(1), centroids)
            return (features * assignment.unsqueeze(2)).sum(0), features, assignment

        # Hard-assign
        dists, indexes = torch.topk(torch.cdist(features.detach(), centroids.detach()),
                                    self.nearest, dim=1, largest=False, sorted=False)
        assignment = self.assignment_function(dists)
        features = self.feature_function(features.unsqueeze(1), attentions.unsqueeze(1), centroids[indexes])
        descriptor = features * assignment.unsqueeze(2)
        # Dense vector building
        dense_descriptor = torch.zeros(centroids.shape + features.shape[:1], device=centroids.device)
        dense_descriptor[indexes,:,torch.arange(features.shape[0]).unsqueeze(1)] = descriptor
        dense_assignment = torch.zeros((features.shape[0], centroids.shape[0]), device=features.device)
        dense_assignment[torch.arange(features.shape[0]).unsqueeze(1),indexes] = assignment
        return dense_descriptor.sum(-1), features, dense_assignment

    @staticmethod
    def str_func_call(func, functions):
        """Convert string with the signature func-arg1-arg2-flag1-flag2 into a function call
        func(arg1, arg2, flag1=True, flag2=True) by looking up the function in the provided functions
        dictionary. Args are numeric (int/float) which distinguishes them from flags (string)."""
        name, *params = func.lower().split("-")
        args = []
        kwargs = {}
        for param in params:
            try:
                args.append(float(param) if "." in param else int(param))
            except ValueError:
                kwargs[param] = True
        return functions[name](*args, **kwargs)

    @staticmethod
    def _parse_size(size):
        """Convert string sizes like 64k to integer"""
        if isinstance(size, str):
            return SIZE_SHORTCUTS[size]
        return size

    @staticmethod
    def _detach_flatten(images):
        """Detaches all featues and flattens them across multiplie images"""
        return torch.cat([x.detach() for feats, atts in images for x in feats])

    @staticmethod
    def _filter_features(images, feature_mask):
        """Filter out features (and attentions) split into images given a boolean mask of flattened
        features"""
        pointer = 0
        result = []
        for i, (feats, atts) in enumerate(images):
            f_acc, a_acc = [], []
            for f, a in zip(feats, atts):
                mask = feature_mask[pointer:pointer+f.shape[0]]
                f_acc.append(f[mask])
                a_acc.append(a[mask])
                pointer += f.shape[0]
            result.append((f_acc, a_acc))
        assert pointer == feature_mask.shape[0]
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join('%s=%s' % (x, y) for x, y in self.params.items())})"


class BatchClustering(Grouping):
    """Clustering is performed for each batch"""

    clustering_functions = {
        "kmeans": lambda: (lambda f, c, i: functional.iterate_kmeans(f, c, i)),
        "cmeans": lambda fuzzifier: (lambda f, c, i: functional.iterate_cmeans(f, c, i, fuzzifier, 1e-6)),
        "softmax": lambda base: (lambda f, c, i: functional.iterate_softmax(f, c, i, base, 1e-6)),
    }

    def __init__(self, centroids, features, nearest, assignment, descriptor, weights, clustering,
                 iterations, *, outputdim):
        super().__init__(centroids, features, nearest, assignment, descriptor, weights)
        self.clustering = self.str_func_call(clustering, self.clustering_functions)
        self.params.update({"clustering": clustering, "iterations": iterations})

    def _forward(self, images):
        features = self._detach_flatten(images)
        clusters = functional.init_clusters_forgy(features, self.params['centroids'])
        clusters = self.clustering(features, clusters, self.params['iterations'])
        return self.assign_images(images, clusters)

#
# Codebooks
#

class Codebook(Grouping):
    """Base class for codebook-based groupings (with possibly large codebooks, e.g. 64k)"""

    def __init__(self, codebook, features, nearest, assignment, descriptor, weights, lr_multiplier,
                 top_centroids):
        super().__init__(len(codebook), features, nearest, assignment, descriptor, weights)
        self.codebook = torch.nn.Parameter(codebook)
        self.lr_multiplier = lr_multiplier
        self.top_centroids = self._parse_size(top_centroids)
        if self.top_centroids:
            assert [x for x in ["max", "sum", "avg", "unif"] \
                    if self.params['weights'].lower().startswith(x)], self.params['weights']

    def _forward(self, images):
        codebook = self.codebook
        if self.top_centroids:
            pospair = images[:2] # Compute weights only on query and positive
            atts = torch.cat([x.detach() for feats, atts in pospair for x in atts])
            if self.nearest is None:
                features = self._detach_flatten(pospair)
                weights = self._chunk_weights_all(features, atts, codebook.detach(), 1_000_000)
                codebook = codebook[weights.topk(self.top_centroids, largest=True, sorted=False)[1]]
            else:
                assert self.nearest == 1, "ma with chunking not implemented"
                # Compute assignment for all image features but weights using only query and positive
                features = self._detach_flatten(images)
                wgt, ass = self._chunk_weights_topk(features, atts, codebook.detach(), 1_000_000)
                codebook, feature_mask = self._reduce_codebook(wgt, ass, codebook, self.top_centroids)
                if feature_mask is not None:
                    images = self._filter_features(images, feature_mask)

        return self.assign_images(images, codebook)

    def _chunk_weights_topk(self, features, atts, codebook, block_size):
        """Compute centroid weights and assignment without allocating large distance matrices"""
        n_features = features.shape[0]
        step = max(block_size // codebook.shape[0], 1) if block_size is not None else n_features
        slices = list(range(0, n_features, step))
        slices = list(zip(slices, slices[1:] + [n_features]))
        assignment = torch.empty(n_features, device=codebook.device, dtype=torch.int64)
        weights = torch.zeros(len(slices), codebook.shape[0], device=codebook.device)
        for i, (start, end) in enumerate(slices):
            idx = torch.cdist(features[start:end], codebook).argmin(dim=1)
            assignment[start:end] = idx
            # Weights from attention which can be reduced
            if atts.shape[0] <= start:
                continue
            if atts.shape[0] < end:
                idx = idx[:atts.shape[0]-start]
            ass = torch.zeros(idx.shape[0], codebook.shape[0], device=codebook.device)
            ass[torch.arange(idx.shape[0]),idx] = 1
            weights[i] = self.weight_function(None, None, atts[start:end], ass)
        return self._reduce_slices(weights, slices, self.params['weights']), assignment

    def _chunk_weights_all(self, features, atts, codebook, block_size):
        """Compute centroid weights and assignment without allocating large distance matrices"""
        n_features = features.shape[0]
        step = max(block_size // codebook.shape[0], 1) if block_size is not None else n_features
        slices = list(range(0, n_features, step))
        slices = list(zip(slices, slices[1:] + [n_features]))
        weights = torch.zeros(len(slices), codebook.shape[0], device=codebook.device)
        for i, (start, end) in enumerate(slices):
            ass = self.assignment_function(torch.cdist(features[start:end], codebook))
            weights[i] = self.weight_function(None, None, atts[start:end], ass)
        return self._reduce_slices(weights, slices, self.params['weights'])

    @staticmethod
    def _reduce_slices(result, slices, reduction):
        """Apply reduction function (max, sum, avg, ..) on slices result"""
        reduction = reduction.lower()
        if reduction.startswith("max") or reduction.startswith("unif"):
            return result.max(dim=0)[0]
        elif reduction.startswith("sum"):
            return result.sum(dim=0)
        elif reduction.startswith("avg"):
            sizes = torch.tensor([y-x for x, y in slices], device=result.device)
            return (result * sizes.unsqueeze(1)).sum(dim=0) / sizes.sum()

    @staticmethod
    def _reduce_codebook(weights, assignment, codebook, top_centroids):
        """Reduce the number of centroids based on their weights. Filter out features assigned to
        them"""
        nonzero = weights > 0
        if nonzero.sum() < top_centroids:
            # Assigned to less centroids than reduced codebook size
            return codebook[nonzero], None

        # Filtering features assigned to filtered centroids
        idx = weights[nonzero].argsort(descending=True)
        idx = torch.arange(nonzero.shape[0])[nonzero][idx] # Convert to original indexes
        codebook = codebook[idx[:top_centroids]]
        exclude = idx[top_centroids:].to(assignment.device)
        feature_mask = (assignment.unsqueeze(1) != exclude).all(dim=1)
        return codebook, feature_mask


class ClusteringCodebook(Codebook):
    """Clustering is performed at the beginning of the first epoch"""

    recompute = False

    def __init__(self, centroids, features, nearest, assignment, descriptor, weights, lr_multiplier,
                 top_centroids, iterations, *, outputdim, **inference_params):
        super().__init__(torch.zeros((self._parse_size(centroids), outputdim)), features, nearest,
                         assignment, descriptor, weights, lr_multiplier, top_centroids)
        self.params['iterations'] = iterations
        self.inference_params = inference_params

    def compute_codebook(self, descriptors):
        centroids = functional.init_clusters_forgy(descriptors, self.params['centroids'])
        self.codebook.data = self.clustering(descriptors, centroids, self.params['iterations'])


class LoadedCodebook(Codebook):
    """Already trained codebook"""

    def __init__(self, centroids, features, nearest, assignment, descriptor, weights, lr_multiplier,
                 top_centroids, *, outputdim):
        super().__init__(self.load_codebook(centroids), features, nearest, assignment, descriptor,
                         weights, lr_multiplier, top_centroids)

    @staticmethod
    def load_codebook(path):
        if isinstance(path, torch.Tensor):
            return path
        with open(path, "rb") as handle:
            state = pickle.load(handle)
            return torch.from_numpy(state['state']['centroids'])


class FaissCodebook(Codebook):
    """Clustering is performed at the beginning of the first epoch using the faiss library."""

    recompute = False

    def __init__(self, centroids, features, nearest, assignment, descriptor, weights, lr_multiplier,
                 top_centroids, *, outputdim, **inference_params):
        super().__init__(torch.zeros((self._parse_size(centroids), outputdim)), features, nearest,
                         assignment, descriptor, weights, lr_multiplier, top_centroids)
        self.inference_params = inference_params

    def compute_codebook(self, descriptors):
        self.codebook.data = functional.cluster_faiss(descriptors, self.params['centroids'])


GROUPINGS = {
    "BatchClustering": BatchClustering,
    "ClusteringCodebook": ClusteringCodebook,
    "LoadedCodebook": LoadedCodebook,
    "FaissCodebook": FaissCodebook,
}