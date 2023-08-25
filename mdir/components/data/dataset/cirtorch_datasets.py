import torch
from daan.ml.tools import path_join
from daan.core.path_resolver import resolve_path
from cirtorch import datasets as cirdatasets


def cir_tuples_dataset(*args, **kwargs):
    return _cir_tuples_dataset(cirdatasets.traindataset.TuplesDataset, *args, **kwargs)

def cir_diverse_anchors_dataset(*args, **kwargs):
    return _cir_tuples_dataset(DiverseAnchorsDataset, *args, **kwargs)

def _cir_tuples_dataset(cls, data, transform, **params):
    assert not data
    dparams = {
        "name": params.pop("dataset"),
        "mode": params.pop("split"),
        "imsize": params.pop("image_size"),
        "nnum": params.pop("neg_num"),
        "transform": transform,
        "dataset_pkl": resolve_path(params.pop("dataset_pkl")),
        "ims_root": resolve_path(params.pop("image_dir")),
        "qsize": params.pop("query_size"),
        "poolsize": params.pop("pool_size"),
    }

    dataset = cls(**dparams, **params)
    setattr(dataset, "loader_params", {"drop_last": True, "collate_fn": cirdatasets.datahelpers.collate_tuples})
    setattr(dataset, "prepare_epoch", dataset.create_epoch_tuples)
    return dataset


def cir_image_list_dataset(data, transform, **params):
    if params.pop("image_labels", False):
        *data, params['image_labels'] = data
    images, bbxs = (data[0], None) if len(data) == 1 else data
    image_dir = resolve_path(params.pop("image_dir"))
    if not image_dir.endswith(".h5"):
        image_dir, images = "", [path_join(image_dir, x) for x in images]

    dataset = cirdatasets.genericdataset.ImagesFromList(
        root=image_dir,
        images=images,
        imsize=params.pop("image_size"),
        bbxs=bbxs,
        transform=transform,
        **params
    )
    setattr(dataset, "loader_params", {})
    return dataset


class DiverseAnchorsDataset(cirdatasets.traindataset.TuplesDataset):
    """A version of TuplesDataset with diverse anchor mining"""

    def __init__(self, *args, qpool_size, similar_exclude, similar_include, mark_easy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.qpool_size = min(qpool_size, self.db['qsize']) if qpool_size is not None else self.qsize
        self.similar_exclude = similar_exclude
        self.similar_include = similar_include
        self.mark_easy = mark_easy
        assert similar_exclude <= similar_include
        assert mark_easy is None or 0 <= mark_easy <= 1

    def _select_positive_pairs(self, net, device):
        return self._select_positive_pairs_db(net, device, self.db, self.qsize)

    def _select_positive_pairs_db(self, net, device, db, qsize, mine_label="pool"):
        assert qsize <= self.qpool_size

        # draw qsize random queries for tuples
        idxs2qpool = self._randperm(len(db['qidxs']), self.qpool_size)
        qidxs = [db['qidxs'][i] for i in idxs2qpool]
        pidxs = [db['pidxs'][i] for i in idxs2qpool]
        qvecs = self._extract_descriptors(qidxs, f"anc-{mine_label}", net, device)

        with torch.no_grad():
            print('>> Searching for diverse queries...')
            idx = 0
            idxs = [idx]
            dists = torch.empty(self.qpool_size, 0, device=qvecs.device)
            qscore_acc = []
            for _ in range(qsize-1):
                dist = torch.mm(qvecs.t(), qvecs[:,idx:idx+1])
                dists = torch.cat([dists, dist], dim=1)
                most_similar = dists.max(dim=1)[0]
                # Do not need to exclude already picked as their similarity is maximal
                valid_size = self.qpool_size - len(idxs)
                similar_split = max(int(valid_size * (1 - self.similar_exclude)), 1)
                dissimilar_split = min(int(valid_size * (1 - self.similar_include)), similar_split-1)
                dissimilar_part = most_similar.argsort()[dissimilar_split:similar_split]
                if self.shuffle:
                    choice = torch.randint(dissimilar_part.shape[0], (1,)).item()
                else:
                    choice = dissimilar_part.shape[0]-1
                idx = dissimilar_part[choice].item()
                qscore_acc.append(most_similar[idx].item())
                idxs.append(idx)

            print('>>>> Average new query max score: {:.2f}'.format(sum(qscore_acc)/len(qscore_acc)))

        qidxs = [qidxs[x] for x in idxs]
        pidxs = [pidxs[x] for x in idxs]
        difficulties = [""] * len(qidxs)

        if self.mark_easy is not None:
            qvecs = qvecs[:,idxs]
            pvecs = self._extract_descriptors(pidxs, f"pos-{mine_label}", net, device)
            sim_ord = (qvecs * pvecs).sum(0).argsort()
            easy_set = set(sim_ord[-int(self.mark_easy * qsize):].tolist())
            difficulties = ["-easy" if i in easy_set else "-hard" for i in range(len(qidxs))]

        tuple_labels = ["anc", "pos", self.first_neg] + ["neg"] * (self.nnum - 1)
        tuple_labels = [[x+y for y in difficulties] for x in tuple_labels]
        return qidxs, pidxs, tuple_labels, {"average_new_query_max_score": qscore_acc}
