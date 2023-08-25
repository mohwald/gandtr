import os
import pickle
import pdb
import re
import io
import hashlib
import h5py
from urllib.request import urlopen

import torch
import torch.utils.data as data

from cirtorch.datasets.datahelpers import default_loader, imresize, cid2filename
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root

class TuplesDataset(data.Dataset):
    """Data loader that loads training and validation tuples of
        Radenovic etal ECCV16: CNN image retrieval learns from BoW

    Args:
        name (string): dataset name: 'retrieval-sfm-120k'
        mode (string): 'train' or 'val' for training and validation parts of dataset
        imsize (int, Default: None): Defines the maximum size of longer image side
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        qsize (int, Default:1000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:10000): Pool size for negative images re-mining

     Attributes:
        images (list): List of full filenames for each image
        clusters (list): List of clusterID per image
        qpool (list): List of all query image indexes
        ppool (list): List of positive image indexes, each corresponding to query at the same position in qpool

        qidxs (list): List of qsize query image indexes to be processed in an epoch
        pidxs (list): List of qsize positive image indexes, each corresponding to query at the same position in qidxs
        nidxs (list): List of qsize tuples of negative images
                        Each nidxs tuple contains nnum images corresponding to query image at the same position in qidxs

        Lists qidxs, pidxs, nidxs are refreshed by calling the ``create_epoch_tuples()`` method,
            ie new q-p pairs are picked and negative images are remined
    """

    def __init__(self, name, mode, imsize=None, nnum=5, qsize=2000, poolsize=20000, transform=None,
                 loader=default_loader, dataset_pkl=None, ims_root=None, shuffle=True, swap_qp=False,
                 first_neg="neg"):

        if not (mode == 'train' or mode == 'val'):
            raise(RuntimeError("MODE should be either train or val, passed as string"))

        if name.startswith('retrieval-SfM'):
            # setting up paths
            data_root = get_data_root()
            db_root = os.path.join(data_root, 'train', name)
            ims_root = ims_root or os.path.join(db_root, 'ims')

            # Read db
            if dataset_pkl:
                print('>> Using external dataset for {}: {}...'.format(mode, dataset_pkl))
            db_fn = dataset_pkl or os.path.join(db_root, '{}.pkl'.format(name))
            db = self.load_pkl_dataset(db_fn, mode)

            # setting fullpath for images
            if ims_root and ims_root.endswith(".h5"):
                with h5py.File(ims_root, 'r') as img_data:
                    assert img_data.attrs['storage_type'].tostring().decode("utf8") == "flat_by_cid"
                    self.images = [img_data[x][:] for x in db['cids']]
            else:
                self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        elif name.startswith('gl'):
            ## TODO: NOT IMPLEMENTED YET PROPOERLY (WITH AUTOMATIC DOWNLOAD)

            # setting up paths
            db_root = '/mnt/fry2/users/datasets/landmarkscvprw18/recognition/'
            ims_root = os.path.join(db_root, 'images', 'train')

            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
                db['qsize'] = len(db['qidxs'])

            # setting fullpath for images
            self.images = [os.path.join(ims_root, db['cids'][i]+'.jpg') for i in range(len(db['cids']))]
        else:
            raise(RuntimeError("Unknown dataset name!"))

        # initializing tuples dataset
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.clusters = db['cluster']
        self.db = db

        if swap_qp:
            self.db['qidxs'], self.db['pidxs'] = self.db['pidxs'], self.db['qidxs']

        ## If we want to keep only unique q-p pairs
        ## However, ordering of pairs will change, although that is not important
        # qpidxs = list(set([(self.qidxs[i], self.pidxs[i]) for i in range(len(self.qidxs))]))
        # self.qidxs = [qpidxs[i][0] for i in range(len(qpidxs))]
        # self.pidxs = [qpidxs[i][1] for i in range(len(qpidxs))]

        # size of training subset for an epoch
        self.nnum = nnum
        self.qsize = min(qsize, self.db['qsize'])
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None
        self.tuple_labels = None

        self.transform = transform
        self.loader = loader
        self.shuffle = shuffle
        self.first_neg = first_neg

        self.print_freq = 100

    @classmethod
    def load_pkl_dataset(cls, db_fn, mode):
        # Load file
        if db_fn.startswith("http://") or db_fn.startswith("https://"):
            with urlopen(db_fn) as handle:
                loaded = io.BytesIO(handle.read())
        else:
            with open(db_fn, 'rb') as handle:
                loaded = io.BytesIO(handle.read())

        # Verify
        match = re.search(r'.*-([a-f0-9]{8}[a-f0-9]*)\.pth', db_fn)
        if match:
            stored_hsh = match.group(1)
            computed_hsh = hashlib.sha256(loaded).hexdigest()[:len(stored_hsh)]
            if computed_hsh != stored_hsh:
                raise ValueError("Computed hash '%s' is not consistent with stored hash '%s'" \
                    % (computed_hsh, stored_hsh))

        # Load db
        db = pickle.load(loaded)[mode]
        db['qsize'] = len(db['qidxs'])
        return db

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        # query image
        output.append(self.loader(self.images[self.qidxs[index]]))
        output[-1].info['_metadata']['image_label'] = self.tuple_labels[0][index]
        # positive image
        output.append(self.loader(self.images[self.pidxs[index]]))
        output[-1].info['_metadata']['image_label'] = self.tuple_labels[1][index]
        # negative images
        for i in range(self.first_neg == "exc", len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))
            output[-1].info['_metadata']['image_label'] = self.tuple_labels[2+i][index]

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]

        if self.transform is not None:
            output = [x.unsqueeze_(0) for x in self.transform(*output)]
            # output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]
            # output = [self.transform(output[i]) for i in range(len(output))]

        first_neg = {"neg": [0], "pos": [1], "exc": []}[self.first_neg] if self.nidxs[index] else []
        target = torch.Tensor([-1, 1] + first_neg + [0]*(len(self.nidxs[index])-1))

        return output, target

    def __len__(self):
        # if not self.qidxs:
        #     return 0
        # return len(self.qidxs)
        return self.qsize

    def get_identifier(self, idx):
        return [self.images[x] for x in [self.qidxs[idx], self.pidxs[idx]] + self.nidxs[idx]]

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.db['qidxs']))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    # Mining

    def _randperm(self, size, samples):
        if self.shuffle:
            return torch.randperm(size)[:samples]
        return list(range(size))[:samples]

    def _extract_descriptors(self, idxs, image_labels, net, device):
        if isinstance(image_labels, str):
            image_labels += "-mine"
        else:
            image_labels = [f"{x}-mine" for x in image_labels]

        net.eval()
        with torch.no_grad():
            # prepare query loader
            images = [self.images[i] for i in idxs]
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=images, imsize=self.imsize, transform=self.transform,
                               image_labels=image_labels),
                batch_size=1, shuffle=False, num_workers=6, pin_memory=True
            )
            # extract query vectors
            vecs = torch.zeros(net.meta['out_channels'], len(idxs)).to(device)
            for i, input in enumerate(loader):
                vecs[:, i] = net(input.to(device)).data.squeeze()
                if (i+1) % self.print_freq == 0 or (i+1) == len(idxs):
                    print('\r>>>> {}/{} done...'.format(i+1, len(idxs)), end='')
            print('')
        return vecs

    def _select_positive_pairs(self, net, device):
        # draw qsize random queries for tuples
        idxs2qpool = self._randperm(len(self.db['qidxs']), self.qsize)
        qidxs = [self.db['qidxs'][i] for i in idxs2qpool]
        pidxs = [self.db['pidxs'][i] for i in idxs2qpool]

        tuple_labels = ["anc", "pos", self.first_neg] + ["neg"] * (self.nnum - 1)
        tuple_labels = [[x]*self.qsize for x in tuple_labels]
        return qidxs, pidxs, tuple_labels, {}

    def _search_hard_negatives(self, qidxs, qvecs, idxs2images, poolvecs):
        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():
            print('>> Searching for hard negatives...')
            # compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().to(poolvecs.device)  # for statistics
            n_ndist = torch.tensor(0).float().to(poolvecs.device)  # for statistics
            ndist_acc = []
            # selection of negative examples
            nidxs = []
            for q in range(len(qidxs)):
                # do not use query cluster,
                # those images are potentially positive
                qcluster = self.clusters[qidxs[q]]
                clusters = [qcluster]
                nidx = []
                r = 0
                while len(nidx) < self.nnum:
                    potential = idxs2images[ranks[r, q]]
                    # take at most one image from the same cluster
                    if not self.clusters[potential] in clusters:
                        nidx.append(potential)
                        clusters.append(self.clusters[potential])
                        ndist = torch.pow(qvecs[:,q]-poolvecs[:,ranks[r, q]]+1e-6, 2).sum(dim=0).sqrt()
                        avg_ndist += ndist
                        n_ndist += 1
                        ndist_acc.append(ndist.item())
                    r += 1
                nidxs.append(nidx)
            print('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist/n_ndist))
            print('>>>> Done')
        return nidxs, {"average_negative_distance": ndist_acc} # return average negative l2-distance

    def _select_negatives(self, qidxs, tuple_labels, net, device):
        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            return [[] for _ in range(len(qidxs))], {}

        # draw poolsize random images for pool of negatives images
        idxs2images = self._randperm(len(self.images), self.poolsize)

        # Mine descriptors
        qvecs = self._extract_descriptors(qidxs, tuple_labels[0], net, device)
        poolvecs = self._extract_descriptors(idxs2images, "neg-pool", net, device)
        return self._search_hard_negatives(qidxs, qvecs, idxs2images, poolvecs)

    def create_epoch_tuples(self, net, device=None):
        if not device:
            device = torch.device("cuda")

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        self.qidxs, self.pidxs, self.tuple_labels, pairs_meta = self._select_positive_pairs(net, device)
        self.nidxs, neg_meta = self._select_negatives(self.qidxs, self.tuple_labels, net, device)
        return {**pairs_meta, **neg_meta}
