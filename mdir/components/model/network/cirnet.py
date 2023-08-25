import os.path
from .. import layers
from cirtorch.networks import imageretrievalnet
from cirtorch.utils.general import get_root

# Default cirnet

class CirRetrievalNet(imageretrievalnet.ImageRetrievalNet):
    """Cirtorch official retrieval net"""

    def parameter_groups(self, optimizer_opts):
        parameters = [{'params': self.features.parameters()}]

        # Local whitening
        if self.meta['local_whitening']:
            parameters.append({'params': self.lwhiten.parameters()})

        # Pooling
        if not self.meta['regional']:
            # global, only pooling parameter p weight decay should be 0
            parameters.append({'params': self.pool.parameters(), 'lr': optimizer_opts['lr']*10, 'weight_decay': 0})
        else:
            # regional, pooling parameter p weight decay should be 0,
            # and we want to add regional whitening if it is there
            parameters.append({'params': self.pool.rpool.parameters(), 'lr': optimizer_opts['lr']*10, 'weight_decay': 0})
            if self.pool.whiten is not None:
                parameters.append({'params': self.pool.whiten.parameters()})

        # Final whitening
        if self.whiten is not None:
            parameters.append({'params': self.whiten.parameters()})

        return parameters

    @staticmethod
    def _set_batchnorm_eval(mod):
        if mod.__class__.__name__.find('BatchNorm') != -1:
            # freeze running mean and std
            mod.eval()

    def train(self, mode=True):
        res = super().train(mode)
        if mode:
            self.apply(CirRetrievalNet._set_batchnorm_eval)
        return res


def init_cirnet(**params):
    for key in ["local_whitening", "pooling", "regional", "whitening", "pretrained"]:
        if key not in params:
            raise ValueError("Key '%s' not in params" % key)
    params['mean'] = [0.485, 0.456, 0.406]
    params['std'] = [0.229, 0.224, 0.225]
    params['model_dir'] = os.path.join(get_root(), "weights")
    params['architecture'] = params.pop("cir_architecture")

    net = imageretrievalnet.init_network(params)
    net.meta["in_channels"] = 3
    net.meta["out_channels"] = net.meta["outputdim"]

    pool = net.pool
    if isinstance(params['pooling'], dict):
        pool = layers.pooling.POOLINGS[params['pooling'].pop("type")](**params['pooling'])

    return CirRetrievalNet(net.features, net.lwhiten, pool, net.whiten, net.meta)


# Cirnet with preprocessing

class CirRetrievalNetPreprocessing(CirRetrievalNet):

    def __init__(self, preprocessing, features, lwhiten, pool, whiten, meta):
        super().__init__(features, lwhiten, pool, whiten, meta)
        self.preprocessing = preprocessing

    def forward(self, x):
        return super().forward(self.preprocessing(x))

    def parameter_groups(self, optimizer_opts):
        parameters = super().parameter_groups(optimizer_opts)
        parameters.append({'params': self.preprocessing.parameters(), 'lr': optimizer_opts['lr']*10})
        return parameters


def init_cirnet_inchan(inputs, **params):
    """Initialization of advanced input channels handling (number, preprocessing)"""
    model = init_cirnet(**params)

    if inputs['channels'] == 1:
        model.features[0].in_channels = 1
        model.features[0].weight.data = model.features[0].weight.data.sum(dim=1, keepdim=True)
        model.meta["in_channels"] = model.features[0].in_channels
    elif inputs['channels'] != 3:
        raise NotImplemented("Unsupported number of channels %s" % inputs['channels'])

    if inputs["preprocessing"]:
        model.meta["preprocessing"] = inputs["preprocessing"].pop("type")
        if model.meta["preprocessing"] == "edgefilter":
            preprocessing = layers.preprocessing.EdgeFilter(**inputs["preprocessing"])
        else:
            raise NotImplemented("Unsupported preprocessing type '%s'" % model.meta["preprocessing"])
        model = CirRetrievalNetPreprocessing(preprocessing, model.features, model.lwhiten, model.pool, model.whiten, model.meta)

    return model


# Cirnet with attention

class CirRetrievalNetAttention(CirRetrievalNet):
    """Cirtorch retrieval net with attention before pooling"""

    def __init__(self, attention, features, lwhiten, pool, whiten, meta):
        super().__init__(features, lwhiten, pool, whiten, meta)
        self.attention = attention

    def forward(self, x):
        assert not self.lwhiten and not self.whiten

        feats = self.features(x)
        att = self.attention(feats)
        o = feats * att

        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        return o.permute(1,0)

    def parameter_groups(self, optimizer_opts):
        parameters = super().parameter_groups(optimizer_opts)
        parameters.append({'params': self.attention.parameters(), 'lr': optimizer_opts['lr']*100})
        return parameters


def init_cirnet_attention(attention, **params):
    model = init_cirnet(**params)
    attention = layers.attention.ATTENTIONS[attention.pop("type")](**attention)
    return CirRetrievalNetAttention(attention, model.features, model.lwhiten, model.pool,
                                      model.whiten, model.meta)


if __name__ == "__main__":
    net = init_cirnet(local_whitening=False, pooling="gem", regional=False, whitening=False, pretrained=False, cir_architecture="vgg16")
    print(sum(p.numel() for p in net.parameters()))
