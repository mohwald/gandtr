from . import cirscore, visual

SCORES = {
    "cirdatasetap": cirscore.CirDatasetAp,
    "visual": visual.VisualDataset,
}

def initialize_score(params):
    return SCORES[params.pop("type")](params)
