import os
import copy
import torch
import numpy as np

from ..tools import stats
from ..components.data.dataset import initialize_dataset_loader
from ..components.data.output import initialize_output
from ..learning import load_network

# Limit threads
torch.set_num_threads(3)
os.environ['MKL_NUM_THREADS'] = "3"
os.environ['OMP_NUM_THREADS'] = "3"


def infer(params, data):
    # General initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    if not len(data[0]):
        # Speedup nothing-done scenario by not loading the cnn on gpu - can be removed without consequences
        output_tmp = initialize_output(copy.deepcopy(params["output"]["inference"]), copy.deepcopy(params['data']['test']), data)
        if not output_tmp.preprocess()[0]:
            return ({"status": "skipped"},) + output_tmp.postprocess()

    # Load network and dataset
    network = load_network(params["network"], device).eval()
    data_params = {**network.network_params.runtime.get("data", {}), **params['data']['test']}

    # Output init
    output = initialize_output(copy.deepcopy(params["output"]["inference"]), copy.deepcopy(data_params), data)
    data = output.preprocess()
    if not data[0]:
        return ({"status": "skipped"},) + output.postprocess()

    # Data init
    loader = initialize_dataset_loader(data, "test", copy.deepcopy(data_params), {"batch_size": 1})

    # Stats
    meter = stats.AverageMeter("Infer", len(loader), debug=params["output"].get("debug", False))
    resources = stats.ResourceUsage()

    # Get descriptors
    with torch.no_grad():
        forward = getattr(network, params['forward']['method']) if "forward" in params else network
        for i, indata in enumerate(loader):
            if isinstance(indata, dict) and indata == {}:
                output.add(i, None, None)
            else:
                if "forward" in params:
                    out = forward(indata.to(device), **params['forward']['params'])
                else:
                    out = network(indata)
                output.add(i, indata, out)

            # Stats
            if i == len(loader)-1:
                resources.take_current_stats()
            meter.update(i, None)

    metadata = {"stats": meter.total_stats(),
                "resource_usage": resources.get_resources()}
    return (metadata,) + output.postprocess()


def infer_incrementally(params, data):
    # Parse input data
    identifier_existing, value_existing, identifier_new = data
    existing = dict(zip(identifier_existing, value_existing))
    for_inference = [x for x in identifier_new if x not in existing]

    # Infer
    metadata, identifier_added, value_added = infer(params, (for_inference,))

    # Build output data
    added = dict(zip(identifier_added, value_added))
    value_new = [existing[x] if x in existing else added[x] for x in identifier_new]
    if isinstance(value_existing, np.ndarray):
        value_new = np.array(value_new)

    return metadata, identifier_new, value_new
