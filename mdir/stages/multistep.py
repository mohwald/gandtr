import pickle
from . import infer, whiten

from daan.core.path_resolver import resolve_path
from daan.data.fs_driver import fs_driver


def infer_and_learn_whitening(params, data):
    """Perform inference and learn whitening on resulting descriptors"""
    assert not data
    whitening = params.pop("whitening")
    assert whitening.keys() == {"type", "dataset_pkl", "directory"}

    # Handle saving and potential execution skipping
    path = None
    if whitening["directory"]:
        path = fs_driver(resolve_path(whitening["directory"])) / ("whitening/%s-%s.pkl" % \
                (whitening["type"], whitening["dataset_pkl"].rsplit("/", 1)[-1].split("-", 1)[0]))
        if path.exists():
            return {"status": "skipped", "whitening_path": path.path if path else None}, None
        path.makedirs("..")

    pkl = fs_driver(resolve_path(whitening["dataset_pkl"])).load()

    # Infer
    paths = ["/".join([x[-2:], x[-4:-2], x[-6:-4], x]) for x in pkl['cids']]
    metadata_infer, _cids, descriptors = infer.infer(params, (paths,))

    # Learn whitening
    learn_whitening = {
        "lw": whiten.learn_lw_whitening,
        "pca": whiten.learn_pca_whitening,
    }[whitening["type"]]

    qidxs, pidxs = [pkl["cids"][x] for x in pkl["qidxs"]], [pkl["cids"][x] for x in pkl["pidxs"]]
    meatadata_learn_whitening, whit = learn_whitening({}, (pkl['cids'], descriptors, qidxs, pidxs))

    # Store
    if path:
        path.store("", lambda x: pickle.dump(whit, x))

    return {"infer": metadata_infer, "learn_whitening": meatadata_learn_whitening,
            "whitening_path": path.path if path else None}, whit
