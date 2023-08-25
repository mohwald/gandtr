#!/usr/bin/env python3
"""
Perform specified target from a .yml scenario.
"""

import os.path
import sys
import argparse
import yaml

sys.path.append(os.path.abspath(__file__ + "/../../../"))

from mdir.tools.utils import load_yaml_scenario, resolve_variables
from mdir.tools.download import rsfm120k, roxf5k_rpar6k_247tokyo1k
import mdir.stages
from cirtorch.utils import download, general


def print_scores(parameters, _data):
    scores = {
        "roxford5k/validation/score_avg:map_medium": "roxford.5k medium",
        "rparis6k/validation/score_avg:map_medium": "rparis.6k medium",
        "247tokyo1k/validation/score_avg:map": "247tokyo.1k",
        "val/validation/roxford5k/score_avg:map_medium": "roxford.5k medium",
        "val/validation/rparis6k/score_avg:map_medium": "rparis.6k medium",
        "val/validation/val_eccv20/score_avg:map": "validation eccv20",
    }
    losses = [
        "val/validation/loss_avg:dist",
    ]
    assert parameters.keys() == {"metadata"}, parameters.keys()
    for heading, section in parameters["metadata"].items():
        print("\n%s\n" % heading.capitalize())
        for key, value in section.items():
            if key in scores:
                print("    %-20s %s" % (scores[key], round(100*value, 2)))
            for loss in losses:
                if loss in key:
                    print("    %-20s %s" % (key.split(":")[-1], round(float(value.tolist()), 8)))
        print()
    return {},


FUNCTIONS = {
    "mdir.stages.train.train": mdir.stages.train.train,
    "mdir.stages.validate.validate": mdir.stages.validate.validate,
    "mdir.stages.infer.infer": mdir.stages.infer.infer,
    "mdir.stages.multistep.infer_and_learn_whitening": mdir.stages.multistep.infer_and_learn_whitening,
    "cirtorch.utils.download.download_train": lambda x, y: (download.download_train(general.get_data_root()),),
    "cirtorch.utils.download.download_test": lambda x, y: (download.download_test(general.get_data_root()),),
    "mdir.utils.download.rsfm120k": lambda x, y: (rsfm120k(general.get_data_root()),),
    "mdir.utils.download.roxf5k_rpar6k_247tokyo1k": lambda x, y: (roxf5k_rpar6k_247tokyo1k(general.get_data_root()),),
    "print_scores": print_scores,
}
NEEDS_DATA = { "mdir.stages.infer.infer" }


def parse_targets(scenario, targets, path):
    acc = []
    for target in targets:
        target_acc = []
        steps = {x: scenario[target][x] for x in sorted(scenario[target]) if x[0] != "_"}
        for step, section in steps.items():
            # Check keys
            function = section.pop("__function__", None)
            if function not in FUNCTIONS:
                raise ValueError("Step '%s' in target '%s' needs undefined function '%s'" \
                        % (step, target, section.pop("__function__", None)))

            target_acc.append((step, function, section))
        acc.append((target, target_acc))
    return acc


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Executes specified targets from a scenario.")
    parser.add_argument("target", help="targets from a scenario or 'list' or 'validate'")
    parser.add_argument("--print-steps", action="store_true", help="instead of executing, " \
        "print input parameters for all steps in a target")
    parser.add_argument("scenario", metavar="scenario.yml",
                        help='path to a scenario in .yml format')
    parser.add_argument("scenarios", metavar="overlay_scenario.yml", nargs='*',
                        help='scenario overlays (inheritance-style)')
    args = parser.parse_args()

    # Parse scenarios
    paths = [args.scenario] + args.scenarios
    if not args.scenario.endswith(".yml"):
        path_pieces = args.scenario.split(".")
        paths[0] = os.path.abspath(__file__ + "/../%s.yml" % "/".join(path_pieces))
    scenario = load_yaml_scenario(paths)

    # Listing targets only
    if args.target == "list":
        print()
        for target in scenario:
            doc = " - %s" % scenario[target]["__doc__"] if "__doc__" in scenario[target] else ""
            print("-", target + doc)
        print()
        sys.exit(0)

    # Validate scenario only
    if args.target == "validate":
        parse_targets(scenario, scenario.keys(), paths[0])
        sys.exit(0)

    # Print steps
    targets = parse_targets(scenario, [args.target], paths[0])
    if args.print_steps:
        target, = targets
        sys.stdout.write(yaml.safe_dump({x: z for x, y, z in target[1]}))
        sys.exit(0)

    # Perform targets
    context = {"SCENARIO_NAME": os.path.splitext(os.path.basename(paths[-1]))[0]}
    for target, steps in targets:
        for step, function, parameters in steps:
            context[step] = {"function": function, "parameters": parameters}

            # Load stdin data
            data = ()
            if function in NEEDS_DATA and not sys.stdin.isatty():
                data = ([x.strip() for x in sys.stdin],)

            # Perform
            params = resolve_variables(parameters, context)
            print("-- %s --" % step)
            output_metadata, *_output_data = FUNCTIONS[function](params, data)
            context[step]["_output_metadata"] = output_metadata


if __name__ == "__main__":
    main()
