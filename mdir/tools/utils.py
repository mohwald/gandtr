import torch
import os.path
import copy
import json
import re
from functools import reduce
import yaml

from mdir.tools.download import download_and_load_pretrained
from daan.core.experiments import dict_deep_overlay, get_deeply


def indent(string, indent=1):
    return string.replace("\n", "\n" + "    " * indent)

def deep_set(params, deep_key, value):
    reduce(lambda x, y: x.setdefault(y, {}), deep_key[:-1], params)[deep_key[-1]] = value
    return params


def load_yaml_scenario(scenarios):
    """Load scenarios deeply"""
    if scenarios[0].endswith(".yml"):
        with open(scenarios[0], 'r') as handle:
            scenario = yaml.safe_load(handle)
    elif "=" in scenarios[0]:
        # Parse the section.subsection.key=value command-line parameters
        deep_key, value = scenarios[0].split("=")
        deep_key, value = deep_key.split("."), json.loads(value)
        scenario = deep_set({}, deep_key, value)

    if scenarios[1:]:
        scenario = dict_deep_overlay(scenario, load_yaml_scenario(scenarios[1:]))
    if scenarios[0].endswith(".yml"):
        scenario = load_nested_templates(scenario, os.path.dirname(scenarios[0]))
    return scenario


def load_nested_templates(params, root_path):
    """Find keys '__template__' in nested dictionary and replace corresponding value with loaded
        yaml file"""
    if not isinstance(params, dict):
        return params

    if "__template__" in params:
        # Handle deep keys
        for key in list(params.keys()):
            if "." in key:
                deep_set(params, key.split("."), params.pop(key))

        # Handle template
        path = os.path.join(root_path, params.pop("__template__"))
        root_path = os.path.dirname(path)
        with open(path, "r") as handle:
            template = yaml.safe_load(handle)
        params = dict_deep_overlay(template, params)

    for key, value in params.items():
        # copy() fixes shared references - necessary as the parameter dictionary gets altered
        params[key] = load_nested_templates(copy.copy(value), root_path)

    return params


def _resolve_single_variable(hit, data, reference):
    """Expand single variable given by the argument hit that occured in data. Use reference
        for the variable expansion."""
    try:
        var_value = copy.deepcopy(get_deeply(reference, hit.split("."), support_list=True))
    except KeyError:
        raise ValueError("Variable '%s' in '%s' cannot be expanded in context '%s'" % \
                (hit, data, reference))

    var_value = resolve_variables(var_value, reference)
    return var_value if data == "${%s}" % hit else data.replace("${%s}" % hit, str(var_value))

def resolve_variables(data, reference):
    """Resolve variables deeply."""
    if isinstance(data, str):
        for hit in sorted(set(re.findall(r'\$\{([A-Za-z_\-0-9.]+)\}', data)), reverse=True):
            data = _resolve_single_variable(hit, data, reference)
    elif isinstance(data, dict):
        for key, value in list(data.items()):
            newkey = resolve_variables(key, reference)
            if newkey != key:
                del data[key]
            data[newkey] = resolve_variables(value, reference)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = resolve_variables(value, reference)

    return data


def splitp(seq, sep, pairs=("()", "[]", "{}"), check_valid_pairs=False):
    """Split seq by sep without splitting parts inside any of pairs."""
    acc = [""]
    lpairs, rpairs = zip(*pairs)
    pair_stack = []
    for ch in seq:
        if ch == sep and len(pair_stack) == 0:
            acc += [""]
        else:
            if ch in lpairs:
                pair_stack.append(ch)
            elif len(pair_stack) > 0 and ch in rpairs and pair_stack[-1] == lpairs[rpairs.index(ch)]:
                pair_stack.pop()
            acc[-1] += ch
    if check_valid_pairs:
        assert len(pair_stack) == 0,\
            "Invalid seq \"{}\" with pairs={} resulting in pair_stack={}".format(seq, pairs, pair_stack)
    return acc


def load_pretrained(path):
    if path.startswith("http://") or path.startswith("https://"):
        return download_and_load_pretrained(path)
    else:
        return torch.load(path, map_location=lambda storage, loc: storage)

