"""
Functions for resolving paths that can be e.g. host-dependent by regex-defined rules.
"""

import os.path
import re
from socket import gethostname

#: A set of rules that define the path-rewriting, the value substitutes the key that is regex
PATH_RULES = {"^": "$CIRTORCH_ROOT/"}

def _match_string_once(patterns, string):
    """Given a list of patters, return the exact one that matches the string, or raise an error"""
    matched = [x for x in patterns if re.search(x, string)]
    if not matched:
        raise ValueError("No pattern (%s) matched string '%s'" % (", ".join(patterns), string))
    elif len(matched) > 1:
        raise ValueError("More than one pattern (%s) matched string '%s'" % \
                (", ".join(matched), string))
    return matched[0]

def set_rules(rules):
    """Replace stored rules by given rules"""
    global PATH_RULES # pylint: disable=global-statement
    PATH_RULES = rules

def set_rules_by_key(rules, key):
    """Given multiple set of rules in a dictionary where the key is a regex, find which dict key
        matches given key and set the corresponding value as global rules. If rules not provided,
        do nothing."""
    if rules:
        set_rules(rules[_match_string_once(rules, key)])

def set_rules_by_hostname(rules):
    """Same as set_rules_by_key, except the key is current hostname"""
    set_rules_by_key(rules, gethostname())

def resolve_path(path):
    """Using the PATH_RULES, resolve (rewrite) given path. PATH_RULES is a dictionary whre the key
        is a regex that gets substituted for the value. In rewritten path, shell variables are
        expanded."""
    if isinstance(path, list):
        return [resolve_path(x) for x in path]
    if isinstance(path, dict):
        return {x: resolve_path(y) for x, y in path.items()}

    if not isinstance(path, str) or not path or path[0] == "/" \
            or path.startswith("http://") or path.startswith("https://"):
        return path

    pattern = _match_string_once(PATH_RULES, path)
    res, nsubs = re.subn(pattern, PATH_RULES[pattern], path)
    assert nsubs == 1, nsubs
    return os.path.expandvars(res)
