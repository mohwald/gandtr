import unittest
import re
import os

from pathlib import Path

from test.integration.scenarios import EXAMPLES_PATH
from mdir.examples.perform_scenario import parse_targets
from mdir.tools.utils import load_yaml_scenario


class TestScenarioValidity(unittest.TestCase):
    def test_scenarios_validity(self):
        paths = [str(path)
                 for d in os.listdir(EXAMPLES_PATH)
                 for path in Path(os.path.join(EXAMPLES_PATH, d)).glob('*/*.yml')
                 if not re.match(pattern=re.compile('.*({}_).*|.*(parameters).*$'.format(os.sep)), string=str(path))]
        # Do not test paths, like "*/_*.yml" or "*/parameters/*.yml. These are template scenarios."
        for path in paths:
            with self.subTest(path=str(path).split(os.path.sep)[-2:]):
                scenario = load_yaml_scenario([path])
                targets = parse_targets(scenario, scenario.keys(), path)
                self.assertTrue(targets is not None and len(targets) > 0, msg=path)
