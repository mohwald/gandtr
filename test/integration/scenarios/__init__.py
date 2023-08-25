import os
import re

EXAMPLES_PATH = re.sub("gandtr" + '.*', os.path.join("gandtr", "mdir", "examples"), os.path.realpath(__file__))
