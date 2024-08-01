from .scorer import scorer

import json
import os


MODEL2MAXLEN = json.load(open(os.path.join(os.path.dirname(__file__), "config/model2maxlen.json"), "r"))
DATASET2PROMPT = json.load(open(os.path.join(os.path.dirname(__file__), "config/dataset2prompt.json"),"r"))
DATASET2MAXLEN = json.load(open(os.path.join(os.path.dirname(__file__), "config/dataset2maxlen.json"), "r"))
