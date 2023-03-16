import json
from pathlib import Path

def load_json(file):
    path = Path(__file__).parent / "../json_placeholder/{}".format(file)
    with open(path) as f:
        return json.load(f)
    
    
    
    