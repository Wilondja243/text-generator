import yaml
from pathlib import Path

# function to return a dict of config
def config_load(filename="config.yaml"):
    config_path = Path(__file__).parent.parent / "configs" / filename

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

        return config

