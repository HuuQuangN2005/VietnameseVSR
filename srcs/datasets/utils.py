import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_PATH = os.path.join(PROJECT_PATH, "data")
os.makedirs(DATA_PATH, exist_ok=True)


def setup_hf(dataset_dir: str = DATA_PATH) -> dict:
    hf_path = os.path.join(dataset_dir, ".hf")
    cache_path = os.path.join(hf_path, ".cache")

    configs = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "HF_HOME": hf_path,
        "HF_HUB_CACHE": os.path.join(cache_path, "hub"),
        "HF_DATASETS_CACHE": os.path.join(cache_path, "datasets"),
        "HF_ASSETS_CACHE": os.path.join(cache_path, "assets"),
        "HF_XET_CACHE": os.path.join(cache_path, "xet"),
    }

    for k, v in configs.items():
        if k == "HF_TOKEN":
            if v is None:
                continue
            else:
                os.environ[k] = v
        else:
            os.makedirs(v, exist_ok=True)
            os.environ[k] = v

    return configs
