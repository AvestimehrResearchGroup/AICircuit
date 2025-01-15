from Pipeline.pipeline import pipeline
from Utils.utils import seed_everything
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Train Config File Path', default="./Config/train_config.yaml")
    parser.add_argument('--seed', help='Random Seed', default=0, type=int)
    args = parser.parse_args()
    config_path = args.path

    seed_everything(args.seed)
    pipeline(config_path)
