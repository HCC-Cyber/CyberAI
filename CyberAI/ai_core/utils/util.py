import argparse
from datetime import datetime
import os
from pathlib import Path
import uuid

import yaml


class Util:
    @staticmethod
    def get_process_dir():

        root = Path(__file__).resolve().parent.parent.parent
        return f"{root}{os.sep}"
    
    @staticmethod
    def get_config_file_path():
        default_config_file = "config.yaml"
        if os.path.exists(Util.get_process_dir() + "data/." + default_config_file):
            config_file = "data/." + default_config_file
        else:
            config_file = Util.get_process_dir() + default_config_file
        return config_file
    
    @staticmethod
    def get_config():
        parser = argparse.ArgumentParser(description="Server Configuration")
        config_file = Util.get_config_file_path()
        parser.add_argument("--config_path", type=str, default=config_file, help="Path to the configuration file")
        args = parser.parse_args()
        print(f"Loading configuration from {args.config_path}")
        with open(args.config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        Util.init_output_dirs(config)
        return config
    
    @staticmethod
    def init_output_dirs(config):
        results = set()
        def _traverse(data):
            if isinstance(data, dict):
                if "output_dir" in data:
                    results.add(data["output_dir"])
                for value in data.values():
                    _traverse(value)
            elif isinstance(data, list):
                for item in data:
                    _traverse(item)

        _traverse(config)
        for dir_path in results:
            try:
                os.makedirs(Util.get_process_dir() + dir_path,exist_ok=True)
            except FileExistsError:
                print(f"warning: 无法创建目录 {dir_path}.")

    @staticmethod
    def get_random_file_path(dir: str, ex_name: str):

        file_name = f"{datetime.now().date()}_{uuid.uuid4().hex}.{ex_name}"
        file_path = os.path.join(dir, file_name)
        return file_path


if __name__ == "__main__":
    print(Util.get_process_dir())
    # Util.get_config()
    print( Util.get_random_file_path(Util.get_process_dir() , "mp3"))
