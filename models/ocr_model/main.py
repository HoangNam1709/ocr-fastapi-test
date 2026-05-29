import argparse

import yaml

from layout import PPDocLayoutV3Model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract ID")
    parser.add_argument(
        "--config", 
        default="./config.yaml",
        type=str,
        help="Path to yaml config file"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    args = parser.parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    layout_config = config["pipeline"]
    layout_model = PPDocLayoutV3Model(layout_config)
    
    result = layout_model.extract_card()
    print(result)
