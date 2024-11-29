from typing import List, Dict
from itertools import product
import yaml


def load_config_from_yaml(config_path:str)-> Dict:
    """Load configuration from a YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def flatten_config(config: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """
    Flatten a nested dictionary for easier processing.
    
    Args:
        config (Dict): Configuration dictionary.
        parent_key (str): Key to prepend to the current level (used for recursion).
        sep (str): Separator to use for flattened keys.

    Returns:
        Dict: Flattened dictionary.
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_config(flat_config: Dict, sep: str = ".") -> Dict:
    """
    Unflatten a dictionary into its original nested structure.
    
    Args:
        flat_config (Dict): Flattened dictionary.
        sep (str): Separator used for flattened keys.

    Returns:
        Dict: Nested dictionary.
    """
    nested = {}
    for key, value in flat_config.items():
        parts = key.split(sep)
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested


def generate_param_combinations(config: Dict, ignore_keys: List[str]) -> List[Dict]:
    """
    Generate all combinations of parameter values from the config,
    ignoring specified keys in a nested structure.

    Args:
        config (Dict): Configuration dictionary.
        ignore_keys (List[str]): Keys to ignore when generating combinations.

    Returns:
        List[Dict]: List of parameter combinations.
    """
    ignore_keys = get_full_lowest_level_keys(config,ignore_keys)
    flat_config = flatten_config(config)

    # Separate sweepable parameters and fixed parameters
    sweep_params = {k: v for k, v in flat_config.items() if isinstance(v, list) and k not in ignore_keys}
    fixed_params = {k: v for k, v in flat_config.items() if k in ignore_keys or not isinstance(v, list)}

    # Generate combinations for sweepable parameters
    param_combinations = list(product(*sweep_params.values()))
    param_keys = list(sweep_params.keys())

    # Merge fixed parameters with each combination
    experiments = []
    for combination in param_combinations:
        experiment_config = fixed_params.copy()
        experiment_config.update(dict(zip(param_keys, combination)))
        experiments.append(unflatten_config(experiment_config))

    return experiments

def pretty_print_config(config: Dict, indent: int = 0) -> None:
    """
    Pretty-print a nested dictionary configuration.

    Args:
        config (Dict): Configuration dictionary to print.
        indent (int): Current indentation level for nested keys.
    """
    for key, value in config.items():
        if isinstance(value, dict):  # If the value is a nested dictionary
            print(" " * indent + f"{key}:")
            pretty_print_config(value, indent + 2)  # Recursive call with increased indentation
        else:  # Handle primitive values
            print(" " * indent + f"{key}: {value}")

def filter_hyperparameters(config: Dict, ignore_keys: List[str]) -> Dict:
    """
    Filter out ignored keys from a nested dictionary.

    Args:
        config (Dict): Configuration dictionary.
        ignore_keys (List[str]): Keys to ignore.

    Returns:
        Dict: Filtered dictionary.
    """
    ignore_keys = get_full_lowest_level_keys(config,ignore_keys)
    flat_config = flatten_config(config)
    filtered_config = {k: v for k, v in flat_config.items() if k not in ignore_keys}
    return unflatten_config(filtered_config)

def extract_hyperparameters(config: Dict, extracted_keys: List[str]) -> Dict:
    """
    Extract keys from a nested dictionary.

    Args:
        config (Dict): Configuration dictionary.
        extracted_keys (List[str]): Keys to extract.

    Returns:
        Dict: Filtered dictionary.
    """
    extracted_keys = get_full_lowest_level_keys(config,extracted_keys)
    flat_config = flatten_config(config)
    filtered_config = {k: v for k, v in flat_config.items() if k in extracted_keys}
    return unflatten_config(filtered_config)



def get_full_lowest_level_keys(data, keys,sep:str='.'):
    """
    Extract full keys with only the lowest level key from a nested dictionary based on input keys.

    Args:
    - data (dict): The nested dictionary to process.
    - keys (list of str): Keys to use as parents to retrieve the lowest-level keys.

    Returns:
    - List[str]: A list of full keys representing the lowest level for each key path.
    """
    def extract_lowest_level_keys(data, parent_key):
        """
        Helper function to extract only the lowest-level keys starting from a specific parent key.
        """
        lowest_keys = []
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):  # Recurse if it's a nested dictionary
                lowest_keys.extend(extract_lowest_level_keys(value, full_key))
            else:  # Add the key if it's the lowest level
                lowest_keys.append(full_key)
        return lowest_keys

    # Result to store all matching keys
    result_keys = []

    # Process each input key
    for key in keys:
        # Traverse the dictionary to find the starting node for the given key
        parts = key.split(sep)
        current = data
        for part in parts:
            if part in current and isinstance(current, dict):
                current = current[part]
            else:
                current = None
                break
        # If the key exists in the dictionary, extract the lowest-level keys
        if current is not None:
            if isinstance(current, dict):
                result_keys.extend(extract_lowest_level_keys(current, key))
            else:
                result_keys.append(key) # keys that contain the lowest level keys

    return result_keys


def string_tuple_to_tuple(string_tuple):
    return tuple(map(int, string_tuple.strip("()").split(",")))


if __name__ == "__main__":
    print(f"run your quick test here for all functions/classes in this script {__file__}.")
    print("-----------------------------------------------------------")
    config = load_config_from_yaml("config/cnn.yaml")
    filtered_config = extract_hyperparameters(config, extracted_keys=[
        'model.shape_in',
        'train.mlflow',
        'train.skip_loading',
        'train.data',
        'train.device',
        'train.verbose',
        'train.save_path',
        'deploy',
        'eval'])
    pretty_print_config(filtered_config)