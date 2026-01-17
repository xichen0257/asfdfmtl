from data_handling.celeba import DMCelebA
from data_handling.cifar import DMCifar
from data_handling.nyuv2 import DMNYUDepthV2
from data_handling.pascal_context import DMPascalContext
from typing import List, Dict, Optional

"""
Helper script to streamline the retrieval of datamanagers.
"""


def get_data_manager(dataset, task_1_clients_ids, task_2_clients_ids, dataset_fraction):
    """Intializes and returns a dataset specific data manager.

    Args:
        dataset (str): The name of the dataset
        task_1_clients_ids (str): The client ids belonging to the first task.
        task_2_clients_ids (str): The client ids belonging to the second task.
        tasksplits ([x,x]]): How the data shall be split among task. For instance [0.5, 0.5] results in an equal split.

    Returns:
        DataManager: Returns a data manager for a specific dataset and its two task types.
    A data manager has train, val and test loaders for each client. The loaders are different for each client.
    There exists no overlap between train and validation data among clients. Test data is the same among clients belonging to the same task.
    """

    print("Getting Data Manager, This might take a few minutes")
    if dataset == "cifar10":
        data_manager = DMCifar(
            seed=1,
            an_num_clients=len(task_1_clients_ids),
            ob_num_clients=len(task_2_clients_ids),
            dataset_fraction=dataset_fraction,
        )  # datasetsplit should not sum up to more than 1!
    elif dataset == "celeba":
        data_manager = DMCelebA(
            seed=1,
            ml_num_clients=len(task_1_clients_ids),
            fl_num_clients=len(task_2_clients_ids),
            dataset_fraction=dataset_fraction,
        )  # datasetsplit should not sum up to more than 1!
    else:
        raise "Dataset not properly defined!"
    return data_manager


def get_multitask_data_manager(
    dataset: str,
    num_clients: int,
    task_weights_per_client: List[Dict[str, float]],
    dataset_fraction: float = 1.0,
    seed: int = 1,
    batch_size: int = 8
):
    """
    Initialize and return a multi-task data manager (for NYU Depth V2, Pascal Context, etc.)

    Args:
        dataset: Name of the dataset ('nyuv2', 'pascal_context')
        num_clients: Number of clients
        task_weights_per_client: List of task weight dicts for each client
            Example for NYU V2: [
                {'depth': 0.7, 'segmentation': 0.3, 'normal': 0.0},
                {'depth': 0.0, 'segmentation': 0.7, 'normal': 0.3},
                ...
            ]
            Example for Pascal Context: [
                {'segmentation': 0.7, 'human_parts': 0.3, 'edge': 0.0},
                {'segmentation': 0.0, 'human_parts': 0.7, 'edge': 0.3},
                ...
            ]
        dataset_fraction: Fraction of dataset to use
        seed: Random seed
        batch_size: Batch size for data loaders

    Returns:
        DataManager: Multi-task data manager with train/val/test loaders for each client
    """
    print(f"Getting Multi-Task Data Manager for {dataset}")
    print("This might take a few minutes (downloading if needed)...")

    if dataset == "nyuv2":
        data_manager = DMNYUDepthV2(
            seed=seed,
            num_clients=num_clients,
            task_weights_per_client=task_weights_per_client,
            dataset_fraction=dataset_fraction,
            batch_size=batch_size,
            download=True
        )
    elif dataset == "pascal_context":
        data_manager = DMPascalContext(
            seed=seed,
            num_clients=num_clients,
            task_weights_per_client=task_weights_per_client,
            dataset_fraction=dataset_fraction,
            batch_size=batch_size,
            download=True
        )
    else:
        raise ValueError(f"Multi-task dataset '{dataset}' not supported. "
                        f"Available: 'nyuv2', 'pascal_context'")

    return data_manager
