import pandas as pd 
import os
import random

random.seed(14)

def domain_adaptation_dataset(train_val_domains, test_domains, output_folder, train_ratio=0.8):
    """
    Create a new metadata.csv file for domain adaptation.

    :param train_val_domains: List of paths to metadata.csv files for training and validation domains.
    :param test_domains: List of paths to metadata.csv files for testing domains.
    :param output_folder: Path to the folder where the new metadata.csv file will be saved.
    :param train_ratio: Ratio of training data in the train/val dataset (default is 0.8 for 80%).
    """

    # Create train/val splits
    train_val_data = pd.concat([pd.read_csv(file) for file in train_val_domains])
    train_val_data['split'] = ['train' if random.random() < train_ratio else 'val' for _ in range(len(train_val_data))]

    # Create test split
    test_data = pd.concat([pd.read_csv(file) for file in test_domains])
    test_data['split'] = 'test'

    # Combine
    combined_data = pd.concat([train_val_data, test_data])
    
    # Write new metadata.csv
    output_path = os.path.join(output_folder, 'metadata.csv')
    combined_data.to_csv(output_path, index=False)

def main(train_val_domains = ["/home/jennamansueto/data/tx_2000_mean_vv_vh_rgb/metadata.csv"],
    test_domains = ["/home/jennamansueto/data/la_2000_mean_vv_vh_rgb/metadata.csv"],
    output_folder = "/home/jennamansueto/data/source_tx_target_la"
    ):

    domain_adaptation_dataset(train_val_domains, test_domains, output_folder)

if __name__ == "__main__":
    main()