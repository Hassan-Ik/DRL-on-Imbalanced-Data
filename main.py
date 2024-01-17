from DDQNAgent import Agent
from Datasets import Dataset, CassavaLeafDataset, PersonalityDataset
from QNetwork import QNetwork
from Memory import Memory
import argparse
import ast

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset_name', type=str, default='cassava', help='Name which dataset to use for DRL classification')
    parser.add_argument('--new_class', type=ast.literal_eval, default='{0:[0], 1:[1], 2:[2], 3: [3], 4:[4]}',
                        help='Change existing class in Cifar10 data to new class')
    parser.add_argument('--minor_classes', type=ast.literal_eval, default=[0, 1, 2],
                        help='Classes to be used as minor classes among the newly changed classes')
    parser.add_argument('--minority_subsample_rate', type=float, default=0.16,
                        help='Imbalance ratio = Ratio of the amount of data in major and minor classes before data subsampling X minority_subsample_rate')

    # Train
    parser.add_argument('--train_step', type=int, default=100000, help='Total train steps')
    parser.add_argument('--restore_model_path', type=str, default='', help='For transfer learning')
    parser.add_argument('--gamma', type=float, default=0.1, help='Reinforcement learning parameter')
    parser.add_argument('--learning_rate', type=float, default=0.00035, help='Learning rate')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--epsilon_range', type=ast.literal_eval, default=[0.01, 1], help='Exploration range')
    parser.add_argument('--epsilon_polynomial_decay_step', type=int, default=120000,
                        help='Steps taken for epsilon to reach minimum value')
    parser.add_argument('--target_soft_update', type=float, default=1., help='Rate of updating the target q network')
    parser.add_argument('--target_update_step', type=int, default=1000, help='Period to update the target q network')

    # Log
    parser.add_argument('--save_folder', type=str, default='./model', help='Folder for saving trained models')
    parser.add_argument('--save_term', type=int, default='100000', help='Period to save the trained model')
    parser.add_argument('--evaluation_term', type=int, default=500, help='Period to show the progress of training')
    parser.add_argument('--show_phase', type=str, default='Validation', choices=['Validation', 'Train', 'Both'],
                        help='Dataset to be evaluated during training')

    config = parser.parse_args()


    if config.dataset_name == 'cassava':
        dataset = CassavaLeafDataset(image_size = (128, 128), batch_size=config.batch)
        q_network = QNetwork(config, 128, 'complex')
        memory = Memory()
        agent = Agent(q_network, dataset, memory, config)
        agent.train()
    elif config.dataset_name == 'personality':
        dataset = PersonalityDataset(batch_size=config.batch)
        q_network = QNetwork(config, 60, 'simple')
        memory = Memory()
        agent = Agent(q_network, dataset, memory, config)
        agent.train()
    else:
        dataset = Dataset(config)
        q_network = QNetwork(config, 32, "")
        memory = Memory()
        agent = Agent(q_network, dataset, memory, config)
        agent.train()
