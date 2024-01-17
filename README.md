# DRL-on-Imbalanced-Data


In this repository, we are using three diferent types of dataset and applying `Double Deep Q Network` to these dataset to check if Reinforcement learning can be able to perform better on an imbalanced dataset to be able to use in any real world applications.

To run the Agent training for different datasets try following

### For Cassava

`python main.py --dataset_name 'cassava' --new_class '{0:[0], 1:[1], 2:[2], 3:[3], 4:[4]}'`

### For Personality

`python main.py --dataset_name 'personality' --new_class '{0:[0], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5], 6:[6], 7:[7], 8:[8], 9:[9], 10:[10], 11:[11], 12:[12], 13:[13], 14:[14], 15:[15]}'`

### For Cifar10 Dataset

`python main.py --dataset_name 'cifar10' --new_class '{0:[0], 1:[1], 2:[2], 3:[3], 4:[4]}'`