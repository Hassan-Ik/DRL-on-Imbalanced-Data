from agent import Agent
from dataset import Dataset, CassavaLeafDataset, PersonalityDataset
from network import QNetwork
from memory import Memory
from option import config

if __name__ == '__main__':
    if config.dataset_name == 'cassava':
        dataset = CassavaLeafDataset(image_size = (128, 128), batch_size=config.batch)
        q_network = QNetwork(config, 128, 'complex')
        memory = Memory()
        agent = Agent(q_network, dataset, memory, config)
        agent.train()
    elif config.dataset_name == 'personality':
        config.new_class = "{0:[0], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5], 6:[6], 7:[7], 8:[8], 9:[9], 10:[10], 11:[11], 12:[12], 13:[13], 14:[14], 15:[15]}"
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