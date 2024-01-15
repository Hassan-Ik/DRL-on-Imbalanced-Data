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