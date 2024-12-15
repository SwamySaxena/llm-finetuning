import json
import os
from torch.utils.data import Dataset, DataLoader

class SquadDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            squad_data = json.load(f)
        self.samples = self._process_data(squad_data)

    def _process_data(self, squad_data):
        samples = []
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = qa['answers']
                    if len(answers) == 0:
                        continue
                    samples.append((context, question, answers[0]['text']))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, question, answer = self.samples[idx]
        return {
            'input': f"Context: {context}\nQuestion: {question}\nAnswer:",
            'target': answer
        }

def get_data_loader(file_path, batch_size=2):
    dataset = SquadDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
