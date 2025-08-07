# @ adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

import torch
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed, DataCollatorForLanguageModeling #BERT mlm
from torch.utils.data import ConcatDataset
import string
from sklearn.model_selection import train_test_split

# Initialize the tokenizer from the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Get BERT masked language model from Hugging Face
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model.eval()

# Define training dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, target_ids=None):
        self.encodings = encodings
        self.target_ids = target_ids # list of strings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}

        if self.target_ids is not None:
            item['target_ids'] = torch.tensor(self.target_ids[idx])

        return item


def get_training_data():
    # Create dataset from the control transcripts
    control_file_path = '../control_training_combined_output.cex'
    control_dataset = dataset_from_file(control_file_path)

    # Create dataset from the repaired aphasia transcripts
    aphasia_file_path = '../aphasia_training_combined_output.cex'
    aphasia_dataset = dataset_from_file(aphasia_file_path)

    # Concat datasets to prevent resetting the optimisers between training sets
    combined_dataset = ConcatDataset([control_dataset, aphasia_dataset])
    # Randomly mask tokens and prepare to be processed
    dataloader = load_dataset(combined_dataset)
    return dataloader



# Read in training data from the given path, tokenise and create dataset
def dataset_from_file(file_path):
    # Read test data
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip().rstrip('.') for line in f] # remove trailing periods
    content = ' '.join(lines) # remove line breaks

    # Convert to list, splitting at each transcript break delimiter
    train_data = content.split('|')
    #print(train_data)

    # -- Tokenise data
    inputs = tokenizer(
        train_data, max_length=512, truncation=True, padding=True, return_tensors=None
    ) # keys: input_ids, token_type_ids, attention_mask

    # Create dataset to define how to load and batch the tokenized data
    dataset = TextDataset(inputs)
    return dataset



def load_dataset(dataset):
    # Hugging Face’s masking collator to automatically mask 15% of the tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=data_collator
    )

    return dataloader




