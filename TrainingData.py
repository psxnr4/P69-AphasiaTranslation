# @ adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

import torch
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed, DataCollatorForLanguageModeling #BERT mlm
from torch.utils.data import ConcatDataset
import os
import string
from sklearn.model_selection import train_test_split

# Initialize the tokenizer from the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Get BERT masked language model from Hugging Face
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model.eval()

directory_root = 'C:/Users/nat/OneDrive - The University of Nottingham/Documents/Dissertation/P69 - Aphasia Project/4. Analysis and Results'
control_directory = directory_root + '/Capilouto-Umbrella-Gem/Cleaned-Flo'
aphasia_directory = directory_root + '/Aphasia Training Data/Repaired-Flo'

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
            if idx < len(self.target_ids):
                item['target_ids'] = torch.tensor(self.target_ids[idx])
            else:
                raise IndexError(f"Index {idx} out of range for target_ids of length {len(self.target_ids)}")
        return item


# Get suitable datasets from hardcoded directories based on passed flags
def get_training_data( control_data_flag, aphasia_data_flag, minimum_context_length):
    datasets = []

    if control_data_flag:
        print("Getting control data..")
        control_dataset = dataset_from_directory(control_directory, minimum_context_length)
        datasets.append(control_dataset)

    if aphasia_data_flag:
        print("Getting aphasia data..")
        aphasia_dataset = dataset_from_directory(aphasia_directory, minimum_context_length)
        datasets.append(aphasia_dataset)

    # Combine if more than one, else use single dataset
    # Concat datasets to prevent resetting the optimisers between training sets
    final_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # Randomly mask tokens and prepare to be processed
    dataloader = load_dataset(final_dataset)
    return dataloader



def dataset_from_directory(dir_path, minimum_context_length):
    # Create 2d array of all data
    all_data = []
    # Get all .cex files in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.cex'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
               data = f.read()
            # Separate each utterance to be processed individually
            data = data.split('\n')
            # Add context to each utterance
            data_w_context = add_context(data, minimum_context_length)
            all_data.extend(data_w_context)

    print(type(all_data), len(all_data))
    print(all_data[:5])

    # -- Tokenise data
    inputs = tokenizer(
        all_data, max_length=512, truncation=True, padding=True, return_tensors=None
    )  # keys: input_ids, token_type_ids, attention_mask

    # Create dataset to define how to load and batch the tokenized data
    dataset = TextDataset(inputs)

    print(f" -- Created Dataset size: {len(dataset)}")

    return dataset




# Read in training data from the given path, tokenise and create dataset
def dataset_from_file(file_path):
    # Read test data as each line a separate item in a list
    with open(file_path, 'r', encoding='utf-8') as f:
        train_data = [line.rstrip().rstrip('.') for line in f] # remove trailing periods

    # Convert to list of transcripts, splitting at each transcript break delimiter
    # content = ' '.join(lines) # remove line breaks
    #train_data = content.split('|')

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


# Add previous and trailing lines to each utterance to introduce context
# Build an adaptive context window around the masked line of at least min number of words
# text: array of strings
def add_context(text, min_size):
    text_w_context = []
    for index in range(len(text)):
        line_w_context = text[index].split()
        length = len(line_w_context)
        buffer = 1

        while length < min_size:
            prev_line = text[index - buffer].split() if index - buffer >= 0 else []
            next_line = text[index + buffer].split() if index + buffer < len(text) else []
            buffer += 1

            # Add context around the line
            line_w_context = prev_line + ['[SEP]'] + line_w_context + ['[SEP]'] + next_line
            length = len(line_w_context)

        # Join the combined lines into a single string and add to array
        final_line = ' '.join(line_w_context)
        text_w_context.append(final_line)

        #print('\n')
        #for l in text_w_context:
        #    print(l)
    return text_w_context


