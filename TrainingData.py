# @ adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/
# create dataset from a directory of .cex files

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
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item


# Get suitable datasets from hardcoded directories based on passed flags
def get_training_data( control_data_flag, aphasia_data_flag, max_context_length, batch_size):
    #print("--------- Preparing training data ----------")
    datasets = []

    if control_data_flag:
        print("Getting control data..")
        control_dataset = dataset_from_directory(control_directory, max_context_length)
        datasets.append(control_dataset)

    if aphasia_data_flag:
        print("Getting aphasia data..")
        aphasia_dataset = dataset_from_directory(aphasia_directory, max_context_length)
        datasets.append(aphasia_dataset)

    # Combine if more than one, else use single dataset
    # Concat datasets to prevent resetting the optimisers between training sets
    final_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # Randomly mask tokens and prepare to be processed
    dataloader = load_dataset(final_dataset, batch_size)
    return dataloader



def dataset_from_directory(dir_path, max_context_length):
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
            if max_context_length:
                # Add context to each utterance
                #print("Adding context to utterances..")
                data_w_context = add_context(data, max_context_length) # returns list of strings

                all_data.extend(data_w_context)  # add elements to array
            else:
                data = ' '.join(data)
                all_data.append(data) # add string to end of array

    print("Retrieved datasets: ")
    print("type: ", type(all_data), " len: ",len(all_data))
    print("Example: ")
    print(all_data[:1])

    # -- Tokenise data
    inputs = tokenizer(
        all_data, max_length=512, truncation=True, padding=True, return_tensors=None
    )  # keys: input_ids, token_type_ids, attention_mask

    # Create dataset to define how to load and batch the tokenized data
    dataset = TextDataset(inputs)

    print(f" -- Created Dataset size: {len(dataset)}")

    return dataset




def load_dataset(dataset, batch_size):
    # Hugging Face’s masking collator to automatically mask 15% of the tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,  # ⚡ use multiple subprocesses to load data
        pin_memory=True,  # ⚡ enables faster transfer to GPU
        prefetch_factor=2,  # ⚡ overlap data loading with training
        persistent_workers=True  # ⚡ keep workers alive between epochs
    )
    return dataloader


# Add previous and trailing lines to each utterance to introduce context
# Build an adaptive context window around the masked line of at least min number of words
# text: array of strings
def add_context(text, max_size):
    #print("Adding context to utterances..")
    text_w_context = []
    for index in range(len(text)):
        line_w_context = text[index].split()
        length = len(line_w_context)
        buffer = 1

        while length < max_size:
            prev_line = text[index - buffer].split() + ['[SEP]'] if index - buffer >= 0 else []
            next_line = ['[SEP]'] +  text[index + buffer].split() if index + buffer < len(text) else []
            buffer += 1

            # Add context around the line
            line_w_context = prev_line + line_w_context + next_line
            length = len(line_w_context)

            if prev_line == [] and next_line == []:
                # Entire text has been added so break even if length has not been met
                break

        # Join the combined lines into a single string and add to array
        final_line = ' '.join(line_w_context)
        text_w_context.append(final_line)

        #print('\n')
        #for l in text_w_context:
        #    print(l)
    return text_w_context


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
