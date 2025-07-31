# code adapted from the provided base file in https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
# model fine-tuned following https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

# A random token in each control-passage is masked and the predicted word is compared to the original to calculate its accuracy
# - this is using only the context of the surrounding passage to predict the missing word
# AdamW optimiser is then placed on the model + we run training on the control data in batches
# The testing stage is then repeated on the same data to predict new random tokens.
# - this is able to use the context of all control data
# Displays only the predictions that do not match the original token - some of these may be accepted if they give the same meaning

# -- Training on the combined dataset on control and repaired aphasia transcripts
# -- On W01 this gives accuracy of Accuracy: 71.43% on 7 sequences using learning rate 5e-05
# -- Accuracy is 42.86% on 7 sequences before training
'''Epoch: 0: 100%|██████████| 11/11 [08:04<00:00, 44.09s/it, loss=1.79]
Epoch: 1: 100%|██████████| 11/11 [08:58<00:00, 48.97s/it, loss=1.91]
Epoch: 2: 100%|██████████| 11/11 [13:12<00:00, 72.06s/it, loss=1.77]'''

from transformers import BertTokenizer, BertForMaskedLM, pipeline
import torch
from tqdm.auto import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed, DataCollatorForLanguageModeling
import pandas as pd
import warnings
import string
import random
from torch.utils.data import ConcatDataset

input_file_path = 'Masked-transcript.txt'

set_seed(42)
warnings.filterwarnings("ignore")

# Initialize the tokenizer from the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Get BERT masked language model from Hugging Face
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model.eval()


# Define training dataset
# @ adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


#Tokenise training data and create dataloader masking 15% of tokens
def prep_training_data(file_path):
    # @ adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/
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

    ''''# -- Create a mask over the data
    random_tensor = torch.rand(inputs['input_ids'].shape)
    print(inputs['input_ids'].shape, random_tensor.shape)

    # Mask 15% of the data -- avoid masking special tokens:
    # sentence boundaries [CLS] (token ID 101), [SEP] (token ID 102), and padding tokens (token ID 0)
    masked_tensor = (random_tensor < 0.15) * (inputs['input_ids'] != 101) * (inputs['input_ids'] != 102) * (
                inputs["input_ids"] != 0)

    # Extract non-zero values and replace the corresponding tokens with the MASK value
    nonzero_indices = [torch.nonzero(row).flatten().tolist() for row in masked_tensor]
    tokenizer.convert_tokens_to_ids("[MASK]")

    # Apply the mask
    for i in range(len(inputs['input_ids'])):
        inputs['input_ids'][i, nonzero_indices[i]] = 103
'''
    # Create dataset to define how to load and batch the tokenized data
    dataset = TextDataset(inputs)

    return dataset

# Concat datasets to prevent resetting the AdamW optimiser between training sets
def combine_training_data(dataset1, dataset2):
    combined_dataset = ConcatDataset([dataset1, dataset2])

    # Hugging Face’s masking collator to automatically mask 15% of the tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=data_collator
    )
    return dataloader

'''
    # Sanity check: Get one batch from the DataLoader [chatgpt gen]
    batch = next(iter(dataloader))
    # Print keys and tensor shapes
    print("Batch keys:", batch.keys())
    for key, value in batch.items():
        print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")
    masked_count = (batch['input_ids'] == tokenizer.mask_token_id).sum().item()
    print(f"Number of [MASK] tokens in this batch: {masked_count}")
'''



def training(model, dataloader, device):
    # adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/
    epochs = 3  # Number of passes over full dataset
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()  # Puts the model in training mode

    for epoch in range(epochs):
        loop = tqdm(dataloader)  # We use this to display a progress bar

        for batch in loop:
            optimizer.zero_grad()  # Reset gradients before each batch
            # Move input_ids, labels, attention_mask to be on the same device as the model
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # Forward pass
            loss = outputs.loss

            loss.backward()  # Compute gradients, backward pass
            optimizer.step()  # Update model parameters

            loop.set_description("Epoch: {}".format(epoch))  # Display epoch number
            loop.set_postfix(loss=loss.item())  # Show loss in the progress bar





# Mask random words within the data + predict them -- calculate success across all predictions
def calculate_accuracy_on_dataset(data, model, tokenizer, device):
    # adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

    model.eval() # Puts the model in evaluation mode
    correct = 0
    total = 0

    for sentence in data:
        #print('\n', sentence)
        # Replace a random token with [MASK] and store the original token
        tokens = tokenizer.encode(sentence, return_tensors='pt')[0]
        #masked_index = torch.randint(0, len(tokens), (1,)).item()

        # Get list of tokens that can be masked -- avoid special tokens or punctuation
        special_tokens = tokenizer.all_special_ids
        punct_tokens = tokenizer.encode(string.punctuation, add_special_tokens=False)

        candidate_tokens = [
            i for i, token_id in enumerate(tokens)
            if token_id not in special_tokens and token_id not in punct_tokens
        ]
        masked_index = random.choice(candidate_tokens)


        original_token = tokens[masked_index].item()
        tokens[masked_index] = tokenizer.mask_token_id

        inputs = {'input_ids': tokens.unsqueeze(0).to(device)}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_token_id = logits[0, masked_index].argmax().item()

        if predicted_token_id == original_token:
            correct += 1
        else:
            # display incorrect prediction
            #print('\n', tokenizer.decode(tokens))
            print("No match: ")
            print("predicted: ", predicted_token_id, "| original: ", original_token)
            print('predicted: ', tokenizer.decode(predicted_token_id), '| original: ', tokenizer.decode(original_token))

        total += 1
    accuracy = correct / total
    print(f"\n Accuracy: {accuracy * 100:.2f}% on {total} sequences")




def get_masked_input():
    # Define input sentence constructed from utterances
    with open(input_file_path, 'r', encoding='utf-8') as f:
        print('..Reading File..')
        # First line of file contains the target words that have been suggested for repairs
        target_words = f.readline().strip()
        target_words = target_words.split(',')

        # Read in the rest of the transcript
        masked_content = f.read()

        #print(target_words)
        #print(masked_content)

    return target_words, masked_content


def tokenise_input(masked_sentences):
    # @ adapted from https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
    print('..Tokenising Input..')
    # Tokenise sentences and encode -- incld. padding as sentences are different lengths
    encoded_inputs = tokenizer(masked_sentences, return_tensors='pt', padding=True)
    input_ids = encoded_inputs["input_ids"]  # shape: [batch_size, sequence_length]

    # Dynamically find [MASK] token positions in input sentence
    print('..Finding Masks..')
    mask_token_id = tokenizer.mask_token_id  # This is 103 for BERT
    pos_masks = [torch.where(seq == mask_token_id)[0].tolist() for seq in
                 input_ids]  # 2d array -- list of indexes for mask tokens in each sentence

    return encoded_inputs, input_ids, pos_masks



def output_results(target_words, masked_sentences, encoded_inputs, input_ids, pos_masks ):
    # @ adapted from https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
    print('..Passing into model..')
    # Pass into MLM model and get raw score outputs
    outputs = mlm_model(**encoded_inputs)
    # Get top-k word predictions for each masked token
    top_k = 5

    for i, mask_positions in enumerate(pos_masks):
        # Display sentence and tokens
        #print('\n---- Sentence ', i + 1)
        #print('Original:', masked_sentences[i])
        #print('Tokenized:', tokenizer.tokenize(masked_sentences[i]))
        #print('Token IDs:', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(masked_sentences[i])))

        # Keep a list of predicted tokens for each
        predicted_tokens = []
        mask_num = -1
        correct = 0
        total = 0
        print(target_words)

        for pos in mask_positions:
            mask_num += 1
            logits = outputs.logits[i, pos]  #  contains raw prediction scores
            probs = torch.nn.functional.softmax(logits, dim=-1) # softmax raw scores into probabilities
            top_k_probs, top_k_ids = torch.topk(probs, top_k) # get top_k probs from tensor

            print("\nSuggested target word is : ", target_words[mask_num])
            # Display top scores
            print("Top predictions to replace masked token:")
            for prob, token_id in zip(top_k_probs, top_k_ids):
                token_str = tokenizer.decode([token_id.item()]).strip()
                #tokens = tokenizer.convert_ids_to_tokens([token_id.item()])
                #token_str = tokenizer.convert_tokens_to_string(tokens).strip()
                print(f"Token: {token_str}, Score: {prob.item():.4f}")

            # Replace mask token with the top prediction
            #input_ids[i, pos] = top_k_ids[0]
            # Get top prediction and store it with a highlight
            best_prediction = tokenizer.decode([top_k_ids[0].item()]).strip()
            predicted_tokens.append(f"*{best_prediction}*")

            #TODO: Compare target word to predicted word + keep running track of accuracy of run
            #TODO: Compare accuracy pre and post training on random control tokens

            if best_prediction == target_words[mask_num]:
                correct += 1
                print("Match: ")
            else:
                print("!!! No match: ")

            print("predicted: ", best_prediction, "| original: ", target_words[mask_num])
            total += 1
        accuracy = correct / total
        print(f"\n Accuracy: {accuracy * 100:.2f}% on {total} sequences")

        # Repair sentence with predictions - Replace each mask token one by one
        unmasked_sentence = masked_sentences
        for predicted in predicted_tokens:
            unmasked_sentence = unmasked_sentence.replace(tokenizer.mask_token, predicted, 1)

        # Decode repaired sentence from tokens
        #unmasked_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        print(f"\nUnmasked sentence with top predictions:\n{unmasked_sentence}\n")







def main():
    # Create dataset from the control transcripts
    control_file_path = 'control_training_combined_output.cex'
    control_dataset = prep_training_data(control_file_path)

    # Create dataset from the repaired aphasia transcripts
    aphasia_file_path = 'aphasia_training_combined_output.cex'
    aphasia_dataset = prep_training_data(aphasia_file_path)

    dataloader = combine_training_data(control_dataset, aphasia_dataset)

    # Setup model
    mlm_model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mlm_model.to(device)  # Move the model to the device ("cpu" or "cuda")

    # run model on random tokens in the control data + calculate produced accuracy
    #calculate_accuracy(training_data, mlm_model, tokenizer, device)

    # Train the model on the combined dataset
    training(mlm_model, dataloader, device)

    # test again on control data -- using a random token in each story
    #calculate_accuracy(training_data, mlm_model, tokenizer, device)

    # Prepare input data
    target_words, masked_sentences = get_masked_input()
    # Tokenise sentences
    encoded_inputs, input_ids, pos_masks = tokenise_input(masked_sentences)
    # Pass into mlm model and output predictions for masked words
    output_results(target_words, masked_sentences, encoded_inputs, input_ids, pos_masks)




if __name__ == '__main__':
    main()
