# code adapted from the provided base file in https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
# model fine-tuned following https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

# A random token in each control-passage is masked and the predicted word is compared to the original to calculate its accuracy
# - this is using only the context of the surrounding passage to predict the missing word
# AdamW optimiser is then placed on the model + we run training on the control data in batches
# The testing stage is then repeated on the same data to predict new random tokens.
# - this is able to use the context of all control data
# Displays only the predictions that do not match the original token - some of these may be accepted if they give the same meaning

# -- Accuracy calculated across tests using random tokens pre- and post- training using three epochs
# Before training: Accuracy is 59.72% on 72 sequences
# 62.50% on 72 sequences when restricting masks to avoid special tokens or punctuation

# - learning rate 1e-5
# Run 1. 66.67% on 72 sequences

# - learning rate 3e-5
# 75.00% on 72 sequences
# 69.44% on 72 sequences

# - learning rate 5e-5
#  80.56% on 72 sequences
#  72.22% on 72 sequences




from transformers import BertTokenizer, BertForMaskedLM, pipeline
import torch
from tqdm.auto import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed, DataCollatorForLanguageModeling
import pandas as pd
import warnings
import string
import random


set_seed(42)
warnings.filterwarnings("ignore")

# Initialize the tokenizer from the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Get BERT masked language model from Hugging Face
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model.eval()

# Dictionary of filler words to remove from utterance
filler_words = {
        'uh' : 7910
}

# Define training dataset
# @ adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}



def prep_training_data():
    # @ adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/
    # -- preprocess data by transforming into a list of sentences
    # Read test data -- transcripts of control speech
    file_path = 'combined_output.cex'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip().rstrip('.') for line in f] # remove trailing periods
    content = ' '.join(lines) # remove line breaks

    # Convert to list, splitting at each transcript break delimiter
    train_data = content.split('|')
    print(train_data)

    # -- Tokenise data
    inputs = tokenizer(
        train_data, max_length=512, truncation=True, padding=True, return_tensors='pt'
    ) # keys: input_ids, token_type_ids, attention_mask
    # Add key for prediction labels
    inputs['labels'] = inputs['input_ids'].detach().clone()

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
    return train_data, inputs, dataloader


def calculate_accuracy(data, model, tokenizer, device):
    # adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

    model.eval() # Puts the model in evaluation mode
    correct = 0
    total = 0

    #count = 0
    #while count < 10:
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
            print('\n', tokenizer.decode(tokens))
            print("predicted: ", predicted_token_id, "| original: ", original_token)
            print('predicted: ', tokenizer.decode(predicted_token_id), '| original: ', tokenizer.decode(original_token))

        total += 1
    accuracy = correct / total
    print(f"\n Accuracy: {accuracy * 100:.2f}% on {total} sequences")


def training(model, dataloader, device):
    # adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/
    epochs = 3  # The model will train for 3 full passes over the dataset.
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()  # Puts the model in training mode

    #epoch_losses = []

    for epoch in range(epochs):
        loop = tqdm(dataloader)  # We use this to display a progress bar
        #batch_losses = []

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
'''
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        #epoch_losses.append(avg_loss)
        print(f"\nEpoch {epoch} average loss: {avg_loss:.4f}")
'''

def sample_input():
    # Define input sentence constructed from utterances
    sentence = ["and then he uh he he he must have had the umbrella on his back it looks like in the pack. and they went out on the rain and he was brained on."]
    original = sentence[0]
    tokenized = tokenizer.tokenize(sentence[0])
    tokenIDs = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence[0]))

   # print('Original:', original)
    #print('Tokenized:', tokenized)
    #print('Token IDs:', tokenIDs)
    return tokenIDs


# Analyse tokens for filler words
def remove_fillers(tokens):
    print('1..Removing Filler Words..')
    # Create set of the tokens corresponding to filler words for efficient searching
    filler_tokens = set(filler_words.values())
    filtered_tokens = [t for t in tokens if t not in filler_tokens]

    #print('Filtered tokens:', filtered_tokens)
    #print('Decoded ', tokenizer.decode(filtered_tokens) )
    return filtered_tokens

# Analyse tokens for repeated words
def remove_repetition(tokens):
    print('2..Removing Repeated Words..')
    repeated = []
    i = 0
    while i < len(tokens)-1:
        # If token matches the one after it, then remove
        if tokens[i] == tokens[i + 1]:
            repeated.append((i, tokens[i]))
            del tokens[i]
        else:
            i += 1

    #print('Repeated words found at: ', repeated)
    #print('Filtered rep. :', tokens)
    #print('Decoded: ', tokenizer.decode(tokens))

    return tokens



# TODO: locate erroneous words + mask
def mask_tokens(tokens):
    masked_sentence = ['and then he must have had the umbrella on his back it looks like in the pack and [MASK] went out [MASK] the rain and he was [MASK] on.']
    #masked_sentence = ['it was raining and he must have had the umbrella in his bag so he got [MASK].']
    print('\nMasked sentence: ', masked_sentence)
    return masked_sentence


def tokenise_input(masked_sentences):
    # @ adapted from https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
    print('..Tokenise Input..')
    # Tokenise sentences and encode -- incld. padding as sentences are different lengths
    encoded_inputs = tokenizer(masked_sentences, return_tensors='pt', padding=True)
    input_ids = encoded_inputs["input_ids"]  # shape: [batch_size, sequence_length]

    # Dynamically find [MASK] token positions in input sentence
    mask_token_id = tokenizer.mask_token_id  # This is 103 for BERT
    pos_masks = [torch.where(seq == mask_token_id)[0].tolist() for seq in
                 input_ids]  # 2d array -- list of indexes for mask tokens in each sentence

    return encoded_inputs, input_ids, pos_masks



def output_results(masked_sentences, encoded_inputs, input_ids, pos_masks ):
    # @ adapted from https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
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

        for pos in mask_positions:
            logits = outputs.logits[i, pos]  #  contains raw prediction scores
            probs = torch.nn.functional.softmax(logits, dim=-1) # softmax raw scores into probabilities
            top_k_probs, top_k_ids = torch.topk(probs, top_k) # get top_k probs from tensor

            # Display top scores
            print("\nTop predictions to replace masked token:")
            for prob, token_id in zip(top_k_probs, top_k_ids):
                #token_str = tokenizer.decode([token_id.item()]).strip()
                tokens = tokenizer.convert_ids_to_tokens([token_id.item()])
                token_str = tokenizer.convert_tokens_to_string(tokens).strip()
                print(f"Token: {token_str}, Score: {prob.item():.4f}")

            # Replace mask token with the top prediction
            input_ids[i, pos] = top_k_ids[0]


        # Decode repaired sentence from tokens
        unmasked_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        print(f"\nUnmasked sentence with top prediction:\n{unmasked_sentence}\n")




def prepare_input_data():
    # Clean utterance of syntax errors
    tokens = sample_input()
    filtered = remove_fillers(tokens)
    filtered = remove_repetition(filtered)
    # Mask erroneous words
    masked_sentences = mask_tokens(filtered)
    return masked_sentences


def main():
    # Train mlm model on control data
    training_data, mask_training_data, dataloader = prep_training_data()

    mlm_model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mlm_model.to(device)  # Move the model to the device ("cpu" or "cuda")

    # run model on random tokens in the control data + calculate produced accuracy
    #calculate_accuracy(training_data, mlm_model, tokenizer, device)

    # use the control data to train the model
    training(mlm_model, dataloader, device)

    # test again on control data -- using a random token in each story
    calculate_accuracy(training_data, mlm_model, tokenizer, device)

    '''
    # Prepare input data
    masked_sentences = prepare_input_data()
    # Tokenise sentences
    encoded_inputs, input_ids, pos_masks = tokenise_input(masked_sentences)
    # Pass into mlm model and output predictions for masked words
    output_results(masked_sentences, encoded_inputs, input_ids, pos_masks)
    '''



if __name__ == '__main__':
    main()
