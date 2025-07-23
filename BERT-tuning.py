# code adapted from the provided base file in https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
# model fine-tuned following https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

# Input sentences into the pre-trained BERT model, each with a masked word
# Generate the top 5 predictions to replace this word
# Output these and their confidence scores
# Complete sentences with the highest scoring word

from transformers import BertTokenizer, BertForMaskedLM, pipeline
import torch

import torch
from tqdm.auto import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed
import pandas as pd
import warnings

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



def prep_training_data():
    # @ adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/
    # -- preprocess data by transforming into a list of sentences
    # Read test data -- transcripts of control speech
    file_path = 'capilouto01a.umbrella.gem2.flo.cex'
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

    # -- Create a mask over the data
    random_tensor = torch.rand(inputs['input_ids'].shape)
    print(inputs['input_ids'].shape, random_tensor.shape)

    # Mask 15% of the data -- avoid masking special tokens:
    # sentence boundaries [CLS] (token ID 101), [SEP] (token ID 102), and padding tokens (token ID 0)
    masked_tensor = (random_tensor < 0.15) * (inputs['input_ids'] != 101) * (inputs['input_ids'] != 102) * (
                inputs["input_ids"] != 0)

    # Extract non-zero values and replace the coresponding tokens with the MASK value
    nonzero_indices = [torch.nonzero(row).flatten().tolist() for row in masked_tensor]
    tokenizer.convert_tokens_to_ids("[MASK]")

    # Apply the mask
    for i in range(len(inputs['input_ids'])):
        inputs['input_ids'][i, nonzero_indices[i]] = 103

    return train_data

'''
    dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16, # Each batch contains 16 sequences
    shuffle=True # Shuffle the data to improve training
)
'''

def calculate_accuracy(data, model, tokenizer):
    # taken from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

    model.eval() # Puts the model in evaluation mode
    correct = 0
    total = 0

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)  # Move the model to the device ("cpu" or "cuda")


    for sentence in data:
        print('\n', sentence)
        # Replace a random token with [MASK] and store the original token
        tokens = tokenizer.encode(sentence, return_tensors='pt')[0]
        masked_index = torch.randint(0, len(tokens), (1,)).item()
        print("random masked index: ", masked_index)
        original_token = tokens[masked_index].item()
        tokens[masked_index] = tokenizer.mask_token_id

        inputs = {'input_ids': tokens.unsqueeze(0).to(device)}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_token_id = logits[0, masked_index].argmax().item()

        if predicted_token_id == original_token:
            correct += 1
        total += 1

        print("predicted: ", predicted_token_id, "| original: ", original_token)
        print('predicted: ', tokenizer.decode(predicted_token_id), '| original: ', tokenizer.decode(original_token) )

    accuracy = correct / total
    print(f"\n Accuracy: {accuracy * 100:.2f}% on {total} sequences")



def define_input():
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
    tokens = define_input()
    filtered = remove_fillers(tokens)
    filtered = remove_repetition(filtered)
    # Mask erroneous words
    masked_sentences = mask_tokens(filtered)
    return masked_sentences


def main():
    # Train mlm model on control data
    training_data = prep_training_data()

    mlm_model.eval()
    calculate_accuracy(training_data, mlm_model, tokenizer)

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
