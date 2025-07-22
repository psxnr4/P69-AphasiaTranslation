# code adpated from the provided base file in https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html

# Input three sentences into the pre-trained BERT model, each with a masked word
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

# Dictionary of filler words
filler_words = {
        'uh' : 7910
}
    #[('uh', 7910)]

# Define input sentence constructed from utterances
def define_input():
    sentence = ["and then he uh he he he must have had the umbrella on his back it looks like in the pack. and they went out on the rain and he was brained on."]

    original = sentence[0]
    tokenized = tokenizer.tokenize(sentence[0])
    tokenIDs = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence[0]))

    # Tokenise
    print('Original:', original)
    print('Tokenized:', tokenized)
    print('Token IDs:', tokenIDs)
    return tokenIDs


# Analyse tokens for filler words
def remove_fillers(tokens):
    print('1..Removing Filler Words..')
    # create set of the tokens corresponding to filler words for efficient searching
    filler_tokens = set(filler_words.values())
    filtered_tokens = [t for t in tokens if t not in filler_tokens]

    print('Filtered tokens:', filtered_tokens)
    print('Decoded ', tokenizer.decode(filtered_tokens) )
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

    print('Repeated words found at: ', repeated)
    print('Filtered rep. :', tokens)
    print('Decoded: ', tokenizer.decode(tokens))

    return tokens



# TODO: locate erroneous words + mask
def mask_tokens(tokens):
    masked_sentence = ['and then he must have had the umbrella on his back it looks like in the pack and [MASK] went out [MASK] the rain and he was [MASK] on.']
    #masked_sentence = ['it was raining and he must have had the umbrella in his bag so he got [MASK].']
    print('\nMasked sentence: ', masked_sentence)
    return masked_sentence






def mlm(masked_sentences):
    # Tokenise sentences and encode -- incld. padding as sentences are different lengths
    encoded_inputs = tokenizer(masked_sentences, return_tensors='pt', padding=True)
    input_ids = encoded_inputs["input_ids"] # shape: [batch_size, sequence_length]

    # Dynamically find [MASK] token positions in input sentence
    mask_token_id = tokenizer.mask_token_id  # This is 103 for BERT
    pos_masks = [torch.where(seq == mask_token_id)[0].tolist() for seq in input_ids] # 2d array -- list of indexes for mask tokens in each sentence

    # Pass into MLM model and get raw score outputs
    outputs = mlm_model(**encoded_inputs)

    # Get top-k word predictions for each masked token
    top_k = 5
    for i, mask_positions in enumerate(pos_masks):
        # Display sentence and tokens
        print('\n---- Sentence ', i + 1)
        print('Original:', masked_sentences[i])
        print('Tokenized:', tokenizer.tokenize(masked_sentences[i]))
        print('Token IDs:', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(masked_sentences[i])))

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

            # Get top score and fill in the mask
            #top_token = tokenizer.decode([top_k_ids[0].item()]).strip()
            #unmasked_sentence = masked_sentences[i].replace('[MASK]', top_token)

        unmasked_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        print(f"\nUnmasked sentence with top prediction:\n{unmasked_sentence}\n")



def main():
    # clean utterance of syntax errors
    tokens = define_input()
    filtered = remove_fillers(tokens)
    filtered = remove_repetition(filtered)

    # mask erroneous words
    masked_sentences = mask_tokens(filtered)

    # predict suitable replacements
    mlm(masked_sentences)

if __name__ == '__main__':
    main()
