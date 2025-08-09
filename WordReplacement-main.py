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

'''Predicted  6  out of  22
Accuracy:  0.2727272727272727
All accuracies:  [1.0, 0.4, 0.0, 0.0, 0.3333333333333333, 0.2857142857142857]'''


import torch
from tqdm.auto import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed  #BERT mlm
from transformers import BartForConditionalGeneration, BartTokenizer      #BART paraphrase

import warnings

from typing_extensions import override

# Python modules
import TrainingData # TextDataset #LoadDataset #get_training_data -- returns training datasets
import MaskTranscript # mask_from_directory -- masks training data stored in a given directory and returns dataset
                        # Tokenise input

set_seed(42)
warnings.filterwarnings("ignore")

# Initialize the tokenizer from the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Get BERT masked language model from Hugging Face
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model.eval()

# Get BART paraphrase model from Hugging Face
bart_model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
bart_tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')



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





def run_bert():

    # Get a dataset containing the masked version of all transcripts within the training data directory
    test_dataset = MaskTranscript.mask_from_directory()
    # Batch this dataset to step through each data instance
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )
    print("dataset loaded")

    # Keep track of module performance on each batch of data
    accuracies = []
    results = []
    overall_correct_count = 0
    overall_total = 0

    for batch in loader:
        # Get batch information
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        encoded_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Get attached information
        target_ids_list = []
        for i, target in enumerate(batch['target_ids']):
            target_ids_list = target.tolist()

        # Get positions of each mask token
        pos_masks = [torch.where(seq ==  tokenizer.mask_token_id)[0].tolist() for seq in input_ids]  # 2d array -- list of indexes for mask tokens in each sentence

        # If no mask tokens have been found there's no need to run the model
        if pos_masks == [[]]:
            print("-- No mask tokens found --")
            unmasked_sentence = tokenizer.decode(encoded_inputs["input_ids"][0], skip_special_tokens=True)
            return unmasked_sentence

        # Pass batch into mlm model
        outputs = mlm_model(**encoded_inputs)
        logits = outputs.logits
        sequence_length = outputs.logits.shape[1]  # -- (batch_size, sequence_length, vocab_size)

        # For each sentence i in the batch
        for i, mask_positions in enumerate(pos_masks):

            # logits for this sequence
            seq_logits = logits[i]  # shape: (seq_length, vocab_size)

            # Get sequence of token IDs from the batch
            input_ids = encoded_inputs['input_ids'][i]

            sequence_results(seq_logits, target_ids_list, mask_positions, input_ids)

'''
overall_correct_count += correct_count
overall_total += total_count

if overall_total > 0:
    print("Predicted ", overall_correct_count, " out of ", overall_total)
    print("Accuracy: ", overall_correct_count/overall_total)

#if total > 0:
#    accuracy = correct / total
#    print(f"\n Accuracy: {accuracy * 100:.2f}% on {total} sequences")

# store outputs
results.append(result)
accuracies.append(accuracy)

print("All accuracies: ", accuracies)
'''

def display_predictions(logits, top_k ):
    # Softmax raw scores into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Get top-k word predictions for each masked token
    top_k_probs, top_k_ids = torch.topk(probs, top_k)
    # Display top scores
    print("\n Top predictions to replace masked token:")
    for prob, token_id in zip(top_k_probs, top_k_ids):
        token_str = tokenizer.decode([token_id.item()]).strip()
        # tokens = tokenizer.convert_ids_to_tokens([token_id.item()])
        # token_str = tokenizer.convert_tokens_to_string(tokens).strip()
        print(f"Token: {token_str}, Score: {prob.item():.4f}")




def evaluate(predicted_token_id, target_token_id):
    # get token as a string
    predicted_word = tokenizer.decode(predicted_token_id)
    target_word = tokenizer.decode(target_token_id)

    # Display top prediction
    print("Suggested target word is : ", target_word, " : ", target_token_id)


    if predicted_token_id == target_token_id:
        print("Match! ")
    else:
        print("!!! No match: ")

    print("predicted: ", predicted_word, "| original: ", target_word)
    print("predicted: ", predicted_token_id, "| original: ", target_token_id)


def print_highlighted_result(pred_tokens, org_sent):
    print("Predicted tokens: ", pred_tokens)
    # Repair sentence with predictions - Replace each mask token one by one
    for predicted in pred_tokens:
        org_sent = org_sent.replace(tokenizer.mask_token, f"\\*{predicted}\\*", 1)

    return org_sent


def sequence_results(seq_logits, target_words, mask_positions, input_ids ):
    # @ adapted from https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
    print('..Getting results..')

    # Display sentence and tokens
    #print('\n---- Sentence ', i + 1)
    #print('Original:', masked_sentences[i])
    #print('Tokenized:', tokenizer.tokenize(masked_sentences[i]))
    #print('Token IDs:', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(masked_sentences[i])))

    masked_sentence = tokenizer.decode(input_ids,  skip_special_tokens=False)
    print("Masked sentence: ", masked_sentence)

    predicted_tokens = []
    mask_num = 0
    print('Target words:', target_words)

    # for each mask token in the sequence - get corresponding prediction and evaluate
    for pos in mask_positions:

        # Get target word
        target_token_id = target_words[mask_num]
        mask_num += 1 # increment for next pass

        # Get raw prediction scores
        token_logits = seq_logits[pos]

        # Display top k predictions to the console
        #display_predictions(token_logits, 2)

        # Get top prediction
        predicted_token_id = token_logits.argmax().item()
        predicted_word = tokenizer.decode(predicted_token_id)
        predicted_tokens.append(f"{predicted_word}")

        # Evaluate the accuracy of the top prediction
        evaluate(predicted_token_id, target_token_id)

        # Replace mask token with the top prediction
        #input_ids[i, pos] = top_k_ids[0]
        input_ids[pos] = predicted_token_id

        # Display new sentence with the predicted words highlighted
        highlighted_result = print_highlighted_result(predicted_tokens, masked_sentence)
        print(highlighted_result)

        # Decode repaired sentence from tokens
        unmasked_sentence = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"\nUnmasked sentence with top predictions:\n{unmasked_sentence}\n")

    return []




def main():

    # Setup models
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mlm_model.to(device)  # Move the model to the device ("cpu" or "cuda")
    bart_model.to(device)

    # Prepare training data and run training on the BERT model
    dataloader = TrainingData.get_training_data()
    #training(mlm_model, dataloader, device)

    # Run BERT on testing data
    run_bert()

    # Run model on random tokens in the dataset + calculate overall produced accuracy
    #accuracy_random_tokens(training_data, mlm_model, tokenizer, device)

    # Paraphrase output
    #paraphrase_text(result)




if __name__ == '__main__':
    main()
