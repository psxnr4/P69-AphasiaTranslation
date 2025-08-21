# code adapted from the provided base file in https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
# model fine-tuned following https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

# A random token in each control-passage is masked and the predicted word is compared to the original to calculate its accuracy
# - this is using only the context of the surrounding passage to predict the missing word
# AdamW optimiser is then placed on the model + we run training on the control data in batches
# The testing stage is then repeated on the same data to predict new random tokens.
# - this is able to use the context of all control data

# If a sentence has multiple mask tokens they will be predicted in sequence, this may propagate some errors

import torch
from tqdm.auto import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed  #BERT mlm
from transformers import BartForConditionalGeneration, BartTokenizer      #BART paraphrase

import warnings
import string
import nltk
from nltk.corpus import wordnet # Detect synonyms https://www.geeksforgeeks.org/python/get-synonymsantonyms-nltk-wordnet-python/

# Python modules
import TrainingData # TextDataset #LoadDataset #get_training_data -- returns training datasets
import TestingData # mask_from_directory -- masks training data stored in a given directory and returns dataset # Tokenise input

# ---- SET UP
minimum_context_length = False         # Word length of input utterance -- adds context from either side to reach length restriction
use_control_training_data = False    # Train the model on situational control data -- gives language context
use_aphasia_training_data = False    # Train the model on aphasia data  -- gives structure context
learning_rate = 5e-5                # Learning rate used during training
epochs = 3                          # Number of passes over full dataset during training

# ------ MODELS
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
    #epochs = 3  # Number of passes over full dataset
    optimizer = AdamW(model.parameters(), lr=learning_rate)
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





def run_bert(loader):
    print("\n -------------------------------------\n Running BERT \n \n -------------------------------------")

    # Keep track of module performance on each batch of data
    correct_count = 0
    total_count = 0
    accuracy = 0
    # Counter
    n = 0
    for batch in loader:
        print(f" \n***** utterance batch {n}  ***** \n -------------------------------------")
        n +=1
        # Get batch information
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_labels = batch['labels']

        # Find positions of masks in each sentence
        mask_positions = (input_ids == tokenizer.mask_token_id)  # bool tensor (batch_size, seq_length)

        # If no mask tokens have been found there's no need to run the model
        if mask_positions == [[]]:
            print("-- No mask tokens found --")
            unmasked_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(input_ids[0])
            print(unmasked_sentence)
            continue

        # Forward pass BERT on batch
        with torch.no_grad():
            outputs = mlm_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch_size, seq_length, vocab_size)

        # For each sequence item in the batch
        for i in range(input_ids.size(0)):
            # Get position of mask tokens in this sequence
            mask_indices = mask_positions[i].nonzero(as_tuple=True)[0]
            if len(mask_indices) == 0:
                continue  # no mask in this sentence

            # Get this sequence's attached information from the batch
            seq_labels = batch_labels[i] if batch_labels is not None else None  # suggested repair words for evaluation
            seq_token_ids = input_ids[i].clone()                                    # copy of seq tokens to modify
            seq_logits = logits[i]                                                  # output from the mlm

            #print(seq_labels)
            #print(seq_input_ids)
            #print(seq_logits)

            # For each mask token in the sequence
            for index in mask_indices:
                # Display masked utterance
                masked_utt = tokenizer.decode(seq_token_ids, skip_special_tokens=False)
                print(masked_utt)
                display_predictions(seq_logits[index], 2)

                # Get corresponding prediction from the batch output
                predicted_token_id = seq_logits[index].argmax().item()
                # Insert prediction into the list of input tokens
                seq_token_ids[index] = predicted_token_id

                # Evaluate prediction if labels are available
                if seq_labels is not None:
                    # Get corresponding token in the label tensor
                    target_token_id = seq_labels[index].item()
                    if predicted_token_id and target_token_id:
                        if evaluate_prediction(predicted_token_id, target_token_id, seq_token_ids ):
                            correct_count += 1
                    total_count += 1

            # Decode predicted sentence
            completed_sentence = tokenizer.decode(seq_token_ids, skip_special_tokens=True)
            print(f"Completed sentence: {completed_sentence}")

    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"\n ------------------------------ \n ------------------------------")
        print(f"Accuracy: {accuracy * 100:.2f}% on {total_count} sequences")
        print(f"------------------------------ \n ------------------------------")


    return accuracy



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




def evaluate_prediction(predicted_token_id, target_token_id, sentence_tokens):
    # get token as a string
    predicted_word = tokenizer.decode(predicted_token_id)
    target_word = tokenizer.decode(target_token_id)

    # Display top prediction
    #print("Suggested target word is : ", target_word, " : ", target_token_id)
    print("predicted: ", predicted_word, "| suggested: ", target_word)
    #print("predicted: ", predicted_token_id, "| original: ", target_token_id)

    # Accept an exact match
    if predicted_token_id == target_token_id:
        print("Match! ")
        return True

    # If not an exact match, check if prediction is a synonym of target
    score = check_synonyms(predicted_word, target_word, sentence_tokens)
    print("Synonym score: ", score)
    if score > 0.75:
        print("Synonym match!")
        return True
    else:
        print("!!! No match ")
        return False


def check_synonyms(predicted, target, sentence_tokens):
    # adapted from https://www.geeksforgeeks.org/python/get-synonymsantonyms-nltk-wordnet-python/

    # Get the sentence we have so far from the tokens
    sentence = tokenizer.decode(sentence_tokens, skip_special_tokens=True)
    # Get the predicted word's position in the sentence  - remove punctuation as this hides the word
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    pred_sent = sentence.split()

    # If prediction is not in the sentence then exit with error
    if predicted not in pred_sent:
        print("*ERR:* cant find predicted word")
        return False
    index = pred_sent.index(predicted)

    # Create a copy of the sentence to contain the suggested target word
    targ_sent = sentence.split()
    targ_sent[index] = target
    print("Target sentence: ", targ_sent)

    # Get the words within the context of the surrounding sentence
    # Tag all words with its 'part-of-speech' and extract the needed index
    pred_tagged = nltk.pos_tag(pred_sent)[index]
    targ_tagged = nltk.pos_tag(targ_sent)[index]

    # Get POS type from nlkt tag -- https://www.nltk.org/book/ch05.html
    #print(nltk.pos_tag([target]))
    if pred_tagged[1] in ["PRP", "PRP$", "WP", "WP$"] or targ_tagged[1] in ["PRP", "PRP$", "WP", "WP$"]:
        print("Word is a pronoun - cannot check synonyms")
        return False

    # Convert to a synsets object take None if the word is not found in wordnet - e.g. wordnet does not include pronouns
    print("-- Checking synonyms between ", pred_tagged, " and ", targ_tagged)
    syns_pred = wordnet.synsets(predicted)
    syns_targ = wordnet.synsets(target)

    # If both words have been found then calculate similarity
    if syns_pred and syns_targ:
        # Compare all combinations of definitions of the word found
        max_score = 0
        for s1 in syns_pred:
            for s2 in syns_targ:
                sim = s1.wup_similarity(s2)
                if sim and sim > max_score:
                    print(s1, ",", s2, sim)
                    max_score = sim
        print("-- Max Synonyms Score: ", max_score)
        return round(max_score,4)
    else:
        print("-- Word has not been found in wordnet")
        return False

# todo: could also test to see if target is in syn set of predicted word
# todo: check to see if target word is in the top 10 predicted words - then accuracy is a little off but not too low



def print_highlighted_result(pred_tokens, org_sent):
    #print("Predicted tokens: ", pred_tokens)
    # Repair sentence with predictions - Replace each mask token one by one
    for predicted in pred_tokens:
        org_sent = org_sent.replace(tokenizer.mask_token, f"\\*{predicted}\\*", 1)
    return org_sent


def main():

    # Setup models
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mlm_model.to(device)  # Move the model to the device ("cpu" or "cuda")
    bart_model.to(device)

    if use_control_training_data or use_aphasia_training_data:
        # Prepare training data and run training on the BERT model
        dataloader = TrainingData.get_training_data(use_control_training_data, use_aphasia_training_data, minimum_context_length)
        training(mlm_model, dataloader, device)

    # Prepare testing data
    # Get a dataset containing the masked version of all transcripts within the training data directory
    test_dataset = TestingData.mask_from_directory(minimum_context_length)
    
    # Batch this dataset to step through each data instance
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )
    print("Dataset Loaded")

    # Run BERT on testing data
    run_bert(loader)

    # Run model on random tokens in the dataset + calculate overall produced accuracy
    #accuracy_random_tokens(training_data, mlm_model, tokenizer, device)

    # Paraphrase output
    #paraphrase_text(result)




if __name__ == '__main__':
    main()
