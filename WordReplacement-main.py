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



import torch
from tqdm.auto import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed  #BERT mlm
from transformers import BartForConditionalGeneration, BartTokenizer      #BART paraphrase

import warnings
import string
import nltk
from nltk.corpus import wordnet # Detect synonyms https://www.geeksforgeeks.org/python/get-synonymsantonyms-nltk-wordnet-python/

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





def run_bert(loader):

    # Keep track of module performance on each batch of data
    correct_count = 0
    total_count = 0
    accuracy = 0
    valid = False

    n = 0
    for batch in loader:
        print(f" \n***** utterance batch {n}  ***** \n -------------------------------------")
        n +=1
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
            print(encoded_inputs["input_ids"][0])
            print(unmasked_sentence)
            continue

        # Pass batch into mlm model
        outputs = mlm_model(**encoded_inputs)
        logits = outputs.logits
        sequence_length = outputs.logits.shape[1]  # -- (batch_size, sequence_length, vocab_size)

        # For each sentence i in the batch
        for i, mask_positions in enumerate(pos_masks):
            print("mask_positions", mask_positions)
            # logits for this sequence
            seq_logits = logits[i]  # shape: (seq_length, vocab_size)

            # Get sequence of token IDs from the batch
            input_ids = encoded_inputs['input_ids'][i]

            # for each mask token in the sequence - get corresponding prediction and evaluate
            count = 0
            for pos in mask_positions:
                print(f"\n ***** mask num {count}  ***** ")
                count +=1
                predicted_token_id, target_token_id, unmasked_sentence = sequence_results(seq_logits, target_ids_list, pos, input_ids)
                # Evaluate the accuracy of the top prediction - if prediction has been made
                if predicted_token_id and target_token_id:
                    valid = evaluate_prediction(predicted_token_id, target_token_id, unmasked_sentence)
                else:
                    continue

                if valid:
                    correct_count += 1
                total_count += 1

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




def evaluate_prediction(predicted_token_id, target_token_id, unmasked_sentence):
    # get token as a string
    predicted_word = tokenizer.decode(predicted_token_id)
    target_word = tokenizer.decode(target_token_id)

    # Display top prediction
    #print("Suggested target word is : ", target_word, " : ", target_token_id)
    print("predicted: ", predicted_word, "| original: ", target_word)
    #print("predicted: ", predicted_token_id, "| original: ", target_token_id)

    # Accept an exact match
    if predicted_token_id == target_token_id:
        print("Match! ")
        return True

    # If not an exact match, check if prediction is a synonym of target
    score = check_synonyms(predicted_word, target_word, unmasked_sentence)
    if score > 0.75:
        print("Synonym of score: ", score)
        return True
    else:
        print("!!! No match ")
        return False


def check_synonyms(predicted, target, sentence):
    # adapted from https://www.geeksforgeeks.org/python/get-synonymsantonyms-nltk-wordnet-python/

    # get position in sentence  - remove punctuation as this hides the word
    cleaned_text = sentence.translate(str.maketrans('', '', string.punctuation))
    words = cleaned_text.split()
    #print("sentence: ", words)
    # If prediction is not in the sentence then exit error
    if predicted not in words:
        return False
    pos = words.index(predicted)

    # Use the completed sentence to extract 'part-of-speech' tag to evaluate the selected words within the given context
    tagged = nltk.pos_tag(words)
    tagged_predicted = tagged[pos]

    # Get POS type from nlkt tag -- https://www.nltk.org/book/ch05.html
    #print(nltk.pos_tag([target]))
    if tagged_predicted[1] in ["PRP", "PRP$", "WP", "WP$"]:
        print("Word is a pronoun - cannot check synonyms")
        return False

    # Convert to a synsets object by getting the first item from the list of word definitions
    # take None if the word is not found in wordnet - e.g. wordnet does not include pronouns
    syns_word1 = wordnet.synsets(predicted)[0] if wordnet.synsets(predicted) else None
    syns_word2 = wordnet.synsets(target)[0] if wordnet.synsets(target) else None

    print("-- Checking synonyms between ", syns_word1, " and ", syns_word2)

    # If both words have been found then calculate similarity
    if syns_word1 and syns_word2:
        syns_score = syns_word1.wup_similarity(syns_word2)
        print("-- Synonyms score: ", syns_score)
        return syns_score
    else:
        print("-- Word has not been found in wordnet")
        return False

# todo: could also test to see if target is in syn set of predicted word




def print_highlighted_result(pred_tokens, org_sent):
    #print("Predicted tokens: ", pred_tokens)
    # Repair sentence with predictions - Replace each mask token one by one
    for predicted in pred_tokens:
        org_sent = org_sent.replace(tokenizer.mask_token, f"\\*{predicted}\\*", 1)
    return org_sent


def sequence_results(seq_logits, target_words, pos, input_ids ):
    # @ adapted from https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
    print('..Getting results..')

    # Display sentence and tokens
    #print('\n---- Sentence ', i + 1)
    #print('Original:', masked_sentences[i])
    #print('Tokenized:', tokenizer.tokenize(masked_sentences[i]))
    #print('Token IDs:', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(masked_sentences[i])))

    masked_sentence = tokenizer.decode(input_ids,  skip_special_tokens=False)
    predicted_tokens = []
    mask_num = 0
    predicted_token_id = None
    target_token_id = None
    unmasked_sentence = masked_sentence

    print(masked_sentence)

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

    # Replace mask token with the top prediction
    #input_ids[i, pos] = top_k_ids[0]
    input_ids[pos] = predicted_token_id

    # Display new sentence with the predicted words highlighted
    highlighted_result = print_highlighted_result(predicted_tokens, masked_sentence)
    #print(highlighted_result)

    # Decode repaired sentence from tokens
    unmasked_sentence = tokenizer.decode(input_ids, skip_special_tokens=True)
    #print(f"\nUnmasked sentence with top predictions:\n{unmasked_sentence}\n")

    return predicted_token_id, target_token_id, unmasked_sentence




def main():

    # Setup models
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mlm_model.to(device)  # Move the model to the device ("cpu" or "cuda")
    bart_model.to(device)

    # Prepare training data and run training on the BERT model
    dataloader = TrainingData.get_training_data()
    #training(mlm_model, dataloader, device)

    # Prepare testing data
    # Get a dataset containing the masked version of all transcripts within the training data directory
    test_dataset = MaskTranscript.mask_from_directory()
    
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
