# code adapted from the provided base file in https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
# model fine-tuned following https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/


import torch
from tqdm.auto import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed, pipeline  # BERT mlm
from transformers import BartForConditionalGeneration, BartTokenizer      #BART paraphrase
from transformers import get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup

import os
import warnings
import string
import csv
import nltk
from nltk.corpus import wordnet # Detect synonyms https://www.geeksforgeeks.org/python/get-synonymsantonyms-nltk-wordnet-python/

# Python modules
import TrainingData # TextDataset #LoadDataset #get_training_data -- returns training datasets
import TestingData # mask_from_directory -- masks training data stored in a given directory and returns dataset # Tokenise input
import LogResults   # create_result_files

import matplotlib.pyplot as plt


# ---- SET UP
max_context_length = 1              # Update in mainloop -- Word length of input utterance -- adds context from either side to reach length restriction
use_control_training_data = True    # Train the model on situational control data -- gives language context
use_aphasia_training_data = True    # Train the model on aphasia data  -- gives structure context
learning_rate = 2e-5                # Learning rate used during training
epochs = 3                          # Number of passes over full dataset during training
batch_size = 16                    # Update in mainloop -- Batch size used on training data

# ------ VARS
epoch_losses = [0,0,0] # loss after each epoch
batch_eval = {'correct_count':0, 'total_count':0, 'exact_match':0, 'syn_match':0} # Keep track of module performance on each batch of data
set_up_vars = [max_context_length, use_control_training_data, use_aphasia_training_data, learning_rate, batch_size]


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


sentences = []


# Training the model on passed data
# adapted from https://smartcat.io/tech-blog/llm/fine-tuning-bert-with-masked-language-modelling/

def training(model, dataloader, device):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # set up
    total_steps = len(dataloader) * epochs
    num_warmup_steps = int(0.1 * total_steps)  # 10% warmup
    num_training_steps = total_steps
    lr_end = 1e-6
    power = 1.0  # 1.0 = linear decay

    ''' # decay to 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    '''
    # Use polynomial decay with power=1.0 (i.e. linear decay) down to final_lr
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,  # 10% warmup
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        power=power  # 1.0 = linear decay
    )
    model.to(device)
    model.train()  # Puts the model in training mode

    lr_history = []  # store learning rate at each step

    # log file
    f = open("learning rates.txt", "a")
    f.write(f"----- {learning_rate}, {num_warmup_steps}, {num_training_steps}, {lr_end}, {power}")
    f.write("step,learning_rate\n")

    global_step = 0
    for epoch in range(epochs):
        loop = tqdm(dataloader)

        for batch in loop:
            # Reset gradients before each batch
            optimizer.zero_grad(set_to_none=True)
            # Move input_ids, labels, attention_mask to be on the same device as the model
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            #print("---- % of tokens masked -----")
            #masked_count = (batch["labels"] != -100).sum().item()
            #total_count = batch["labels"].numel()
            #print(f"Masked tokens: {masked_count}/{total_count}")

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            #print("loss: ", outputs.loss.item())

            # Compute gradients, backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update model parameters
            optimizer.step()
            scheduler.step()

            # --- Log current learning rate ---
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            f.write(f"{global_step},{current_lr:.8e}\n")

            # Display progress bar with epoch number and loss
            loop.set_description("Epoch: {}".format(epoch))
            loop.set_postfix(loss=loss.item())

            epoch_losses[epoch] = loss.item()

            global_step += 1

    plt.plot(lr_history)
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Adaptive Learning Rate Over Training")
    plt.show()

    return model


def run_bert( loader, run_writer):
    print("\n -------------------------------------\n Running BERT \n \n -------------------------------------")

    # Keep track of module performance on each batch of data - rest values
    batch_eval.update({'correct_count':0, 'total_count':0, 'exact_match':0, 'syn_match':0})

    n = 0
    for batch in loader:
        print("--------------------------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------------------------")
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
                print("No mask token")
                unmasked_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                print(input_ids[0])
                print(unmasked_sentence)
                continue  # no mask in this sentence

            # Get this sequence's attached information from the batch
            seq_labels = batch_labels[i] if batch_labels is not None else None  # suggested repair words for evaluation
            seq_token_ids = input_ids[i].clone()                                    # copy of seq tokens to modify
            seq_logits = logits[i]                                                  # output from the mlm

            #print(seq_labels)
            #print(seq_input_ids)
            #print(seq_logits)

            predictions = []

            # For each mask token in the sequence
            for index in mask_indices:
                # Display masked utterance
                masked_utt = tokenizer.decode(seq_token_ids, skip_special_tokens=False)
                print("\n", masked_utt)
                #print(index)
                #print(seq_token_ids)

                display_predictions(seq_logits[index], 3)

                # Get corresponding prediction from the batch output
                predicted_token_id = seq_logits[index].argmax().item()
                # Insert prediction into the list of input tokens
                seq_token_ids[index] = predicted_token_id

                predictions.append(tokenizer.decode(predicted_token_id))

                # Evaluate prediction if labels are available
                if seq_labels is not None:
                    # Get corresponding token in the label tensor
                    target_token_id = seq_labels[index].item()
                    if predicted_token_id and target_token_id:
                        results = evaluate_prediction(predicted_token_id, target_token_id, seq_token_ids )
                        # Write prediction to log
                        print(results)
                        run_writer.writerow(results)
                    batch_eval['total_count'] += 1

            # Decode predicted sentence
            completed_sentence = tokenizer.decode(seq_token_ids, skip_special_tokens=True)
            print(f"\nCompleted sentence: {completed_sentence}")

            sentences.append( (masked_utt, completed_sentence, predictions))
            # Paraphrase output
            #paraphrase_text(completed_sentence)

    if batch_eval['total_count'] > 0:
        accuracy = batch_eval['correct_count'] / batch_eval['total_count']
        print(f"\n ------------------------------ \n ------------------------------")
        print(f"Accuracy: {accuracy * 100:.2f}% on {batch_eval['total_count'] } sequences")
        print(f"Exact Match: {(batch_eval['exact_match'] / batch_eval['correct_count'])*100:.2f}% of {batch_eval['correct_count']} matches")
        print(f"Synonym Match: {(batch_eval['syn_match'] / batch_eval['correct_count'])*100:.2f}% of {batch_eval['correct_count']} matches")
        print(f"------------------------------ \n ------------------------------")
        print("SetUp")
        print(f"max_context_length: {max_context_length}")
        print(f"use_control_training_data: {use_control_training_data}")
        print(f"use_aphasia_training_data: {use_aphasia_training_data}")
        print(f"learning_rate: {learning_rate}")
        print(f"epochs: {epochs}")
        print(f"batch_size: {batch_size}")
        print(f"------------------------------ \n ------------------------------")



def display_predictions(logits, top_k ):
    # Softmax raw scores into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Get top-k word predictions for each masked token
    top_k_probs, top_k_ids = torch.topk(probs, top_k)
    # Display top scores
    print("\n Top predictions to replace masked token:")
    print("----------------------------------------")
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
    print("----------------------------------------")
    print("predicted: ", predicted_word, "| suggested: ", target_word)
    print("predicted: ", predicted_token_id, "| original: ", target_token_id)
    print("\n")

    # Accept an exact match
    if predicted_token_id == target_token_id:
        print("!! Exact Match with suggested repair ")
        batch_eval['correct_count'] += 1
        batch_eval['exact_match'] += 1
        result = "Exact Match"
        score = False

    else: # If not an exact match, check if prediction is a synonym of target
        score = check_synonyms(predicted_word, target_word, sentence_tokens)
        #print("Synonym score: ", score)
        if score > 0.75:
            print("Synonym match!")
            batch_eval['correct_count'] += 1
            batch_eval['syn_match'] += 1
            result = "Synonym Match"
        else:
            print("!!! No match found ")
            result = False

    return [predicted_word, target_word, result, score]





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

    # Get POS type from nlkt tag -- https://www.nltk.o/book/ch05.html
    #print(nltk.pos_tag([target]))
    if pred_tagged[1] in ["PRP", "PRP$", "WP", "WP$"] or targ_tagged[1] in ["PRP", "PRP$", "WP", "WP$"]:
        print("-- Word is a pronoun so cannot check synonyms")
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
                    print(s1, ",", s2, "sim score: ", round(sim,4) )
                    max_score = round(sim,4)
        print("-- Max Synonyms Score: ", max_score)
        return round(max_score,4)
    else:
        print("-- Word has not been found in wordnet")
        return False





def print_highlighted_result(pred_tokens, org_sent):
    #print("Predicted tokens: ", pred_tokens)
    # Repair sentence with predictions - Replace each mask token one by one
    for predicted in pred_tokens:
        org_sent = org_sent.replace(tokenizer.mask_token, f"\\*{predicted}\\*", 1)
    return org_sent


def paraphrase_text(sentence):
    # https://huggingface.co/eugenesiow/bart-paraphrase

    summarizer = pipeline("translation", model="facebook/bart-large-cnn")
    print("1", summarizer(sentence, max_length=500, min_length=30, do_sample=False))

    # sentence = "well he's going get school obvious . mom's telling him to take the umbrella . and he says no I don't need it . I'm gonna be alright . I'll go out . and oo it's raining . run back . I'm all wet . walk in . I'm drenched . changed my clothes . mom gives me the umbrella . and away I go to school . huh ?"
    # print(sentence)

    # Paraphrase output
    print("Paraphrased text:")
    batch = bart_tokenizer(sentence, return_tensors='pt')
    generated_ids = bart_model.generate(batch['input_ids'])
    generated_sentence = bart_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("2", generated_sentence)


def testing(log_writer, run_writer):
    # Prepare testing data
    print("loading test data..")
    # Get a dataset containing the masked version of all transcripts within the training data directory
    test_dataset = TestingData.mask_from_directory(max_context_length)

    # Batch this dataset to step through each data instance
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )
    print("Dataset Loaded")

    # Run BERT on testing data
    print("running model..")
    run_bert(loader, run_writer)

    # [max_context_length, use_control_training_data, use_aphasia_training_data, learning_rate, batch_size,
    #                     "epoch1", "epoch2", "epoch3",
    #                     "correct_count", "total_count", "exact_match", "syn_match", "accuracy"])

    if batch_eval['total_count'] > 0:
        row = set_up_vars + epoch_losses + list(batch_eval.values()) + [
            batch_eval['correct_count'] / batch_eval['total_count'] * 100]
        print(row)
        log_writer.writerow(row)




def train_and_run(save_model_dir):
    # Create log files
    eval_results_filename = f"pred_{max_context_length}_{use_control_training_data}_{use_aphasia_training_data}.csv"
    run_writer, run_file, log_writer, log_file = LogResults.create_results_files(
        eval_results_filename)  # returns writers to append results + file objects to close

    # Setup models
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mlm_model.to(device)  # Move the model to the device ("cpu" or "cuda")
    bart_model.to(device)


    # Prepare training data and run training on the BERT model
    print("getting training data ..")
    if use_control_training_data or use_aphasia_training_data:
        print("--------- Preparing training data ----------")
        dataloader = TrainingData.get_training_data(use_control_training_data, use_aphasia_training_data,
                                                    max_context_length, batch_size)
        training(mlm_model, dataloader, device)

    # Prepare testing data and run on model
    print("--------- Preparing testing data ----------")
    testing(log_writer, run_writer)

    # Close log files
    run_file.close()
    log_file.close()

    # Run model on random tokens in the dataset + calculate overall produced accuracy
    # accuracy_random_tokens(training_data, mlm_model, tokenizer, device)

    #mlm_model.save_pretrained(save_model_dir)
    #print("saving model..")
    #torch.save(mlm_model, save_model_dir)


def run_from_save(save_model_dir):
    # Create log files
    eval_results_filename = f"pred_{max_context_length}_{use_control_training_data}_{use_aphasia_training_data}.csv"
    run_writer, run_file, log_writer, log_file = LogResults.create_results_files(
        eval_results_filename)  # returns writers to append results + file objects to close

    # Setup models
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("loading model..")
    mlm_model = torch.load(save_model_dir)
    mlm_model.to(device)  # Move the model to the device ("cpu" or "cuda")
    bart_model.to(device)

    # Prepare testing data and run on model
    print("testing model..")
    testing(log_writer, run_file)

    # Close log files
    run_file.close()
    log_file.close()






def main():
    global batch_size
    global max_context_length
    global set_up_vars

    # Setup location to save the model
    save_model_dir = "C:/Users/nat/PycharmProjects/PythonProject/utils/models/test-model"
    os.makedirs(save_model_dir, exist_ok=True)

    batch_size_s = [32] # -- update batch size here - as list of trials
    context_length = [10]  # maximum: 512 -- update size of context here
    for size in batch_size_s:
        batch_size = size
        set_up_vars[4] = size

        for length in context_length:
            max_context_length = length
            set_up_vars[0] = max_context_length
            #run_from_save(save_model_dir)
            train_and_run(save_model_dir)

    # display all predictions
    for pair in sentences:
        masked = pair[0]
        completed = pair[1]
        predictions = pair[2]
        repaired = masked

        # highlight generated words
        for predicted in predictions:
            repaired = repaired.replace('[MASK]', f"\\*{predicted}\\*", 1)

        masked = masked.replace('[PAD]', '')
        repaired = repaired.replace('[PAD]','')

        print("\n")
        print("Masked Utterance: ", masked)
       # print("Completed Utterance: ", completed)
        print ("Repaired: ", repaired)



if __name__ == '__main__':
    main()
