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
    # Prepare test data
    #test_file_path = '../masked_testing_williamson.txt'
    #target_words = get_target_words(test_file_path)

    # Create testing dataset
    #test_dataset = TrainingData.dataset_from_file(test_file_path)
    #print("Test dataset size: ", len(test_dataset))

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

    for batch in loader:
        # Get batch information
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        encoded_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Get attached info
        #masked_text = batch['masked_sentence']
        #target_words = batch['target_ids']

        target_ids_list = []
        for i, target in enumerate(batch['target_ids']):
            target_ids_list = target.tolist()

       # print("target words from batch: ", target_ids_list)

        # Get positions of each mask token
        pos_masks = [torch.where(seq ==  tokenizer.mask_token_id)[0].tolist() for seq in input_ids]  # 2d array -- list of indexes for mask tokens in each sentence

        #get_test_labels(input_ids)

        # If no mask tokens have been found there's no need to run the model
        if pos_masks == [[]]:
            print("-- No mask tokens found --")
            unmasked_sentence = tokenizer.decode(encoded_inputs["input_ids"][0], skip_special_tokens=True)
            return unmasked_sentence

        # Pass into mlm model and output predictions for masked words
        result, accuracy = output_results(target_ids_list, encoded_inputs, pos_masks)

        # store outputs
        results.append(result)
        accuracies.append(accuracy)

    print("All accuracies: ", accuracies)

    return results, accuracies





def display_predictions(logits, top_k ):
    # Softmax raw scores into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Get top-k word predictions for each masked token
    top_k_probs, top_k_ids = torch.topk(probs, top_k)
    # Display top scores
    print("Top predictions to replace masked token:")
    for prob, token_id in zip(top_k_probs, top_k_ids):
        token_str = tokenizer.decode([token_id.item()]).strip()
        # tokens = tokenizer.convert_ids_to_tokens([token_id.item()])
        # token_str = tokenizer.convert_tokens_to_string(tokens).strip()
        print(f"Token: {token_str}, Score: {prob.item():.4f}")




def output_results(target_words, encoded_inputs, pos_masks ):
    # @ adapted from https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
    print('..Passing into model..')
    # Pass into MLM model and get raw score outputs
    outputs = mlm_model(**encoded_inputs)

    accuracy = 0
    unmasked_sentence = tokenizer.decode(encoded_inputs["input_ids"][0], skip_special_tokens=True)

    # For each sentence i, predict each mask in sentence
    for i, mask_positions in enumerate(pos_masks):

        # Skip sentence if there are no masked tokens
        if  len(mask_positions) == 0:
            continue

        # Display sentence and tokens
        #print('\n---- Sentence ', i + 1)
        #print('Original:', masked_sentences[i])
        #print('Tokenized:', tokenizer.tokenize(masked_sentences[i]))
        #print('Token IDs:', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(masked_sentences[i])))

        predicted_tokens = []
        mask_num = -1
        correct = 0
        total = 0
        print('Target words:', target_words)

        sequence_length = outputs.logits.shape[1]

        for pos in mask_positions:
            mask_num += 1

            # Get target word from list
            target_token_id = target_words[mask_num]
            target_word = tokenizer.decode(target_token_id)

            # Get raw prediction scores
            logits = outputs.logits[i, pos]
            # Get top prediction
            predicted_token_id = logits.argmax().item()
            predicted_word = tokenizer.decode(predicted_token_id)

            # predicted_tokens.append(f"*{predicted_word}*") # highlight replaced word in output
            predicted_tokens.append(f"{predicted_word}")

            # Display top predictions
            print("\nSuggested target word is : ", target_word, " : ", target_token_id)

            # Replace mask token with the top prediction #input_ids[i, pos] = top_k_ids[0]

            if predicted_token_id == target_token_id:
                correct += 1
                print("Match! ")
            else:
                print("!!! No match: ")
                display_predictions(logits, 2)
                print("predicted: ", predicted_word, "| original: ", target_word)
                print("predicted: ", predicted_token_id, "| original: ", target_token_id)
            total += 1

        if total > 0:
            accuracy = correct / total
            print(f"\n Accuracy: {accuracy * 100:.2f}% on {total} sequences")

        # Repair sentence with predictions - Replace each mask token one by one
        for predicted in predicted_tokens:
            unmasked_sentence = unmasked_sentence.replace(tokenizer.mask_token, predicted, 1)

        # Decode repaired sentence from tokens
        #unmasked_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        #print(f"\nUnmasked sentence with top predictions:\n{unmasked_sentence}\n")

    return unmasked_sentence, accuracy




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
