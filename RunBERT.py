import string
import random
import torch

# Mask random words within the data + predict them -- calculate success across all predictions
def accuracy_random_tokens(data, model, tokenizer, device):
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
        punc_tokens = tokenizer.encode(string.punctuation, add_special_tokens=False)

        candidate_tokens = [
            i for i, token_id in enumerate(tokens)
            if token_id not in special_tokens and token_id not in punc_tokens
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

