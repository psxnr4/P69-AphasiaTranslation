# Finds the erroneous words within the utterance using the coding annotations, and masks this word in the label-free transcript.
# Writes a file containing the mask version of the transcript, writes the target words in a header line
# Creates a dataset using TrainingData module
# -- masks training data stored in a given directory and returns dataset
# -- and helper functions to read masked text from a single file and extract target words from the header

import re
import difflib
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed, DataCollatorForLanguageModeling
import os
import torch

import TrainingData

# Initialize the tokenizer from the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# INPUT DIRECTORY
# TODO: These should be parameters to the function mask_from_directory - but for now we use the same one
main_dir = 'C:/Users/nat/OneDrive - The University of Nottingham/Documents/Dissertation/P69 - Aphasia Project/4. Analysis and Results/Test Data'
raw_gem_dir = main_dir + '/Raw Transcript'
flo_gem_dir = main_dir + '/Flo Output'


# Get gem output from file -- saved under name.umbrella.gem.cex in main raw_gem_dir
def get_gem(filename):
    raw_gem_path = os.path.join(raw_gem_dir, filename)
    # read in the transcript and analyse each line independently
    with open(raw_gem_path, 'r', encoding='utf-8') as f:
        raw_gem_content = f.read()

    # Remove error labels attached to words
    # Split into a list to analyse each utterance separately
    raw_gem = raw_gem_content.replace('@u', '').split('\n')
    raw_gem = list(filter(None, raw_gem))  # filter empty lines

    # remove header lines from transcript - all indexes up to gem marker
    index = raw_gem.index('@G:\tUmbrella')
    del raw_gem[0:index + 1]
    return raw_gem


# Get FLO output saved as name.umbrella.gem.flo.cex in flo_gem_dir
def get_flo(filename):
    name, ext = os.path.splitext(filename)  # ('--.umbrella.gem' , '.cex')
    flo_gem_path = os.path.join(flo_gem_dir, name + '.flo.cex')

    # read in the cleaned transcript - this has no error coding and will be the input to our models
    with open(flo_gem_path, 'r', encoding='utf-8') as f:
        flo_gem_content = f.read()
    # Split each line
    flo_gem = flo_gem_content.split('\n')
    return flo_gem


def mask_from_directory( ):
    print("Retrieving files from directory..")
    # Store across all files
    all_masked_files = []
    all_target_words = []
    all_mask_pos = []
    all_orig_files = []
    mask_count = 0

    # Get all .cex files in the directory
    for filename in os.listdir(raw_gem_dir):
        # Store across all lines in file
        target_words = []
        masked_pos = []
        masked_file = ''
        orig_file = ''

        if filename.endswith('.cex'):
            # Use filename in directory to navigate the needed transcript parts
            raw_gem = get_gem(filename)
            flo_gem = get_flo(filename)

            # Analyse each line separately
            for index in range(len(raw_gem)):
                '''
                print('\n--Utterance ', index, '--')
                print("Labelled : ", raw_gem[index])
                print("Clean : ", flo_gem[index])
                '''

                # remove error annotations on words
                raw_gem[index] = raw_gem[index].replace('@u', '')

                # Find coding label for the repair and save the relevant token
                # Match regex pattern:
                #  - Capture a word (\b\w+\b)
                #  - Followed by whitespace and an annotation \[: .*? \]
                pattern = r'(\b\w+\b)\s+\[:\s*.*?\]'
                err = re.findall(pattern, raw_gem[index])
                #print("Word errors found: ", err)

                orig_file = orig_file + flo_gem[index] + ' '

                # If no word error has been found then continue to next line
                if len(err) == 0:
                    masked_file = masked_file + flo_gem[index] + ' '
                    continue

                # Split utterance at whitespace into tokens
                orig_tokens = raw_gem[index].split()
                # -- Produces Utterance with error coding + target word suggestion e.g. ['the', 'mother', 'and', 'child', 'are', 'arguing', 'over', 'the', 'raincoat', '[:', 'umbrella]' ]
                flo_tokens = flo_gem[index].split()
                # -- Produces Cleaned version of the utterance without repairs e.g.  ['the', 'mother', 'and', 'child', 'are', 'arguing', 'over', 'the', 'raincoat']

                # Define result_tokens string from this line using clean utterance as a base
                result_tokens = flo_tokens[:]

                # Find position in flo-output that corresponds to the highlighted word error
                matcher = difflib.SequenceMatcher(None, orig_tokens, flo_tokens)
                # -- https://docs.python.org/3/library/difflib.html
                # -- "Return list of 5-tuples describing how to turn a into b. Each tuple is of the form (tag, i1, i2, j1, j2)." Tag is a string: 'replace', 'delete', 'insert', 'equal'

                # Step through Sequence Matcher analysis
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    # Display analysis nicely
                    #print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, orig_tokens[i1:i2], flo_tokens[j1:j2]))
                    '''
                    Example output:
                    equal a[0:9] --> b[0:9]['the', 'mother', 'and', 'child', 'are', 'arguing', 'over', 'the', 'raincoat'] --> ['the', 'mother', 'and', 'child', 'are', 'arguing', 'over', 'the', 'raincoat']
                    delete a[9:11] --> b[9:9]['[:', 'umbrella]'] --> []
                    '''

                    # Find the block of tokens that match between the raw + clean utterance
                    # The last token in this block will be the erroneous word - as
                    if tag == 'equal':
                        # Get size of block to find the last token
                        block_len = i2 - i1 -1
                        last_match = orig_tokens[i1 + block_len]
                        # Get corresponding index in the cleaned utterance
                        flo_index = j1 + block_len
                        # If the word is in the error list then it is likely incorrect
                        if last_match in err:
                            # Check that the next token in the original utterance is an error label
                            if i1 + block_len + 1< len(orig_tokens):
                                next_token = orig_tokens[i1 + block_len + 1] # this should be '[:'
                                if next_token.startswith('[:'):
                                    # Replace this token in the line with a mask token
                                    result_tokens[flo_index] = tokenizer.mask_token
                                    err.remove(last_match)

                                    # Get target word from list of string tokens
                                    target = orig_tokens[i1 + block_len + 2] # this will be 'word]'
                                    target = target[:-1] # remove last char
                                    # Get token ID to store value
                                    targetid = tokenizer.convert_tokens_to_ids(target)
                                    target_words.append(targetid)

                                    # Store position of masked token
                                    masked_pos.append(flo_index)
                                    mask_count +=1

                                    continue

                # -- END OF LINE
                # Create the complete string by joining tokens in result_tokens
                final_line = ' '.join(result_tokens)
                #print('Masked : ', [final_line])

                # Append to the masked file
                masked_file = masked_file + final_line + ' '
                #print(masked_file)

             # -- END OF FILE
            # Each line in this file has been collated to:
            # 1. masked sentence as a [string]
            all_masked_files.append(masked_file)
            # 2. target words as an [array]
            all_target_words.append(target_words)
            # 3. mask positions as an [array]
            all_mask_pos.append(masked_pos)
            # 4. original sentence as a [string]
            all_orig_files.append(orig_file)


    # -- END OF ALL FILES
    print("Files masked.")

    # Write all data from this directory
    #write_to_file(all_target_words, all_masked_files, '../masked_testing_williamson.txt')
    # TODO: storing target words as tokenids rather than strings so there will be errors carried over

    print(all_target_words)

    # Create dataset
    dataset = write_to_dataset(all_masked_files, all_target_words)


    return dataset




def tokenise_input(masked_sentences):
    # @ adapted from https://docs.pytorch.org/TensorRT/_notebooks/Hugging-Face-BERT.html
    print('..Tokenising Input..')
    # Tokenise sentences and encode -- include padding as sentences are different lengths
    encoded_inputs = tokenizer(masked_sentences, return_tensors='pt', padding=True)
    input_ids = encoded_inputs["input_ids"]  # shape: [batch_size, sequence_length]

    # Dynamically find all [MASK] token positions in input sentence
    print('..Finding Masks..')
    pos_masks = [torch.where(seq ==  tokenizer.mask_token_id)[0].tolist() for seq in input_ids]  # 2d array -- list of indexes for mask tokens in each sentence

    return encoded_inputs, pos_masks




def write_to_dataset(all_masked_files, all_target_words):
    # Tokenise inputs
    input_encodings, mask_pos = tokenise_input(all_masked_files)

    # Create a dataset of the token encodings and link the masked strings
    dataset = TrainingData.TextDataset(input_encodings, all_target_words)


    print(f"Created Dataset size: {len(dataset)}")

    '''
    # Check first few items
    for i in range(1):
        print(f"\n--- Sample {i} ---")
        sample = dataset[i]
        for key, val in sample.items():
            print(f"{key}: {val}")
    '''


    return dataset


def write_to_file( all_target_words, output_lines, output_path):
    print(output_lines)
    with open(output_path, "w", encoding="utf-8") as file:

        for words in all_target_words:
            for w in words:
                file.write(w + ",")

        for l in output_lines:
            file.write("\n"+l)


# TODO: storing target words as tokenids rather than strings so there will be errors carried over
def get_target_words(file_path):
    # First line of file contains the target words that have been suggested for repairs
    with open(file_path, 'r', encoding='utf-8') as f:
        print('..Reading File..')
        target_words = f.readline().strip()
        target_words = target_words.split(',')

    # Convert target words to token IDs
    target_token_ids = tokenizer(target_words, add_special_tokens=False, return_tensors=None)
    return target_token_ids



def get_masked_input_from_file():
    # Define input sentence constructed from utterances
    with open('../Masked-transcript.txt', 'r', encoding='utf-8') as f:
        print('..Reading File..')
        # First line of file contains the target words that have been suggested for repairs
        target_words = f.readline().strip()
        target_words = target_words.split(',')
        # Read in the rest of the transcript
        masked_content = f.read()

    return target_words, [masked_content.strip()]

