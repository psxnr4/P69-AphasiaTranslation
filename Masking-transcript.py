# Finds the erroneous words within the utterance using the coding annotations, and masks this word in the label-free transcript.")
# Writes a file containing the mask version of the transcript, writes the target words in a header line

import re
import difflib
from transformers import AdamW, BertTokenizer, BertForMaskedLM, set_seed, DataCollatorForLanguageModeling
import os

# Initialize the tokenizer from the model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#output_path = 'Masked-transcript.txt'

#sample williamson02
original = ['&-uh <the mother> [/] &-uh the mother and the child &-uh are &-uh arguing over the raincoat [: umbrella] [* s:r-ret].']
flo = ['the mother the mother and the child are arguing over the raincoat .']

# INPUT TRANSCRIPT
#raw_gem_path = 'C:/Users/nat/OneDrive - The University of Nottingham/Documents/Dissertation/P69 - Aphasia Project/4. Analysis and Results/Williamson-Umbrella-Gem/williamson05a.umbrella.gem.cex'
#flo_gem_path = 'C:/Users/nat/OneDrive - The University of Nottingham/Documents/Dissertation/P69 - Aphasia Project/4. Analysis and Results/Williamson-Umbrella-Gem/Cleaned-Flo/williamson05a.umbrella.gem.flo.cex'


# INPUT DIRECTORY
main_dir = 'C:/Users/nat/OneDrive - The University of Nottingham/Documents/Dissertation/P69 - Aphasia Project/4. Analysis and Results/Test Data'
raw_gem_dir = main_dir + '/Raw Transcript'
flo_gem_dir = main_dir + '/Flo Output'


def get_masked_dir( ):

    output_lines = []
    target_words = []
    mask_count = 0

    # Get all .cex files in the directory
    for filename in os.listdir(raw_gem_dir):
        if filename.endswith('.cex'):
            name, ext = os.path.splitext(filename) # 'williamson18a.umbrella.gem' , '.cex'
            raw_gem_path = os.path.join(raw_gem_dir, filename) # williamson18a.umbrella.gem.cex
            flo_gem_path = os.path.join(flo_gem_dir, name+'.flo.cex') # williamson18a.umbrella.gem.flo.cex

            # read in the transcript and analyse each line independently
            with open(raw_gem_path, 'r', encoding='utf-8') as f:
                raw_gem_content = f.read()

            # Remove error labels attached to words
            # Split into a list to analyse each utterance separately
            raw_gem = raw_gem_content.replace('@u', '').split('\n')
            raw_gem = list(filter(None, raw_gem))  # filter empty lines

            # remove header lines from transcript - all indexes up to gem marker
            index = raw_gem.index('@G:\tUmbrella')
            del raw_gem[0:index+1]

            print(raw_gem)

            # read in the cleaned transcript - this has no error coding and will be the input to our models
            with open(flo_gem_path, 'r', encoding='utf-8') as f:
                flo_gem_content = f.read()
            # Split each line
            flo_gem = flo_gem_content.split('\n')

            # Analyse each line separately
            for index in range(len(raw_gem)):
                print('\n--Utterance ', index, '--')

                # remove error annotations on words
                raw_gem[index] = raw_gem[index].replace('@u', '')

                print("Labelled : ", raw_gem[index])
                print("Clean : ", flo_gem[index])

                # Find coding label for the repair and save the relevant token
                # Match regex pattern:
                #  - Capture a word (\b\w+\b)
                #  - Followed by whitespace and an annotation \[: .*? \]
                pattern = r'(\b\w+\b)\s+\[:\s*.*?\]'
                err = re.findall(pattern, raw_gem[index])
                #print("Word errors found: ", err)

                # If no word error has been found then continue
                if len(err) == 0:
                    output_lines.append(flo_gem[index])
                    continue

                # Split utterance at whitespace into tokens
                orig_tokens = raw_gem[index].split()
                # Utterance with error coding + target word suggestion e.g. ['the', 'mother', 'and', 'child', 'are', 'arguing', 'over', 'the', 'raincoat', '[:', 'umbrella]' ]
                flo_tokens = flo_gem[index].split()
                # Cleaned version of the utterance without repairs e.g.  ['the', 'mother', 'and', 'child', 'are', 'arguing', 'over', 'the', 'raincoat']
                #print("Original tokens: ", orig_tokens)
                #print("Flo tokens: ", flo_tokens)

                # Find position in flo-output that corresponds to the highlighted word error
                matcher = difflib.SequenceMatcher(None, orig_tokens, flo_tokens) # https://docs.python.org/3/library/difflib.html
                # "Return list of 5-tuples describing how to turn a into b. Each tuple is of the form (tag, i1, i2, j1, j2)." Tag is a string: 'replace', 'delete', 'insert', 'equal'
                result = flo_tokens[:] # Use the clean utterance as our output

                #print('\n')
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
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
                                    # replace this token with a mask
                                    result[flo_index] = tokenizer.mask_token
                                    err.remove(last_match)
                                    # store target word
                                    target = orig_tokens[i1 + block_len + 2] # this will be 'word]'
                                    target_words.append(target[:-1]) # remove last char
                                    mask_count +=1
                                    continue

                # Create the complete string
                final_result = ' '.join(result)
                print('Masked : ', [final_result])

                output_lines.append(final_result)

        output_lines.append('|')

    write_to_file(target_words, output_lines, 'masked_testing_williamson.txt')



def write_to_file( target_words, output_lines, output_path):
    with open(output_path, "w", encoding="utf-8") as file:

        for word in target_words:
            file.write(word + ",")

        for line in output_lines:
            file.write("\n"+line)



get_masked_dir()
