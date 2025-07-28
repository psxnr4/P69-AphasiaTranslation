
import re
import difflib

print("\n---\n Finding the erronous words within the utterance using the coding annotations. Masking this word in the 'label-free' transcript.")

#williamson02
original = ['&-uh <the mother> [/] &-uh the mother and the child &-uh are &-uh arguing over the raincoat [: umbrella] [* s:r-ret].']
flo = ['the mother the mother and the child are arguing over the raincoat .']


# Find coding label for the repair and save the relevant token
# Match regex pattern:
#  - Capture a word (\b\w+\b)
#  - Followed by whitespace and an annotation \[: .*? \]
pattern = r'(\b\w+\b)\s+\[:\s*.*?\]'
err = re.findall(pattern, original[0])
print("Errors found within the string: ", err)

# Split utterance at whitespace into tokens
orig_tokens = original[0].split()
flo_tokens = flo[0].split()

print("\n Compare original sentence: ", original)
print("against flo sentence: ", flo, "\n")

# Find corresponding position in flo
matcher = difflib.SequenceMatcher(None, orig_tokens, flo_tokens) # https://docs.python.org/3/library/difflib.html
# "Return list of 5-tuples describing how to turn a into b. Each tuple is of the form (tag, i1, i2, j1, j2)." Tag is a string: 'replace', 'delete', 'insert', 'equal'
result = flo_tokens[:]

for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(
        tag, i1, i2, j1, j2, orig_tokens[i1:i2], flo_tokens[j1:j2]))

    if tag == 'equal':
        # look at the last token in the matching block
        block_len = i2 - i1 -1
        orig_word = orig_tokens[i1 + block_len]
        flo_index = j1 + block_len
        # if the word is in error list then it may be incorrect
        if orig_word in err:
            # check if the next token in the original utterance is an error label
            if i1 + block_len + 1< len(orig_tokens):
                next_token = orig_tokens[i1 + block_len + 1]
                if next_token.startswith('[:'):
                    # replace this token with a mask
                    result[flo_index] = '[MASK]'
                    err.remove(orig_word)
                    continue

# Create the complete string
final_result = ' '.join(result)
print('\n Masked string: ', [final_result])
