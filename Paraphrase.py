from transformers import BartForConditionalGeneration, BartTokenizer #BART paraphrase


# Get BART paraphrase model from Hugging Face
bart_model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
bart_tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')


def paraphrase_text(sentence):
    # https://huggingface.co/eugenesiow/bart-paraphrase

    # summarizer = pipeline("translation", model="facebook/bart-large-cnn")
    # print(summarizer(sentence, max_length=130, min_length=30, do_sample=False))

    # sentence = "well he's going get school obvious . mom's telling him to take the umbrella . and he says no I don't need it . I'm gonna be alright . I'll go out . and oo it's raining . run back . I'm all wet . walk in . I'm drenched . changed my clothes . mom gives me the umbrella . and away I go to school . huh ?"
    # print(sentence)

    # Paraphrase output
    print("Paraphrased text:")
    batch = bart_tokenizer(sentence, return_tensors='pt')
    generated_ids = bart_model.generate(batch['input_ids'])
    generated_sentence = bart_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_sentence)
