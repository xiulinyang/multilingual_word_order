from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import re

random.seed(42)

tokenizer_multi = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model_multi = AutoModelWithLMHead.from_pretrained("bert-base-multilingual-cased")
# DeepPavlov/rubert-base-cased
# dbmdz/bert-base-turkish-cased
# dbmdz/bert-base-german-cased
# nlpaueb/bert-base-greek-uncased-v1
# TurkuNLP/bert-base-finnish-cased-v1
# dkleczek/bert-base-polish-uncased-v1
tokenizer_mono = AutoTokenizer.from_pretrained("bert-base-uncased")
model_mono = AutoModelWithLMHead.from_pretrained("bert-base-uncased")


def get_mask_prob(input_file_path, model, tokenizer):
    sum_prob = 0
    with open(input_file_path, 'r') as input_file:
        parallel_sents = input_file.readlines()
        if len(parallel_sents) > 2000:
            parallel_sents = random.sample(parallel_sents, 2000)
        sentss = [x.split('\t')[0] for x in parallel_sents]
        tokens = [x.split('\t')[1] for x in parallel_sents]
        for sent, tok in tqdm(zip(sentss, tokens)):
            target_words = tokenizer(tok.strip())['input_ids'][1:-1]
            target_words_tokens = tokenizer.convert_ids_to_tokens(target_words)
            num_masks = len(target_words_tokens) - 1
            mask_id = sent.split().index('[MASK]')
            sent_add_mask = sent.split()[:mask_id + 1] + ['[MASK]'] * num_masks + sent.split()[mask_id + 1:]
            sent_add_mask = ' '.join(sent_add_mask)
            inputs = tokenizer(sent_add_mask, return_tensors="pt")
            mask_token_indexes = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits

            # Softmax to convert logits to probabilities
            softmax = torch.nn.Softmax(dim=-1)

            # Get probabilities for each sub-token of the target word
            probabilities = 1.0
            for idx, sub_token in zip(mask_token_indexes, tokenizer.tokenize(tok)):
                sub_token_id = tokenizer.convert_tokens_to_ids(sub_token)
                sub_token_probs = softmax(predictions[0, idx])
                sub_token_prob = sub_token_probs[sub_token_id].item()
                probabilities *= sub_token_prob
            sum_prob += probabilities

    print(sum_prob / len(sentss))
    return sum_prob / len(sentss)


# parallel_prob = last_token_prob('de_parallel.txt', 'de_parallel_output.txt')
# diff_prob = last_token_prob('de_different.txt', 'de_diff_output.txt')
parallel_prob_multi = get_mask_prob('en_parallel_sv_unfill.txt', model_multi, tokenizer_multi)
parallel_prob_mono = get_mask_prob('en_parallel_sv_unfill.txt', model_mono, tokenizer_mono)

parallel_ratio = parallel_prob_multi / parallel_prob_mono

diff_prob_multi = get_mask_prob('en_different_vs_unfill.txt', model_multi, tokenizer_multi)
diff_prob_mono = get_mask_prob('en_different_vs_unfill.txt', model_mono, tokenizer_mono)

diff_ratio = diff_prob_multi / diff_prob_mono
print('results')
print(parallel_prob_multi)
print(parallel_prob_mono)
print(diff_prob_multi)
print(diff_prob_mono)
print(parallel_ratio)
print(diff_ratio)
print(parallel_ratio / diff_ratio)
# parallel_prob = last_token_prob('ru_parallel.txt', 'ru_parallel_output.txt')
# diff_prob = last_token_prob('ru_different.txt', 'ru_diff_output.txt')
