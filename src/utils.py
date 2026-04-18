import re
import contractions
import spacy
from spacy.lang.en import stop_words
import numpy as np

# Load spacy model for utils (using config eventually)
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_sm')
    return _nlp

def remove_telephone_numbers(sentence):
    sentence = str(sentence)
    i = 0
    while i < len(sentence):
        if sentence[i] == '+':
            count = 0
            j = i + 1
            while j < len(sentence):
                if sentence[j].isdigit():
                    count += 1
                    j += 1
                elif sentence[j] in [' ', '(', ')']:
                    j += 1
                else:
                    break
            if 10 <= count <= 13:
                sentence = sentence.replace(sentence[i:j], "")
                i = -1 
        i += 1
    return sentence

def remove_addresses_sent(text):
    text = str(text)
    patterns = [
        r"\S*www\.\S+", r"\S*WWW\.\S+", r"\S*\.com\S+", r"\S*\.co\S+",
        r"\S*\.COM\S+", r"\S*\.gov\S+", r"\S*\.biz\S+", r"\S*\.org\S+", r"\S*\.uk\S+"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text

def preprocess_text(sentence):
    sentence = str(sentence)
    sentence = re.sub(r"[^a-zA-Z0-9]", " ", sentence)
    
    nlp = get_nlp()
    tokens = nlp(sentence)
    stop_list = list(stop_words.STOP_WORDS)

    lemmatized = []
    for token in tokens:
        token_lemma = str(token.lemma_).lower().strip()
        if token_lemma and token_lemma not in stop_list:
            lemmatized.append(token_lemma)
    return lemmatized

def clean_sentence_pipeline(sent):
    sent = remove_telephone_numbers(sent)
    sent = remove_addresses_sent(sent)
    
    expanded_words = [contractions.fix(word) for word in sent.split()]
    expanded_text = ' '.join(expanded_words)
    
    tokens = preprocess_text(expanded_text)
    return ' '.join(tokens)

def get_chunks(sent_list, max_step=1000):
    """Chunk sentences into groups based on word count."""
    sent_length = []
    for sent in sent_list:
        sent = str(sent)
        temp = sent.split(' ')
        sent_length.append(len(temp))
    
    seq_indices = []
    start = 0
    step = 0
    count = 0
    while count < len(sent_length):
        if step < max_step:
            step += sent_length[count]
            count += 1
        else:
            stop = count - 1
            seq_indices.append((start, stop))
            start = count
            step = sent_length[count]
            count += 1
            
    if count > start:
        seq_indices.append((start, count - 1))
    return seq_indices

def get_text_from_indices(sent_list, indices_tuple):
    a, b = indices_tuple
    return ' '.join(str(sent_list[i]) for i in range(a, b + 1))

def get_mean_score(score_list, indices_tuple):
    a, b = indices_tuple
    sub_list = score_list[a:b+1]
    if not sub_list:
        return 0.0
    return np.mean(sub_list)

def normalize_scores(scores):
    scores = np.array(scores)
    max_val = np.max(scores)
    min_val = np.min(scores)
    deno = max_val - min_val
    if deno == 0:
        return np.zeros_like(scores)
    return (scores - min_val) / deno
