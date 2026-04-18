import os
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
from nltk.parse.corenlp import CoreNLPDependencyParser
from node2vec import Node2Vec
from .utils import get_nlp

def calculate_idf(sents_dir, n_gram=3):
    """Calculates IDF from cleaning sentences in a directory."""
    freq_dict = {}
    files = [f for f in os.listdir(sents_dir) if f.endswith('.csv')]
    num_docs = len(files)
    
    for filename in files:
        df = pd.read_csv(os.path.join(sents_dir, filename))
        cleaned_sents = df['cleaned'].dropna().astype(str).tolist()
        
        unique_tokens = set()
        for sent in cleaned_sents:
            words = sent.split()
            for n in range(1, n_gram + 1):
                for i in range(len(words) - n + 1):
                    unique_tokens.add(' '.join(words[i:i+n]))
        
        for token in unique_tokens:
            freq_dict[token] = freq_dict.get(token, 0) + 1
            
    idf_dict = {token: np.log(num_docs / freq) for token, freq in freq_dict.items()}
    return idf_dict, freq_dict, num_docs

def apply_tfidf(sents_dir, idf_dict, n_gram=3):
    """Applies TF-IDF scores to sentences in a directory."""
    files = [f for f in os.listdir(sents_dir) if f.endswith('.csv')]
    
    for filename in files:
        path = os.path.join(sents_dir, filename)
        df = pd.read_csv(path)
        sentences = df['cleaned'].dropna().astype(str).tolist()
        
        if not sentences:
            continue
            
        for n in range(1, n_gram + 1):
            tfidf_scores = []
            for sent in sentences:
                words = sent.split()
                n_grams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                
                if not n_grams:
                    tfidf_scores.append(0.0)
                    continue
                    
                score = 0
                for ng in n_grams:
                    tf = n_grams.count(ng) / len(n_grams)
                    idf = idf_dict.get(ng, 0)
                    score += tf * idf
                tfidf_scores.append(score)
            
            max_s = max(tfidf_scores) if tfidf_scores else 0
            if max_s > 0:
                tfidf_scores = [s / max_s for s in tfidf_scores]
                
            df[f'tf_idf_n_gram_{n}'] = tfidf_scores
            
        df.to_csv(path, index=False)

def generate_bert_embeddings(sents_dir, output_dir, model_name="en_core_web_trf"):
    """Generates BERT embeddings using spaCy transformers."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        bert_nlp = spacy.load(model_name)
    except Exception as e:
        print(f"Error loading transformer model {model_name}: {e}")
        return

    files = [f for f in os.listdir(sents_dir) if f.endswith('.csv')]
    for filename in files:
        df = pd.read_csv(os.path.join(sents_dir, filename))
        sentences = df['sentences'].dropna().astype(str).tolist()
        
        embeddings = []
        for sent in sentences:
            try:
                doc = bert_nlp(sent)
                tokvecs = doc._.trf_data.last_hidden_layer_state[0]
                embed = np.mean(tokvecs.data, axis=0).tolist()
                
                blob = TextBlob(sent)
                embed.extend([blob.sentiment.polarity, blob.sentiment.subjectivity])
                embeddings.append(embed)
            except Exception:
                embeddings.append([0.0] * 770) 
        
        np.save(os.path.join(output_dir, filename[:-4] + '.npy'), np.array(embeddings))

def generate_node_embeddings(sents_dir, output_dir, corenlp_url='http://localhost:9000'):
    """Generates Node2Vec embeddings using Stanford CoreNLP dependency parser."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        dep_parser = CoreNLPDependencyParser(url=corenlp_url)
        nlp = get_nlp()
    except Exception as e:
        print(f"Error connecting to CoreNLP or loading NLP: {e}")
        return

    files = [f for f in os.listdir(sents_dir) if f.endswith('.csv')]
    for filename in files:
        df = pd.read_csv(os.path.join(sents_dir, filename))
        sentences = df['sentences'].dropna().astype(str).tolist()
        
        embeddings = []
        for sent in sentences:
            try:
                parse, = dep_parser.raw_parse(sent)
                word_tokens = [line.split('\t') for line in parse.to_conll(4).split('\n') if line.strip()]
                
                root_idx = 0
                root_word = ""
                for i, tokens in enumerate(word_tokens):
                    if len(tokens) >= 4 and tokens[3] == "ROOT":
                        root_idx = i
                        root_word = tokens[0]
                        break
                
                G = parse.nx_graph()
                node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, quiet=True)
                model = node2vec.fit(window=10, min_count=1)
                
                node_vec = model.wv.get_vector(root_idx)
                word_vec = nlp(root_word).vector
                
                combined = np.concatenate((node_vec, word_vec), axis=0)
                embeddings.append(combined)
            except Exception:
                embeddings.append(np.zeros(224)) 
        
        np.save(os.path.join(output_dir, filename[:-4] + '.npy'), np.array(embeddings))
