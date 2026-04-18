import os
import pandas as pd
import numpy as np
from .utils import get_chunks, get_text_from_indices, get_mean_score, normalize_scores

def generate_sequential_summaries(sents_dir, bert_embed_dir, node_embed_dir, output_dir):
    """Generates sequential summaries for validation set based on predicted labels."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sents_files = [f for f in os.listdir(sents_dir) if f.endswith('.csv')]
    
    for filename in sents_files:
        path = os.path.join(sents_dir, filename)
        df = pd.read_csv(path)
        
        if 'predicted label' not in df.columns:
            continue
            
        sents = df['sentences'].tolist()
        labels = df['predicted label'].tolist()
        
        bert_path = os.path.join(bert_embed_dir, filename[:-3] + 'npy')
        node_path = os.path.join(node_embed_dir, filename[:-3] + 'npy')
        
        if not os.path.exists(bert_path) or not os.path.exists(node_path):
            continue
            
        bert_feat = np.load(bert_path)
        node_feat = np.load(node_path)
        
        bert_scores = normalize_scores([np.mean(f) for f in bert_feat])
        node_scores = normalize_scores([np.mean(f) for f in node_feat])
        
        selected_indices = [i for i, l in enumerate(labels) if l == 1]
        if not selected_indices:
            continue
            
        selected_sents = [sents[i] for i in selected_indices]
        selected_bert = [bert_scores[i] for i in selected_indices]
        selected_node = [node_scores[i] for i in selected_indices]
        
        tfidf_cols = [c for c in df.columns if 'tf_idf_n_gram' in c]
        selected_tfidf = {col: [df[col].tolist()[i] for i in selected_indices] for col in tfidf_cols}

        chunks = get_chunks(selected_sents)
        
        summary_data = []
        for tup in chunks:
            row = {
                "summary": get_text_from_indices(selected_sents, tup),
                "bert_score": get_mean_score(selected_bert, tup),
                "node_score": get_mean_score(selected_node, tup)
            }
            for col, scores in selected_tfidf.items():
                row[f"{col}_score"] = get_mean_score(scores, tup)
            summary_data.append(row)
            
        pd.DataFrame(summary_data).to_csv(os.path.join(output_dir, filename), index=False)
