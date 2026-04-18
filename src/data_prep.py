import os
import pandas as pd
import spacy
from .utils import clean_sentence_pipeline, get_nlp

def sample_ten_percent(input_path, output_path):
    """Sampling the first 10% of sentences from each text file."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    nlp = get_nlp()
    content_files = os.listdir(input_path)
    
    for ar_file_name in content_files:
        path_to_file = os.path.join(input_path, ar_file_name)
        if not os.path.isfile(path_to_file):
            continue
            
        with open(path_to_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        try:
            doc = nlp(text)
            sentences = list(doc.sents)
            n = max(1, int(len(sentences) / 10))
            cleaned_sentences = [str(sent) for sent in sentences[:n]]
            
            df = pd.DataFrame({"sentences": cleaned_sentences})
            output_file = os.path.join(output_path, ar_file_name[:-3] + 'csv')
            df.to_csv(output_file, index=False)
        except Exception as e:
            print(f"Error processing {ar_file_name}: {e}")

def label_data(sents_path, gold_path):
    """Labelling sentences as relevant (1) or not (0) based on gold summaries."""
    nlp = get_nlp()
    sents_files = os.listdir(sents_path)
    gold_files = os.listdir(gold_path)

    gold_summaries_sentences = {}
    for gs_file in gold_files:
        report_id = gs_file.split('_')[0]
        if report_id not in gold_summaries_sentences:
            gold_summaries_sentences[report_id] = set()
        
        with open(os.path.join(gold_path, gs_file), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            doc = nlp(text)
            for sen in doc.sents:
                gold_summaries_sentences[report_id].add(str(sen).strip())

    for ar_name in sents_files:
        report_id = ar_name.split('.')[0]
        csv_path = os.path.join(sents_path, ar_name)
        df = pd.read_csv(csv_path)
        
        sentences = df['sentences'].tolist()
        labels = []
        cleaned_sentences = []
        
        compare_sentences = gold_summaries_sentences.get(report_id, set())
        
        for x in sentences:
            final_form = clean_sentence_pipeline(x)
            cleaned_sentences.append(final_form)
            
            if str(x).strip() in compare_sentences:
                labels.append(1)
            else:
                labels.append(0)
                
        df['cleaned'] = cleaned_sentences
        df['label'] = labels
        df.to_csv(csv_path, index=False)
