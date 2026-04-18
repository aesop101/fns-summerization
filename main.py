import os
import argparse
from config import *
from src.data_prep import sample_ten_percent, label_data
from src.features import calculate_idf, apply_tfidf, generate_bert_embeddings, generate_node_embeddings
from src.models import train_model, predict_labels
from src.summary_gen import generate_sequential_summaries

def run_pipeline(args):
    # Stage 1: Data Preparation
    if args.stage == 'all' or args.stage == 'data':
        print("--- Stage 1: Data Preparation ---")
        print("Sampling 10% sentences...")
        sample_ten_percent(TRAINING_REPORTS, TRAINING_SENTS)
        sample_ten_percent(VALIDATION_REPORTS, VALIDATION_SENTS)
        
        print("Labelling data...")
        label_data(TRAINING_SENTS, TRAINING_GOLD)
        label_data(VALIDATION_SENTS, VALIDATION_GOLD)

    # Stage 2: Feature Extraction
    if args.stage == 'all' or args.stage == 'features':
        print("\n--- Stage 2: Feature Extraction ---")
        # TF-IDF
        print("Calculating IDF...")
        idf_dict, _, _ = calculate_idf(TRAINING_SENTS, n_gram=N_GRAM)
        print("Applying TF-IDF...")
        apply_tfidf(TRAINING_SENTS, idf_dict, n_gram=N_GRAM)
        apply_tfidf(VALIDATION_SENTS, idf_dict, n_gram=N_GRAM)
        
        # BERT
        if args.include_bert:
            print("Generating BERT embeddings...")
            generate_bert_embeddings(TRAINING_SENTS, TRAINING_BERT_EMBED, BERT_MODEL_NAME)
            generate_bert_embeddings(VALIDATION_SENTS, VALIDATION_BERT_EMBED, BERT_MODEL_NAME)
            
        # Node
        if args.include_node:
            print("Generating Node embeddings (requires CoreNLP at localhost:9000)...")
            generate_node_embeddings(TRAINING_SENTS, TRAINING_NODE_EMBED, CORENLP_URL)
            generate_node_embeddings(VALIDATION_SENTS, VALIDATION_NODE_EMBED, CORENLP_URL)

    # Stage 3: Training
    if args.stage == 'all' or args.stage == 'train':
        print("\n--- Stage 3: Training ---")
        if args.include_bert:
            print("Training BERT LSTM model...")
            train_model(TRAINING_SENTS, TRAINING_BERT_EMBED, BERT_MODEL_PATH, BERT_SCALER_PATH, input_dim=770)
            
        if args.include_node:
            print("Training Node LSTM model...")
            # Dimension depends on spacy model + node2vec (standard is 128 + 96 = 224)
            train_model(TRAINING_SENTS, TRAINING_NODE_EMBED, NODE_MODEL_PATH, NODE_SCALER_PATH, input_dim=224)

    # Stage 4: Prediction & Summary Generation
    if args.stage == 'all' or args.stage == 'predict':
        print("\n--- Stage 4: Prediction & Summary Generation ---")
        if args.include_bert:
            print("Predicting with BERT model...")
            predict_labels(VALIDATION_SENTS, VALIDATION_BERT_EMBED, BERT_MODEL_PATH, BERT_SCALER_PATH, input_dim=770)
            
        if args.include_node:
            print("Predicting with Node model...")
            predict_labels(VALIDATION_SENTS, VALIDATION_NODE_EMBED, NODE_MODEL_PATH, NODE_SCALER_PATH, input_dim=224)
            
        print("Generating sequential summaries...")
        generate_sequential_summaries(VALIDATION_SENTS, VALIDATION_BERT_EMBED, VALIDATION_NODE_EMBED, SUM_BERT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNS Research Pipeline")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "data", "features", "train", "predict"], 
                        help="Stage of the pipeline to run")
    parser.add_argument("--include_bert", action="store_true", default=True, help="Include BERT processing")
    parser.add_argument("--include_node", action="store_true", default=True, help="Include Node processing")
    
    args = parser.parse_args()
    run_pipeline(args)
