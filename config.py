import os

DATA_DIR = 'fns2020_dataset'
TRAINING_REPORTS = os.path.join(DATA_DIR, 'training', 'annual_reports')
VALIDATION_REPORTS = os.path.join(DATA_DIR, 'validation', 'annual_reports')
TRAINING_GOLD = os.path.join(DATA_DIR, 'training', 'gold_summaries')
VALIDATION_GOLD = os.path.join(DATA_DIR, 'validation', 'gold_summaries')

TRAINING_SENTS = 'training_sents'
VALIDATION_SENTS = 'validation_sents'
TRAINING_BERT_EMBED = 'training_sents_bert_embedding'
VALIDATION_BERT_EMBED = 'validation_sents_bert_embedding'
TRAINING_NODE_EMBED = 'training_sents_node_embedding'
VALIDATION_NODE_EMBED = 'validation_sents_node_embedding'

SUM_BERT_DIR = 'SUM_Bert'
SUM_NODE_DIR = 'SUM_Node'

BERT_MODEL_PATH = 'bert_embedding_lstm.keras'
NODE_MODEL_PATH = 'node_embedding_lstm.keras'
BERT_SCALER_PATH = 'bert_embedding_scaler.gz'
NODE_SCALER_PATH = 'node_embedding_scaler.gz'

CORENLP_URL = 'http://localhost:9000'

N_GRAM = 3
BERT_MODEL_NAME = "en_core_web_trf"
SPACY_MODEL_NAME = "en_core_web_sm"
LSTM_UNITS = 50
BATCH_SIZE = 128
EPOCHS = 100
VALIDATION_SPLIT = 0.3
PREDICTION_THRESHOLD = 0.3
