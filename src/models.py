import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from matplotlib import pyplot as plt

def build_lstm_model(input_shape=(1, 770)):
    inputs = Input(shape=input_shape)
    l1 = Bidirectional(LSTM(units=50))(inputs)
    outputs = Dense(1, activation='sigmoid')(l1)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(sents_dir, embed_dir, model_save_path, scaler_save_path, input_dim=770, epochs=100, batch_size=128):
    """Trains an LSTM model on embeddings and labels."""
    sents_files = [f for f in os.listdir(sents_dir) if f.endswith('.csv')]
    
    x_data = []
    y_data = []
    
    for filename in sents_files:
        embed_file = filename[:-3] + 'npy'
        embed_path = os.path.join(embed_dir, embed_file)
        if not os.path.exists(embed_path):
            continue
            
        df = pd.read_csv(os.path.join(sents_dir, filename))
        labels = df['label'].tolist()
        embeddings = np.load(embed_path)
        
        if len(embeddings) == len(labels):
            x_data.extend(embeddings)
            y_data.extend(labels)
        else:
            print(f"Dimension mismatch for {filename}: {len(embeddings)} vs {len(labels)}")
            
    if not x_data:
        print("No data found for training.")
        return
        
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x_data)
    joblib.dump(scaler, scaler_save_path)
    
    X = x_scaled.reshape(-1, 1, input_dim)
    
    model = build_lstm_model(input_shape=(1, input_dim))
    
    history = model.fit(X, y_data, epochs=epochs, batch_size=batch_size, validation_split=0.3, verbose=1)
    
    model.save(model_save_path)
    
    plot_history(history, model_save_path.replace('.keras', ''))
    
    return model

def plot_history(history, prefix):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig(f"{prefix}_plots.png")
    plt.close()

def predict_labels(sents_dir, embed_dir, model_path, scaler_path, input_dim=770, threshold=0.3):
    """Predicts labels for sentences and updates CSVs."""
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    files = [f for f in os.listdir(sents_dir) if f.endswith('.csv')]
    for filename in files:
        embed_path = os.path.join(embed_dir, filename[:-3] + 'npy')
        if not os.path.exists(embed_path):
            continue
            
        df = pd.read_csv(os.path.join(sents_dir, filename))
        embeddings = np.load(embed_path)
        
        if len(embeddings) == 0:
            continue
            
        x_scaled = scaler.transform(embeddings)
        X = x_scaled.reshape(-1, 1, input_dim)
        
        preds = model.predict(X, verbose=0)
        binary_preds = [1 if p > threshold else 0 for p in preds]
        
        df['predicted label'] = binary_preds
        df.to_csv(os.path.join(sents_dir, filename), index=False)
