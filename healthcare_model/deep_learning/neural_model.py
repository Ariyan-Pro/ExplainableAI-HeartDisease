"""
Neural Network Models for Heart Disease Prediction
Deep learning alternatives to XGBoost
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Input, Dropout, BatchNormalization, 
                                   Conv1D, MaxPooling1D, Flatten, LSTM, GRU)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, Tuple, List  # ADD THIS IMPORT
import numpy as np

class NeuralHeartModel:
    ""Neural network models for heart disease prediction""
    
    def __init__(self, input_dim: int, model_type: str = "dense"):
        self.input_dim = input_dim
        self.model_type = model_type
        self.model = None
        self.history = None
    
    def build_dense_model(self, hidden_layers: List[int] = [64, 32, 16], 
                         dropout_rate: float = 0.3) -> Model:
        """Build dense neural network"""
        inputs = Input(shape=(self.input_dim,))
        x = Dense(hidden_layers[0], activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        for units in hidden_layers[1:]:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_cnn_model(self, filters: List[int] = [32, 64], 
                       kernel_sizes: List[int] = [5, 3],
                       dense_units: List[int] = [64, 32]) -> Model:
        """Build 1D CNN for sequential data"""
        inputs = Input(shape=(self.input_dim, 1))
        
        x = Conv1D(filters[0], kernel_sizes[0], activation='relu', padding='same')(inputs)
        x = MaxPooling1D(2)(x)
        x = BatchNormalization()(x)
        
        for f, k in zip(filters[1:], kernel_sizes[1:]):
            x = Conv1D(f, k, activation='relu', padding='same')(x)
            x = MaxPooling1D(2)(x)
            x = BatchNormalization()(x)
        
        x = Flatten()(x)
        
        for units in dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(0.3)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_lstm_model(self, lstm_units: List[int] = [64, 32],
                        dense_units: List[int] = [32, 16]) -> Model:
        """Build LSTM model for temporal patterns"""
        inputs = Input(shape=(self.input_dim, 1))
        
        x = LSTM(lstm_units[0], return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        
        for units in lstm_units[1:]:
            x = LSTM(units, return_sequences=(units != lstm_units[-1]))(x)
            x = Dropout(0.2)(x)
        
        x = Flatten()(x)
        
        for units in dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(0.3)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_model(self, **kwargs) -> Model:
        """Build the specified model type"""
        if self.model_type == "dense":
            self.model = self.build_dense_model(**kwargs)
        elif self.model_type == "cnn":
            self.model = self.build_cnn_model(**kwargs)
        elif self.model_type == "lstm":
            self.model = self.build_lstm_model(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs: int = 100, batch_size: int = 32, **kwargs) -> Dict:
        """Train the neural network"""
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                         patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        
        # Reshape data for CNN/LSTM if needed
        if self.model_type in ["cnn", "lstm"]:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            **kwargs
        )
        
        return self.history.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model_type in ["cnn", "lstm"]:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model_type in ["cnn", "lstm"]:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        return self.model.evaluate(X_test, y_test, verbose=0)

class ModelComparator:
    """Compare different neural architectures"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model_type: str, **kwargs):
        """Add a model for comparison"""
        model_builder = NeuralHeartModel(self.input_dim, model_type)
        model = model_builder.build_model(**kwargs)
        self.models[name] = model_builder
    
    def compare_models(self, X_train, y_train, X_test, y_test, 
                      epochs: int = 50) -> pd.DataFrame:
        """Compare all models"""
        import pandas as pd
        
        results = []
        
        for name, model_builder in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            history = model_builder.train(X_train, y_train, epochs=epochs)
            
            # Evaluate
            test_loss, test_accuracy, test_auc = model_builder.evaluate(X_test, y_test)
            
            results.append({
                'model': name,
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'test_loss': test_loss,
                'final_val_accuracy': history.get('val_accuracy', [0])[-1],
                'final_val_auc': history.get('val_auc', [0])[-1]
            })
        
        self.results = pd.DataFrame(results)
        return self.results