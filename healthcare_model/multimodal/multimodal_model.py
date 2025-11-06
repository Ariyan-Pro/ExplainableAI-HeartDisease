"""
Multi-Modal Model for ECG + Structured Data Fusion
Combine ECG signals with clinical features
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                   Conv1D, MaxPooling1D, Flatten, LSTM, GRU,
                                   Concatenate, Attention, Multiply, Add)
from tensorflow.keras.optimizers import Adam
from typing import Dict, Tuple, List
import numpy as np

class MultiModalHeartModel:
    """Multi-modal model combining ECG and structured clinical data"""
    
    def __init__(self, structured_input_dim: int, ecg_seq_length: int):
        self.structured_input_dim = structured_input_dim
        self.ecg_seq_length = ecg_seq_length
        self.model = None
    
    def create_early_fusion_model(self, ecg_filters: List[int] = [32, 64],
                                dense_units: List[int] = [128, 64, 32],
                                dropout_rate: float = 0.3) -> Model:
        """
        Create early fusion model - concatenate features at input level
        
        Args:
            ecg_filters: CNN filters for ECG processing
            dense_units: Dense layer units
            dropout_rate: Dropout rate for regularization
        """
        # Structured data input
        structured_input = Input(shape=(self.structured_input_dim,), name='structured_input')
        structured_stream = Dense(dense_units[0], activation='relu')(structured_input)
        structured_stream = BatchNormalization()(structured_stream)
        structured_stream = Dropout(dropout_rate)(structured_stream)
        
        # ECG data input
        ecg_input = Input(shape=(self.ecg_seq_length, 1), name='ecg_input')
        
        # CNN for ECG feature extraction
        ecg_stream = Conv1D(ecg_filters[0], 5, activation='relu', padding='same')(ecg_input)
        ecg_stream = MaxPooling1D(2)(ecg_stream)
        ecg_stream = BatchNormalization()(ecg_stream)
        
        for filters in ecg_filters[1:]:
            ecg_stream = Conv1D(filters, 3, activation='relu', padding='same')(ecg_stream)
            ecg_stream = MaxPooling1D(2)(ecg_stream)
            ecg_stream = BatchNormalization()(ecg_stream)
        
        ecg_stream = Flatten()(ecg_stream)
        ecg_stream = Dense(dense_units[0], activation='relu')(ecg_stream)
        ecg_stream = Dropout(dropout_rate)(ecg_stream)
        
        # Early fusion - concatenate both streams
        fused = Concatenate()([structured_stream, ecg_stream])
        
        # Additional dense layers after fusion
        for units in dense_units[1:]:
            fused = Dense(units, activation='relu')(fused)
            fused = BatchNormalization()(fused)
            fused = Dropout(dropout_rate)(fused)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(fused)
        
        model = Model(inputs=[structured_input, ecg_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def create_late_fusion_model(self, ecg_filters: List[int] = [32, 64],
                               structured_units: List[int] = [64, 32],
                               fusion_units: List[int] = [64, 32],
                               dropout_rate: float = 0.3) -> Model:
        """
        Create late fusion model - combine predictions from separate models
        """
        # Structured data pathway
        structured_input = Input(shape=(self.structured_input_dim,), name='structured_input')
        x_structured = Dense(structured_units[0], activation='relu')(structured_input)
        x_structured = BatchNormalization()(x_structured)
        x_structured = Dropout(dropout_rate)(x_structured)
        
        for units in structured_units[1:]:
            x_structured = Dense(units, activation='relu')(x_structured)
            x_structured = BatchNormalization()(x_structured)
            x_structured = Dropout(dropout_rate)(x_structured)
        
        structured_output = Dense(16, activation='relu', name='structured_features')(x_structured)
        
        # ECG data pathway
        ecg_input = Input(shape=(self.ecg_seq_length, 1), name='ecg_input')
        x_ecg = Conv1D(ecg_filters[0], 5, activation='relu', padding='same')(ecg_input)
        x_ecg = MaxPooling1D(2)(x_ecg)
        x_ecg = BatchNormalization()(x_ecg)
        
        for filters in ecg_filters[1:]:
            x_ecg = Conv1D(filters, 3, activation='relu', padding='same')(x_ecg)
            x_ecg = MaxPooling1D(2)(x_ecg)
            x_ecg = BatchNormalization()(x_ecg)
        
        x_ecg = Flatten()(x_ecg)
        x_ecg = Dense(64, activation='relu')(x_ecg)
        x_ecg = Dropout(dropout_rate)(x_ecg)
        ecg_output = Dense(16, activation='relu', name='ecg_features')(x_ecg)
        
        # Late fusion - combine feature representations
        fused = Concatenate()([structured_output, ecg_output])
        
        for units in fusion_units:
            fused = Dense(units, activation='relu')(fused)
            fused = BatchNormalization()(fused)
            fused = Dropout(dropout_rate)(fused)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(fused)
        
        model = Model(inputs=[structured_input, ecg_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def create_attention_fusion_model(self, ecg_filters: List[int] = [32, 64],
                                   attention_units: int = 32,
                                   dense_units: List[int] = [128, 64, 32],
                                   dropout_rate: float = 0.3) -> Model:
        """
        Create attention-based fusion model
        Uses attention mechanism to weight importance of different modalities
        """
        # Structured data input
        structured_input = Input(shape=(self.structured_input_dim,), name='structured_input')
        structured_features = Dense(dense_units[0], activation='relu')(structured_input)
        structured_features = BatchNormalization()(structured_features)
        structured_features = Dropout(dropout_rate)(structured_features)
        
        # ECG data input with attention
        ecg_input = Input(shape=(self.ecg_seq_length, 1), name='ecg_input')
        
        # Bidirectional LSTM with attention for ECG
        ecg_lstm = LSTM(64, return_sequences=True)(ecg_input)
        ecg_attention = Dense(1, activation='tanh')(ecg_lstm)
        ecg_attention = tf.keras.layers.Flatten()(ecg_attention)
        ecg_attention = tf.keras.layers.Activation('softmax')(ecg_attention)
        ecg_attention = tf.keras.layers.RepeatVector(64)(ecg_attention)
        ecg_attention = tf.keras.layers.Permute([2, 1])(ecg_attention)
        
        ecg_weighted = Multiply()([ecg_lstm, ecg_attention])
        ecg_weighted = LSTM(32)(ecg_weighted)
        
        # Fusion with attention between modalities
        structured_reshaped = tf.keras.layers.RepeatVector(1)(structured_features)
        ecg_reshaped = tf.keras.layers.RepeatVector(1)(ecg_weighted)
        
        # Cross-modal attention
        cross_attention = Attention()([structured_reshaped, ecg_reshaped])
        cross_attention = Flatten()(cross_attention)
        
        # Final dense layers
        for units in dense_units[1:]:
            cross_attention = Dense(units, activation='relu')(cross_attention)
            cross_attention = BatchNormalization()(cross_attention)
            cross_attention = Dropout(dropout_rate)(cross_attention)
        
        output = Dense(1, activation='sigmoid', name='output')(cross_attention)
        
        model = Model(inputs=[structured_input, ecg_input], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        return model
    
    def build_model(self, fusion_type: str = "early", **kwargs) -> Model:
        """Build the specified fusion model"""
        if fusion_type == "early":
            self.model = self.create_early_fusion_model(**kwargs)
        elif fusion_type == "late":
            self.model = self.create_late_fusion_model(**kwargs)
        elif fusion_type == "attention":
            self.model = self.create_attention_fusion_model(**kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        return self.model
    
    def train(self, structured_data: np.ndarray, ecg_data: np.ndarray, 
              labels: np.ndarray, validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 32, **kwargs) -> Dict:
        """Train the multi-modal model"""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        ]
        
        # Reshape ECG data if needed
        if len(ecg_data.shape) == 2:
            ecg_data = ecg_data.reshape(ecg_data.shape[0], ecg_data.shape[1], 1)
        
        history = self.model.fit(
            [structured_data, ecg_data],
            labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            **kwargs
        )
        
        return history.history
    
    def evaluate(self, structured_data: np.ndarray, ecg_data: np.ndarray, 
                 labels: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if len(ecg_data.shape) == 2:
            ecg_data = ecg_data.reshape(ecg_data.shape[0], ecg_data.shape[1], 1)
        
        results = self.model.evaluate([structured_data, ecg_data], labels, verbose=0)
        
        metrics = {}
        for i, metric in enumerate(self.model.metrics_names):
            metrics[metric] = results[i]
        
        return metrics
    
    def predict(self, structured_data: np.ndarray, ecg_data: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if len(ecg_data.shape) == 2:
            ecg_data = ecg_data.reshape(ecg_data.shape[0], ecg_data.shape[1], 1)
        
        return self.model.predict([structured_data, ecg_data])

class MultiModalComparator:
    """Compare different fusion strategies"""
    
    def __init__(self, structured_dim: int, ecg_length: int):
        self.structured_dim = structured_dim
        self.ecg_length = ecg_length
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, fusion_type: str, **kwargs):
        """Add a fusion model for comparison"""
        model_builder = MultiModalHeartModel(self.structured_dim, self.ecg_length)
        model = model_builder.build_model(fusion_type, **kwargs)
        self.models[name] = model_builder
    
    def compare_fusion_strategies(self, structured_data: np.ndarray, 
                                ecg_data: np.ndarray, labels: np.ndarray,
                                epochs: int = 50) -> pd.DataFrame:
        """Compare all fusion strategies"""
        import pandas as pd
        
        results = []
        
        for name, model_builder in self.models.items():
            print(f"Training {name} fusion model...")
            
            # Train model
            history = model_builder.train(structured_data, ecg_data, labels, epochs=epochs)
            
            # Evaluate
            metrics = model_builder.evaluate(structured_data, ecg_data, labels)
            
            results.append({
                'fusion_strategy': name,
                'test_accuracy': metrics.get('accuracy', 0),
                'test_auc': metrics.get('auc', 0),
                'test_precision': metrics.get('precision', 0),
                'test_recall': metrics.get('recall', 0),
                'final_val_accuracy': history.get('val_accuracy', [0])[-1],
                'final_val_auc': history.get('val_auc', [0])[-1]
            })
        
        self.results = pd.DataFrame(results)
        return self.results