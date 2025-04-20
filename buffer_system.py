# buffer_system.py

import pandas as pd
import os
from datetime import datetime
import joblib

class PredictionBuffer:
    def __init__(self, buffer_path='buffer.csv', dataset_path='Top80Symptoms_WithNoise.csv', 
                 model_path='disease_model.pkl', buffer_size=50):
        self.buffer_path = buffer_path
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.buffer_size = buffer_size
        
        # Initialize or load existing buffer
        if os.path.exists(buffer_path):
            self.buffer = pd.read_csv(buffer_path)
        else:
            # Create empty buffer with same columns as main dataset
            if os.path.exists(dataset_path):
                main_data = pd.read_csv(dataset_path)
                self.buffer = pd.DataFrame(columns=main_data.columns)
                self.buffer.to_csv(buffer_path, index=False)
            else:
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    def add_entry(self, symptoms_dict, prognosis, feedback_correct=True):
        """
        Add a new entry to the buffer
        symptoms_dict: Dictionary with symptom names as keys and 1/0 as values
        prognosis: The diagnosed disease
        feedback_correct: Whether the prediction was correct (from user feedback)
        """
        # Load current buffer
        self.buffer = pd.read_csv(self.buffer_path)
        
        # If feedback indicates incorrect prediction, don't add to buffer
        if not feedback_correct:
            return False
        
        # Create a new row with all symptoms set to 0
        main_data = pd.read_csv(self.dataset_path)
        new_row = pd.DataFrame(columns=main_data.columns, data=[[0] * len(main_data.columns)])
        
        # Set the provided symptoms to 1
        for symptom, value in symptoms_dict.items():
            if symptom in new_row.columns:
                new_row.loc[0, symptom] = value
        
        # Set the prognosis
        new_row['prognosis'] = prognosis
        
        # Add to buffer
        self.buffer = pd.concat([self.buffer, new_row], ignore_index=True)
        self.buffer.to_csv(self.buffer_path, index=False)
        
        # Check if buffer size exceeds threshold
        if len(self.buffer) >= self.buffer_size:
            return True  # Signal that retraining is needed
        return False
    
    def get_buffer_size(self):
        """Return current buffer size"""
        if os.path.exists(self.buffer_path):
            return len(pd.read_csv(self.buffer_path))
        return 0
    
    def clear_buffer(self):
        """Clear the buffer after retraining"""
        if os.path.exists(self.buffer_path):
            # Save backup before clearing
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"buffer_backup_{timestamp}.csv"
            os.rename(self.buffer_path, backup_path)
            
            # Create new empty buffer
            main_data = pd.read_csv(self.dataset_path)
            empty_buffer = pd.DataFrame(columns=main_data.columns)
            empty_buffer.to_csv(self.buffer_path, index=False)