# online_learning.py

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='online_learning.log'
)
logger = logging.getLogger('online_learning')

# File paths
DATA_PATH = 'Top80Symptoms_WithNoise.csv'
MODEL_PATH = 'disease_model.pkl'
BUFFER_PATH = 'buffer.csv'

class OnlineLearning:
    
    def __init__(self, data_path=DATA_PATH, model_path=MODEL_PATH, buffer_path=BUFFER_PATH):
        self.data_path = data_path
        self.model_path = model_path
        self.buffer_path = buffer_path
        self.label_encoder = LabelEncoder()
        
        # Load model if it exists
        self.model, self.columns, self.label_encoder = self._load_or_create_model()
    def plot_model_accuracies(self,accuracies):
        names = list(accuracies.keys())
        scores = list(accuracies.values())
        plt.figure(figsize=(10, 6))
        sns.barplot(x=scores, y=names, palette='viridis')
        plt.xlabel("Accuracy")
        plt.title("Classifier Accuracies")
        plt.xlim(0, 1)
        plt.grid(True, axis='x', linestyle='--')
        plt.tight_layout()
        plt.savefig("accuracies", dpi=300, bbox_inches='tight')
        plt.show()
        
    def _load_or_create_model(self):
        """Load existing model or create new one if none exists"""
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing model from {self.model_path}")
            return joblib.load(self.model_path)
        else:
            logger.info("No existing model found. Will train new model.")
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
            
            # Train new model
            df = pd.read_csv(self.data_path)
            voting_clf, columns, label_encoder = self._train_model(df)
            return voting_clf, columns, label_encoder
    

    def _train_model(self, df):
        """Train a new ensemble model from scratch"""
        logger.info("Training new model from scratch")
        
        X = df.drop('prognosis', axis=1).fillna('')
        X = pd.get_dummies(X)
        y = self.label_encoder.fit_transform(df['prognosis'])
        
        # Define base classifiers suitable for online learning
        clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
        clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
        clf3 = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)  # Online-friendly
        clf4 = SVC(probability=True)
        clf5 = GaussianNB()  # Good for incremental learning
        clf6 = DecisionTreeClassifier(random_state=42)
        clf7 = KNeighborsClassifier(n_neighbors=5)
        clf8 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        
        # Create Voting Classifier
        voting_clf = VotingClassifier(estimators=[
            ('rf', clf1),
            ('gb', clf2),
            ('sgd', clf3),
            ('svc', clf4),
            ('gnb', clf5),
            ('dt', clf6),
            ('knn', clf7),
            ('xgb', clf8)
        ], voting='soft')
        
        # Train the ensemble model
        voting_clf.fit(X, y)
        
        return voting_clf, X.columns, self.label_encoder
    
    def update_model(self):
        """Update the model with new data from buffer"""
        if not os.path.exists(self.buffer_path):
            logger.warning(f"Buffer file not found: {self.buffer_path}")
            return False
        
        buffer_df = pd.read_csv(self.buffer_path)
        if len(buffer_df) == 0:
            logger.info("Buffer is empty. No update needed.")
            return False
        
        logger.info(f"Updating model with {len(buffer_df)} new examples from buffer")
        
        # Merge buffer with original dataset
        original_df = pd.read_csv(self.data_path)
        updated_df = pd.concat([original_df, buffer_df], ignore_index=True)
        updated_df.to_csv(self.data_path, index=False)
        
        # Retrain model with updated dataset
        self.model, self.columns, self.label_encoder = self._train_model(updated_df)
        
        # Save the updated model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"disease_model_backup_{timestamp}.pkl"
        if os.path.exists(self.model_path):
            os.rename(self.model_path, backup_path)
        
        joblib.dump((self.model, self.columns, self.label_encoder), self.model_path)
        logger.info(f"Model updated and saved. Previous model backed up to {backup_path}")
        
        return True
    
    def add_new_symptom(self, symptom_name):
        """Add a new symptom column to the dataset"""
        if not symptom_name or not isinstance(symptom_name, str):
            logger.error("Invalid symptom name")
            return False
        
        # Clean symptom name (lowercase, underscores for spaces)
        symptom_name = symptom_name.lower().strip().replace(' ', '_')
        
        # Load dataset
        df = pd.read_csv(self.data_path)
        
        # Check if symptom already exists
        if symptom_name in df.columns:
            logger.warning(f"Symptom '{symptom_name}' already exists in the dataset")
            return False
        
        # Add new symptom column with all zeros
        df[symptom_name] = 0
        
        # Save updated dataset
        df.to_csv(self.data_path, index=False)
        logger.info(f"Added new symptom '{symptom_name}' to dataset")
        
        # Update buffer with new column
        if os.path.exists(self.buffer_path):
            buffer_df = pd.read_csv(self.buffer_path)
            buffer_df[symptom_name] = 0
            buffer_df.to_csv(self.buffer_path, index=False)
        
        # Retrain model to include new feature
        self.model, self.columns, self.label_encoder = self._train_model(df)
        joblib.dump((self.model, self.columns, self.label_encoder), self.model_path)
        logger.info(f"Model retrained with new symptom '{symptom_name}'")
        
        return True