import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           roc_auc_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Fraud Detection Model Trainer",
    page_icon="💰",
    layout="wide"
)

@st.cache_data
def load_and_sample_data(file_path, sample_size=100000, random_state=42):
    """
    Smart data loading with sampling for large datasets
    """
    st.info(f"Loading and sampling data... (Using {sample_size:,} samples from 8.9M rows)")
    
    # Option 1: Load with chunks and sample
    try:
        # Get total rows for progress
        total_rows = 8900000  # You can calculate this dynamically

        file_path = r'notebooks\processeddataset\final_feature_paySim.csv'
        df = pd.read_csv(file_path)
        feature_columns = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'hour', 'is_weekend', 'high_risk_hour', 'high_risk_type', 'large_amount_flag',
        'zero_balance_orig', 'zero_balance_dest', 'balance_ratio_orig', 'balance_ratio_dest',
        'cust_avg_amt', 'cust_std_amt', 'cust_txn_count'
        ]
        X = df[feature_columns]
        y = df['isFraud']
        
        return df, feature_columns
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

class EfficientModelTrainer:
    def __init__(self):
        self.sample_sizes = {
            'quick': 50000,
            'balanced': 200000,
            'large': 500000
        }
    
    def train_isolation_forest(self, X_train, y_train, params):
        """Train Isolation Forest with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Training Isolation Forest...")
        
        model = IsolationForest(
            n_estimators=params['n_estimators'],
            contamination=params['contamination'],
            max_samples=params['max_samples'],
            max_features=params['max_features'],
            random_state=42,
            n_jobs=-1,  # Use all available cores
            verbose=0
        )
        
        # Train model
        model.fit(X_train)
        progress_bar.progress(100)
        status_text.text("Training completed!")
        
        return model
    
    def train_lightgbm(self, X_train, y_train, params):
        """Train LightGBM with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Training LightGBM...")
        
        # LightGBM parameters
        lgb_params = {
            'n_estimators': params['n_estimators'],
            'learning_rate': 0.1,
            'max_depth': -1,  # No limit
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Train model
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data],
            callbacks=[
                lgb.log_evaluation(0),  # No output
            ]
        )
        
        progress_bar.progress(100)
        status_text.text("Training completed!")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_type):
        """Evaluate model performance"""
        if model_type == "Isolation Forest":
            # Predict anomalies
            y_pred = model.predict(X_test)
            y_scores = model.decision_function(X_test)
            y_pred_binary = (y_pred == -1).astype(int)
        else:  # LightGBM
            y_scores = model.predict(X_test)
            y_pred_binary = (y_scores > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        roc_auc = roc_auc_score(y_test, y_scores)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred_binary,
            'scores': y_scores
        }

def main():
    st.title("💰 Fraud Detection Model Trainer")
    st.markdown("Train and compare Isolation Forest vs LightGBM on large-scale data")
    
    # Initialize trainer
    trainer = EfficientModelTrainer()
    
    # Sidebar - Model Selection
    st.sidebar.header("🔧 Model Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Isolation Forest", "LightGBM"],
        help="Choose between unsupervised (Isolation Forest) or supervised (LightGBM) approach"
    )
    
    # Sample size selection
    sample_size_option = st.sidebar.selectbox(
        "Training Data Size",
        ["quick (50K samples)", "balanced (200K samples)", "large (500K samples)"],
        index=1,
        help="Larger samples = better accuracy but longer training time"
    )
    
    # Map selection to actual size
    sample_size = trainer.sample_sizes[sample_size_option.split(' ')[0]]
    
    # Model-specific parameters
    st.sidebar.subheader(f"{model_choice} Parameters")
    
    if model_choice == "Isolation Forest":
        n_estimators = st.sidebar.slider(
            "n_estimators", 
            min_value=50, 
            max_value=500, 
            value=100,
            help="Number of base estimators in the ensemble"
        )
        
        contamination = st.sidebar.slider(
            "contamination", 
            min_value=0.001, 
            max_value=0.1, 
            value=0.02,
            step=0.001,
            format="%.3f",
            help="Expected proportion of outliers in the data"
        )
        
        max_samples = st.sidebar.slider(
            "max_samples", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5,
            step=0.1,
            help="Fraction of samples to draw for each base estimator"
        )
        
        max_features = st.sidebar.slider(
            "max_features", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5,
            step=0.1,
            help="Fraction of features to draw for each base estimator"
        )
        
        model_params = {
            'n_estimators': n_estimators,
            'contamination': contamination,
            'max_samples': max_samples,
            'max_features': max_features
        }
        
    else:  # LightGBM
        n_estimators = st.sidebar.slider(
            "n_estimators", 
            min_value=50, 
            max_value=500, 
            value=100,
            help="Number of boosting iterations"
        )
        
        learning_rate = st.sidebar.slider(
            "learning_rate",
            min_value=0.01,
            max_value=0.2,
            value=0.1,
            step=0.01,
            help="Shrinks the contribution of each tree"
        )
        
        num_leaves = st.sidebar.slider(
            "num_leaves",
            min_value=20,
            max_value=100,
            value=31,
            help="Maximum number of leaves in one tree"
        )
        
        model_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves
        }
    
    # Load data
    df, feature_names = load_and_sample_data(r'notebooks\processeddataset\final_feature_paySim.csv', sample_size=sample_size) # type: ignore
    
    if df is not None:
        # Prepare features and target
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Features", len(feature_names)) # type: ignore
        with col3:
            fraud_rate = y.mean()
            st.metric("Fraud Rate", f"{fraud_rate:.2%}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Training button
        if st.sidebar.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training {model_choice} with {sample_size:,} samples..."):
                start_time = time.time()
                
                # Train model
                if model_choice == "Isolation Forest":
                    model = trainer.train_isolation_forest(X_train, y_train, model_params)
                else:
                    model = trainer.train_lightgbm(X_train, y_train, model_params)
                
                training_time = time.time() - start_time
                
                # Evaluate model
                results = trainer.evaluate_model(model, X_test, y_test, model_choice)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.results = results
                st.session_state.model_type = model_choice
                st.session_state.training_time = training_time
                st.session_state.params = model_params
        
        # Display results if available
        if 'results' in st.session_state:
            results = st.session_state.results
            model_type = st.session_state.model_type
            
            st.header("📊 Model Performance")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
                st.metric("Training Time", f"{st.session_state.training_time:.2f}s")
            
            with col2:
                st.metric("Precision", f"{results['precision']:.4f}")
                st.metric("Recall", f"{results['recall']:.4f}")
            
            with col3:
                st.metric("F1-Score", f"{results['f1_score']:.4f}")
                st.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
            
            with col4:
                cm = results['confusion_matrix']
                tn, fp, fn, tp = cm.ravel()
                st.metric("False Positives", fp)
                st.metric("False Negatives", fn)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{model_type} - Confusion Matrix')
            st.pyplot(fig)
            
            # Model Parameters Used
            st.subheader("⚙️ Model Parameters")
            params_df = pd.DataFrame.from_dict(st.session_state.params, orient='index', columns=['Value'])
            st.dataframe(params_df)
            
            # Performance Insights
            st.subheader("💡 Performance Insights")
            
            if results['recall'] > 0.9:
                st.success("**Excellent Fraud Detection**: Model is highly effective at identifying fraudulent transactions")
            elif results['recall'] > 0.7:
                st.warning("**Good Fraud Detection**: Model captures most fraud cases but has room for improvement")
            else:
                st.error("**Poor Fraud Detection**: Model misses significant fraudulent activity")
            
            if results['precision'] > 0.9:
                st.success("**Low False Positives**: Minimal impact on legitimate customers")
            elif results['precision'] > 0.7:
                st.warning("**Moderate False Positives**: Some legitimate transactions flagged as fraud")
            else:
                st.error("**High False Positives**: Significant impact on customer experience")
    
    else:
        st.error("Failed to load data. Please check your data file.")

if __name__ == "__main__":
    main()