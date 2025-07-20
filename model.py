# task3_end_to_end_project/model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# FIXED: ModelWithScaler class moved to TOP LEVEL (outside any function)
class ModelWithScaler:
    """
    Wrapper class that includes both the trained model and scaler for predictions
    """
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def predict(self, X):
        # Ensure X has the same features as training data
        if isinstance(X, pd.DataFrame):
            # Add missing engineered features if needed
            if 'house_age' not in X.columns and 'yr_built' in X.columns:
                X = X.copy()
                X['house_age'] = 2023 - X['yr_built']
                X['renovated'] = (X['yr_renovated'] > 0).astype(int)
                X['rooms_per_floor'] = X['bedrooms'] / X['floors']
                X['bathrooms_per_bedroom'] = X['bathrooms'] / X['bedrooms']
                X['sqft_living_per_floor'] = X['sqft_living'] / X['floors']
                X['log_sqft_living'] = np.log1p(X['sqft_living'])
                X['log_sqft_lot'] = np.log1p(X['sqft_lot'])
            
            # Select only training features
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class HousePricePredictor:
    """
    Complete house price prediction model with data preprocessing and training
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_sample_data(self, n_samples=5000):
        """
        Generate sample house data for demonstration
        """
        np.random.seed(42)
        
        data = {
            'bedrooms': np.random.randint(1, 8, n_samples),
            'bathrooms': np.random.uniform(1, 5, n_samples).round(1),
            'sqft_living': np.random.randint(500, 8000, n_samples),
            'sqft_lot': np.random.randint(1000, 50000, n_samples),
            'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, 
                                    p=[0.3, 0.2, 0.3, 0.15, 0.05]),
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'view': np.random.randint(0, 5, n_samples),
            'condition': np.random.randint(1, 6, n_samples),
            'grade': np.random.randint(4, 13, n_samples),
            'yr_built': np.random.randint(1900, 2023, n_samples),
            'yr_renovated': np.random.choice(
                np.concatenate([[0], np.random.randint(1990, 2023, 500)]), 
                n_samples, p=np.concatenate([[0.8], [0.0004] * 500])
            ),
            'zipcode': np.random.choice(range(98001, 98200), n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic price based on features
        price = (
            df['bedrooms'] * 15000 +
            df['bathrooms'] * 12000 +
            df['sqft_living'] * 100 +
            df['sqft_lot'] * 2 +
            df['floors'] * 10000 +
            df['waterfront'] * 200000 +
            df['view'] * 20000 +
            df['condition'] * 8000 +
            df['grade'] * 15000 +
            (2023 - df['yr_built']) * -500 +
            (df['yr_renovated'] > 0) * 30000 +
            np.random.normal(0, 50000, n_samples)  # Add noise
        )
        
        # Ensure positive prices
        df['price'] = np.maximum(price, 50000)
        
        return df
    
    def preprocess_data(self, df, target_column='price'):
        """
        Preprocess the data for training
        """
        # Make a copy
        processed_df = df.copy()
        
        # Handle missing values
        processed_df.fillna(processed_df.median(numeric_only=True), inplace=True)
        
        # Feature engineering
        processed_df['house_age'] = 2023 - processed_df['yr_built']
        processed_df['renovated'] = (processed_df['yr_renovated'] > 0).astype(int)
        processed_df['rooms_per_floor'] = processed_df['bedrooms'] / processed_df['floors']
        processed_df['bathrooms_per_bedroom'] = processed_df['bathrooms'] / processed_df['bedrooms']
        processed_df['sqft_living_per_floor'] = processed_df['sqft_living'] / processed_df['floors']
        
        # Log transformations for skewed features
        processed_df['log_sqft_living'] = np.log1p(processed_df['sqft_living'])
        processed_df['log_sqft_lot'] = np.log1p(processed_df['sqft_lot'])
        
        # Remove outliers (IQR method)
        Q1 = processed_df[target_column].quantile(0.25)
        Q3 = processed_df[target_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        processed_df = processed_df[
            (processed_df[target_column] >= lower_bound) &
            (processed_df[target_column] <= upper_bound)
        ]
        
        self.logger.info(f"Data preprocessed. Final shape: {processed_df.shape}")
        return processed_df
    
    def train_model(self, df=None):
        """
        Train the house price prediction model
        """
        if df is None:
            self.logger.info("No data provided. Generating sample data...")
            df = self.generate_sample_data()
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Separate features and target
        target_column = 'price'
        X = processed_df.drop(target_column, axis=1)
        y = processed_df[target_column]
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select the best
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'Linear Regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('-inf')
        best_name = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=5, scoring='r2')
            mean_score = cv_scores.mean()
            
            self.logger.info(f"{name} CV R² Score: {mean_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        
        # Train the best model
        self.logger.info(f"Training best model: {best_name}")
        best_model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"Model Performance:")
        self.logger.info(f"RMSE: ${rmse:,.2f}")
        self.logger.info(f"MAE: ${mae:,.2f}")
        self.logger.info(f"R² Score: {r2:.4f}")
        
        # FIXED: Use the top-level ModelWithScaler class
        self.model = ModelWithScaler(best_model, self.scaler, self.feature_names)
        
        # Save the model
        with open('house_price_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        self.logger.info("Model saved as 'house_price_model.pkl'")
        
        # Visualize results
        self.visualize_results(y_test, y_pred)
        
        return self.model
    
    def visualize_results(self, y_test, y_pred):
        """
        Create visualizations of model performance
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Actual vs Predicted
            axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
            axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Price')
            axes[0, 0].set_ylabel('Predicted Price')
            axes[0, 0].set_title('Actual vs Predicted Prices')
            
            # Residuals
            residuals = y_test - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Price')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residual Plot')
            
            # Distribution of residuals
            axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Residuals')
            
            # Feature importance (if available)
            if hasattr(self.model.model, 'feature_importances_'):
                importances = self.model.model.feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(10)
                
                axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'])
                axes[1, 1].set_yticks(range(len(feature_importance)))
                axes[1, 1].set_yticklabels(feature_importance['feature'])
                axes[1, 1].set_xlabel('Feature Importance')
                axes[1, 1].set_title('Top 10 Feature Importances')
            
            plt.tight_layout()
            plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            self.logger.warning(f"Could not create visualizations: {str(e)}")

# Example usage
if __name__ == "__main__":
    predictor = HousePricePredictor()
    model = predictor.train_model()
    
    # Test prediction
    sample_house = pd.DataFrame([{
        'bedrooms': 3,
        'bathrooms': 2.5,
        'sqft_living': 2500,
        'sqft_lot': 7500,
        'floors': 2,
        'waterfront': 0,
        'view': 2,
        'condition': 4,
        'grade': 8,
        'yr_built': 1995,
        'yr_renovated': 0,
        'zipcode': 98105
    }])
    
    predicted_price = model.predict(sample_house)[0]
    print(f"Predicted house price: ${predicted_price:,.2f}")