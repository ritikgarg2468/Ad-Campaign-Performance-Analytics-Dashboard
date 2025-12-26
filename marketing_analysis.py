#!/usr/bin/env python3
"""
Marketing Campaign Performance Analysis Script

This script performs comprehensive analysis of marketing campaign data including:
- Data loading and cleaning
- Exploratory data analysis
- Customer segmentation
- Predictive modeling for campaign response

Usage:
    python marketing_analysis.py

Requirements:
    - MySQL database with marketing campaign data
    - Required Python packages (see requirements.txt)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector as mysql
from sqlalchemy import create_engine
import urllib.parse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Raj@199807886',  # Consider using environment variables for security
    'database': 'ownprojects'
}

def load_data_from_mysql():
    """Load marketing campaign data from MySQL database."""
    print("Connecting to MySQL database...")
    try:
        db_connection = mysql.connect(**DB_CONFIG)
        query = "SELECT * FROM marketing_campaign;"
        df = pd.read_sql(query, con=db_connection)
        db_connection.close()
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean and preprocess the data."""
    print("Cleaning data...")

    # Convert date column
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')

    # Handle missing values (basic approach - can be improved)
    df = df.dropna(subset=['Income'])  # Remove rows with missing income

    # Calculate age if Year_Birth exists
    if 'Year_Birth' in df.columns:
        current_year = 2025
        df['Age'] = current_year - df['Year_Birth']

    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows. Removing...")
        df = df.drop_duplicates()

    print(f"Data cleaned. Final shape: {df.shape}")
    return df

def exploratory_analysis(df):
    """Perform exploratory data analysis."""
    print("Performing exploratory data analysis...")

    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

def create_visualizations(df):
    """Create data visualizations."""
    print("Creating visualizations...")

    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')

    # Income distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Income'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Customer Income')
    plt.savefig('income_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Age distribution
    if 'Age' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Age'], bins=20, kde=True)
        plt.title('Distribution of Customer Age')
        plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Campaign response rates
    campaign_cols = [col for col in df.columns if col.startswith('AcceptedCmp')]
    if campaign_cols:
        response_rates = df[campaign_cols].mean() * 100
        plt.figure(figsize=(12, 6))
        response_rates.plot(kind='bar')
        plt.title('Campaign Acceptance Rates')
        plt.xlabel('Campaign')
        plt.ylabel('Acceptance Rate (%)')
        plt.xticks(rotation=45)
        plt.savefig('campaign_response_rates.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("Visualizations saved as PNG files.")

def customer_segmentation(df):
    """Perform customer segmentation using K-means clustering."""
    print("Performing customer segmentation...")

    # Select features
    features = ['Income', 'Age', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

    # Prepare data
    seg_data = df[features].dropna()

    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(seg_data)

    # Optimal k (using elbow method approximation)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    seg_data['Cluster'] = clusters

    # Cluster summary
    cluster_summary = seg_data.groupby('Cluster').mean()
    print("\nCluster Summary:")
    print(cluster_summary)

    # Save cluster summary
    cluster_summary.to_csv('customer_segments.csv')

    # PCA visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Customer Segments (PCA Projection)')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('customer_segments_pca.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Customer segmentation completed. Results saved.")

def predictive_modeling(df):
    """Build predictive models for campaign response."""
    print("Building predictive models...")

    # Prepare target variable
    campaign_cols = [col for col in df.columns if col.startswith('AcceptedCmp')]
    df['AcceptedAnyCampaign'] = df[campaign_cols].any(axis=1).astype(int)

    # Select features
    feature_cols = ['Income', 'Age', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

    # Handle categorical variables
    categorical_cols = ['Education', 'Marital_Status']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Prepare data
    model_features = feature_cols + categorical_cols
    model_data = df[model_features + ['AcceptedAnyCampaign']].dropna()
    X = model_data[model_features]
    y = model_data['AcceptedAnyCampaign']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # Results
    print("\nRandom Forest Model Results:")
    print(classification_report(y_test, rf_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_pred_proba):.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Predictive modeling completed. Results saved.")

def save_cleaned_data(df):
    """Save cleaned data back to MySQL."""
    print("Saving cleaned data to MySQL...")

    try:
        # Create connection
        encoded_password = urllib.parse.quote_plus(DB_CONFIG['password'])
        engine = create_engine(f'mysql+mysqlconnector://{DB_CONFIG["user"]}:{encoded_password}@{DB_CONFIG["host"]}/{DB_CONFIG["database"]}')

        # Upload to MySQL
        df.to_sql(name='marketing_campaign_cleaned', con=engine, if_exists='replace', index=False)
        print("Cleaned data saved to MySQL table 'marketing_campaign_cleaned'.")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    """Main execution function."""
    print("Starting Marketing Campaign Performance Analysis...")

    # Load data
    df = load_data_from_mysql()
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Clean data
    df = clean_data(df)

    # Exploratory analysis
    exploratory_analysis(df)

    # Visualizations
    create_visualizations(df)

    # Customer segmentation
    customer_segmentation(df)

    # Predictive modeling
    predictive_modeling(df)

    # Save cleaned data
    save_cleaned_data(df)

    print("\nAnalysis completed! Check the generated files and visualizations.")

if __name__ == "__main__":
    main()