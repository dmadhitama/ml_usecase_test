import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import argparse
from loguru import logger
import sys
import os

# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
# check if log directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")
logger.add("logs/retail_analysis_{time}.log", rotation="500 MB")

# Load and preprocess data
def load_and_preprocess_data(file_path):
    logger.info(f"Loading data from {file_path}")
    try:
        # Read the data
        df = pd.read_excel(file_path)
        logger.info(f"Successfully loaded {len(df)} rows of data")
        
        # Debug: Print column names and their types
        logger.info("DataFrame columns:")
        for col in df.columns:
            logger.info(f"Column: '{col}' - Type: {df[col].dtype}")
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Remove rows with missing CustomerID
        initial_rows = len(df)
        # Debug: Check if CustomerID exists with case-insensitive match
        customer_id_col = next((col for col in df.columns if col.lower() == 'customerid'), None)
        if customer_id_col:
            logger.info(f"Found CustomerID column as: {customer_id_col}")
            # Rename to standard format if needed
            if customer_id_col != 'CustomerID':
                df = df.rename(columns={customer_id_col: 'CustomerID'})
                logger.info("Renamed column to 'CustomerID'")
        else:
            logger.error("CustomerID column not found in dataset")
            logger.info(f"Available columns: {', '.join(df.columns)}")
            raise ValueError("CustomerID column not found in dataset")
            
        df = df.dropna(subset=['CustomerID'])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing CustomerID")
        
        # Remove cancelled orders (those starting with 'C')
        initial_rows = len(df)
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        logger.info(f"Removed {initial_rows - len(df)} cancelled orders")
        
        # Calculate total amount for each transaction
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
        logger.success("Data preprocessing completed successfully")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def calculate_rfm_metrics(df):
    logger.info("Calculating RFM metrics")
    try:
        # Calculate the most recent date in the dataset
        max_date = df['InvoiceDate'].max()
        
        # Group by customer and calculate RFM metrics
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
            'InvoiceNo': 'count',  # Frequency
            'TotalAmount': 'sum'  # Monetary
        })
        
        # Rename columns
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Reset index to make CustomerID a column again
        rfm = rfm.reset_index()
        
        logger.info(f"RFM metrics calculated for {len(rfm)} customers")
        
        return rfm
    except Exception as e:
        logger.error(f"Error calculating RFM metrics: {str(e)}")
        raise

def perform_customer_segmentation(rfm, n_clusters=4):
    logger.info(f"Performing customer segmentation with {n_clusters} clusters")
    try:
        # Scale the RFM metrics
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(rfm_scaled, rfm['Cluster'])
        logger.info(f"Clustering completed with silhouette score: {silhouette_avg:.3f}")
        
        return rfm, silhouette_avg
    except Exception as e:
        logger.error(f"Error performing customer segmentation: {str(e)}")
        raise

def analyze_clusters(rfm):
    logger.info("Analyzing cluster characteristics")
    try:
        # Calculate cluster characteristics
        cluster_analysis = rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        
        # Add customer count per cluster
        cluster_analysis['Number of Customers'] = rfm.groupby('Cluster').size()
        
        cluster_analysis.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary Value', 'Number of Customers']
        
        return cluster_analysis
    except Exception as e:
        logger.error(f"Error analyzing clusters: {str(e)}")
        raise

def plot_cluster_characteristics(rfm, output_dir):
    logger.info("Creating cluster visualization plots")
    try:
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Recency vs Frequency
        plt.subplot(1, 3, 1)
        plt.scatter(rfm['Recency'], rfm['Frequency'], c=rfm['Cluster'], cmap='viridis')
        plt.xlabel('Recency')
        plt.ylabel('Frequency')
        plt.title('Recency vs Frequency by Cluster')
        
        # Plot 2: Recency vs Monetary
        plt.subplot(1, 3, 2)
        plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], cmap='viridis')
        plt.xlabel('Recency')
        plt.ylabel('Monetary')
        plt.title('Recency vs Monetary by Cluster')
        
        # Plot 3: Frequency vs Monetary
        plt.subplot(1, 3, 3)
        plt.scatter(rfm['Frequency'], rfm['Monetary'], c=rfm['Cluster'], cmap='viridis')
        plt.xlabel('Frequency')
        plt.ylabel('Monetary')
        plt.title('Frequency vs Monetary by Cluster')
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'cluster_analysis.png')
        plt.savefig(output_path)
        plt.close()
        logger.success(f"Plots saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(description='Retail Customer Segmentation Analysis')
    parser.add_argument('--input', '-i', required=True, help='Path to the input CSV file')
    parser.add_argument('--output', '-o', default='output', help='Directory to save output files (default: output)')
    parser.add_argument('--clusters', '-c', type=int, default=4, help='Number of clusters for segmentation (default: 4)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("Starting retail customer segmentation analysis")
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(args.input)
        
        # Calculate RFM metrics
        rfm = calculate_rfm_metrics(df)
        
        # Perform customer segmentation
        rfm, silhouette_score = perform_customer_segmentation(rfm, args.clusters)
        
        # Analyze clusters
        cluster_analysis = analyze_clusters(rfm)
        
        # Print results
        logger.info("\nCluster Analysis:")
        print(cluster_analysis)
        logger.info(f"\nSilhouette Score: {silhouette_score:.3f}")
        
        # Create visualizations
        plot_cluster_characteristics(rfm, args.output)
        
        logger.success("Analysis completed successfully!")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
