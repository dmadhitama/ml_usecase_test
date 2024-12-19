# Customer Analytics and Segmentation Project

This project provides advanced analytics tools for retail and wholesale customer data analysis, featuring both traditional clustering-based segmentation and deep learning approaches.

## Components

The project consists of two main analysis modules:

### 1. Retail Analysis (`retail_analysis.py`)
- Implements RFM (Recency, Frequency, Monetary) analysis
- Performs customer segmentation using K-means clustering
- Generates visualizations of customer segments
- Includes detailed logging and error handling

### 2. Wholesale Analysis (`wholesale_analysis.py`)
- Uses deep learning (PyTorch) for customer classification
- Implements a neural network model for customer segmentation
- Includes data preprocessing and robust scaling
- Provides detailed performance metrics and visualizations

## Requirements

The project requires Python 3.x and the following dependencies:
```
pandas>=1.3.0
numpy<2.0.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
loguru>=0.7.0
argparse>=1.4.0
torch>=2.1.0
torchmetrics>=1.2.0
```

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Retail Analysis
Run the retail analysis with:
```bash
python retail_analysis.py --input <path_to_retail_data.csv> --output <output_directory> --clusters <number_of_clusters>
```

Parameters:
- `--input`, `-i` (required): Path to the input Excel file
  - Format: Excel file (.xlsx, .xls)
  - Required columns:
    - `CustomerID`: Unique identifier for each customer
    - `InvoiceNo`: Invoice number for each transaction
    - `InvoiceDate`: Date of the transaction
    - `Quantity`: Number of items purchased
    - `UnitPrice`: Price per unit
  - Example: `--input data/retail_data.xlsx`

- `--output`, `-o` (optional): Directory to save output files
  - Default: 'output'
  - Creates directory if it doesn't exist
  - Outputs:
    - Cluster visualization plots (.png)
    - Cluster analysis results (.csv)
    - Log files
  - Example: `--output results/retail_analysis`

- `--clusters`, `-c` (optional): Number of clusters for customer segmentation
  - Default: 4
  - Recommended range: 3-8
  - Example: `--clusters 5`

Example command:
```bash
python retail_analysis.py --input data/retail_data.xlsx --output results/retail --clusters 5
```

### Wholesale Analysis
Run the wholesale analysis with:
```bash
python wholesale_analysis.py --input <path_to_wholesale_data.csv> --output <output_directory> --epochs <number_of_epochs> --batch_size <batch_size> --learning_rate <learning_rate>
```

Parameters:
- `--input`, `-i` (required): Path to the input CSV file
  - Format: CSV file (.csv)
  - Required columns:
    - `Region`: Target variable for classification (1-based indexing)
    - Feature columns (numeric): Channel, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
  - Example: `--input data/wholesale_data.csv`

- `--output`, `-o` (optional): Directory for saving results
  - Default: 'wholesale_analysis'
  - Creates directory if it doesn't exist
  - Outputs:
    - Model checkpoints
    - Training plots
    - Confusion matrix
    - Performance metrics
  - Example: `--output results/wholesale_analysis`

- `--epochs` (optional): Number of training epochs
  - Default: 100
  - Recommended range: 50-200
  - Example: `--epochs 150`

- `--batch_size` (optional): Mini-batch size for training
  - Default: 32
  - Recommended range: 16-128
  - Example: `--batch_size 64`

- `--learning_rate` (optional): Learning rate for optimizer
  - Default: 0.001
  - Recommended range: 0.0001-0.01
  - Example: `--learning_rate 0.0005`

Example command:
```bash
python wholesale_analysis.py --input data/wholesale_data.csv --output results/wholesale --epochs 150 --batch_size 64 --learning_rate 0.0005
```

## Features

### Retail Analysis
- RFM metric calculation
- Customer segmentation using K-means
- Silhouette score analysis
- Cluster visualization
- Detailed logging of analysis steps

### Wholesale Analysis
- Neural network-based classification
- Robust data scaling
- Batch normalization and dropout for better model performance
- Confusion matrix and accuracy metrics
- Training progress visualization

## Output

Both analyses generate:
- Detailed logs in the `logs` directory
- Visualizations in the specified output directory
- Analysis results and metrics

## Project Structure
```
.
├── README.md
├── requirements.txt
├── retail_analysis.py
├── wholesale_analysis.py
├── logs/
└── output/
```

## Logging

The project uses the Loguru library for comprehensive logging. Logs are stored in:
- `logs/retail_analysis_{time}.log`
- `logs/wholesale_analysis_{time}.log`

## Error Handling

Both modules include robust error handling and input validation to ensure reliable operation and meaningful error messages.

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
