# Marketing Campaign Performance Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data](#data)
- [Analysis](#analysis)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Welcome to the Marketing Campaign Performance Analysis project! This repository contains a comprehensive data analysis of marketing campaigns aimed at evaluating their effectiveness, understanding customer engagement, and deriving actionable insights. As a data scientist, I've leveraged Python and various data analysis libraries to process, clean, and explore marketing campaign data, focusing on key metrics such as response rates, acceptance rates, and customer behavior trends.

The project demonstrates end-to-end data science workflow from data ingestion and cleaning to exploratory data analysis, providing a foundation for data-driven marketing strategies.

## Features

- **Data Ingestion**: Load marketing campaign data from MySQL database or CSV files.
- **Data Cleaning**: Handle missing values, data type conversions, and duplicate removal.
- **Exploratory Data Analysis (EDA)**: Comprehensive statistical summaries and data profiling.
- **Data Visualization**: Create insightful charts and graphs to represent campaign performance, including income/age distributions, campaign response rates, and correlation heatmaps.
- **Customer Segmentation**: Unsupervised clustering using K-means to identify distinct customer groups.
- **Predictive Modeling**: Machine learning models (Logistic Regression and Random Forest) to predict campaign response likelihood.
- **Database Integration**: Seamless upload of cleaned data back to MySQL for further use.
- **Script Version**: Standalone Python script for automated analysis execution.
- **Reproducible Analysis**: Jupyter Notebook-based workflow for easy replication and modification.

## Project Structure

```
marketing_campaign_insight_analysis/
│
├── main.ipynb                    # Main Jupyter Notebook containing the complete analysis
├── marketing_analysis.py         # Standalone Python script for automated analysis
├── marketing_campaign.csv        # Raw marketing campaign dataset (if available)
├── requirements.txt              # Python dependencies
├── config.example.ini            # Database configuration template
├── README.md                     # Project documentation
├── LICENSE                       # Project license
├── .gitignore                    # Git ignore file
├── income_distribution.png       # Generated visualization (after running)
├── age_distribution.png          # Generated visualization (after running)
├── campaign_response_rates.png   # Generated visualization (after running)
├── customer_segments_pca.png     # Generated visualization (after running)
├── feature_importance.png        # Generated visualization (after running)
├── customer_segments.csv         # Customer segmentation results (after running)
└── ...
```

## Getting Started

### Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.8 or higher**: Download from [python.org](https://www.python.org/)
- **MySQL Database**: A running MySQL server with the marketing campaign data loaded
- **Jupyter Notebook**: For running the analysis notebook
- **Git**: For cloning the repository (optional)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/marketing-campaign-insight-analysis.git
   cd marketing-campaign-insight-analysis
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Database** (Optional but recommended):
   - Copy `config.example.ini` to `config.ini`
   - Update the database credentials in `config.ini`
   - The script will use these credentials instead of hardcoded values

## Data

The dataset used in this project contains comprehensive information about customer responses to various marketing campaigns. Key attributes include:

- Customer demographics (age, education, marital status, income, etc.)
- Campaign response data (acceptance rates, response timestamps)
- Purchase history and spending patterns
- Campaign-specific metrics

**Data Source**: The data is loaded from a MySQL database table `marketing_campaign`. A CSV version (`marketing_campaign.csv`) is also included for reference or alternative loading.

**Data Size**: [Add approximate number of records and features after running the notebook]

## Analysis

The analysis follows a comprehensive data science workflow:

1. **Data Loading**: Connect to MySQL database and retrieve campaign data using pandas
2. **Data Profiling**: Examine data types, missing values, and basic statistics
3. **Data Cleaning**:
   - Convert date columns to appropriate datetime format
   - Handle missing values and duplicates
   - Validate data integrity
4. **Exploratory Data Analysis**:
   - Statistical summaries of key metrics
   - Distribution analysis of customer attributes
   - Campaign performance metrics calculation
5. **Data Visualization**:
   - Income and age distribution plots
   - Campaign acceptance rate comparisons
   - Correlation analysis of numerical variables
6. **Customer Segmentation**:
   - K-means clustering to identify customer segments
   - Elbow method and silhouette analysis for optimal cluster selection
   - PCA visualization of customer groups
7. **Predictive Modeling**:
   - Logistic Regression and Random Forest models for campaign response prediction
   - Model evaluation with classification metrics and ROC curves
   - Feature importance analysis
8. **Data Export**: Upload cleaned dataset back to MySQL for further analysis or reporting

Key metrics analyzed include:
- Campaign acceptance rates
- Customer response patterns
- Demographic correlations with campaign success
- Customer segment characteristics
- Predictive model performance

## Results

The analysis provides valuable insights into marketing campaign effectiveness:

- **Data Quality**: Comprehensive cleaning process ensuring reliable analysis
- **Statistical Insights**: Summary statistics revealing campaign performance trends
- **Customer Segmentation**: Identification of distinct customer groups for targeted marketing
- **Predictive Models**: Machine learning models to forecast campaign response with feature importance analysis
- **Visual Analytics**: Multiple charts and graphs for data-driven decision making
- **Prepared Dataset**: Cleaned data ready for advanced analytics or machine learning models

Generated outputs include:
- Visualization PNG files for key insights
- Customer segment profiles and characteristics
- Model performance metrics and predictions
- Feature importance rankings

Future enhancements could include:
- Interactive dashboards using Streamlit or Dash
- A/B testing frameworks for campaign optimization
- Time series analysis for trend identification
- Advanced ML models (XGBoost, Neural Networks)
- Real-time prediction APIs

## Usage

### Option 1: Jupyter Notebook (Interactive)

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Run the cells sequentially** to execute the analysis pipeline.

3. **Customize the analysis** by modifying parameters or adding new exploration cells.

### Option 2: Python Script (Automated)

1. **Run the analysis script**:
   ```bash
   python marketing_analysis.py
   ```

2. **Check generated outputs** in the project directory (PNG visualizations, CSV results).

**Note**: Ensure your MySQL credentials are correctly configured in both the notebook and script before running database-related operations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Data Scientist**: [Your Name]
**Project Link**: [https://github.com/your-username/marketing-campaign-insight-analysis](https://github.com/your-username/marketing-campaign-insight-analysis)

For questions or collaborations, feel free to reach out!


