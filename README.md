
# RetentionPro - Customer Retention & Segmentation Analytics

> Enterprise-grade analytics platform for customer segmentation, churn prediction, and retention strategy optimization.

## Overview

**RetentionPro** is an end-to-end customer analytics solution that transforms raw transaction data into actionable retention strategies. It combines advanced data science with an intuitive decision interface to help businesses identify at-risk customers and maximize lifetime value.

The platform includes:

- **Advanced Data Science Pipeline**: Cleans data, engineers features, performs customer segmentation via RFM analysis, and predicts churn probability.
- **Professional Analytics Dashboard**: Interactive Streamlit application for exploring customer segments, analyzing risk patterns, simulating campaign ROI, and exporting action plans.

## The Problem

Businesses often struggle with:

- **Visibility**: Who are the most valuable customers? Which segments deserve prioritization?
- **Risk Detection**: Which customers are likely to churn? What's driving their behavior?
- **Action Planning**: How to prioritize retention efforts within budget constraints?
- **Impact Measurement**: What's the expected ROI of different retention strategies?

**RetentionPro** answers these questions with a complete workflow from data ingestion to campaign execution.

## Key Features

✨ **Customer Segmentation**: Behavioral clustering based on RFM (Recency, Frequency, Monetary) signals
🎯 **Churn Prediction**: ML-powered risk scoring for each customer
📊 **Executive Dashboard**: Real-time analytics with interactive filters and visualizations  
🧪 **Campaign Simulator**: A/B test retention strategies and calculate ROI before execution
📋 **Action Plans**: Auto-generated prioritized watchlists ready for immediate outreach
🔍 **Data Explorer**: Full transparency into customer profiles and segment composition

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

1. **Prepare your data**: Ensure you have a transaction CSV file (default: `online_retail.csv`)

2. **Run the analytics pipeline**:
   ```bash
   python customer_segmentation_retention.py
   ```
   This generates:
   - `customer_segments.csv` - Customer profiles with segment assignments and churn scores
   - `segment_summary.csv` - Aggregate segment statistics
   - `churn_predictions.csv` - Churn prediction details
   - Visualization artifacts (elbow plot, cluster map, ROC curve)

3. **Launch the dashboard**:
   ```bash
   streamlit run streamlit_dashboard.py
   ```
   Open your browser to `http://localhost:8501`

## Project Objectives

- **Segment** customers into behavioral groups based on purchase patterns
- **Predict** individual churn risk with interpretable ML models
- **Export** ready-to-use output tables for CRM and marketing automation
- **Visualize** segment profiles, trends, and cohort retention patterns
- **Simulate** campaign impact and optimize retention spend

## Project Structure

```
RetentionPro/
├── customer_segmentation_retention.py   # Main data science pipeline
├── streamlit_dashboard.py               # Interactive analytics dashboard
├── requirements.txt                     # Python dependencies
│
├── Input Data:
│   └── online_retail.csv                # Sample transactional dataset
│
├── Generated Outputs:
│   ├── customer_segments.csv            # Customer profiles with churn scores
│   ├── segment_summary.csv              # Aggregated segment statistics
│   ├── churn_predictions.csv            # Detailed churn predictions
│   ├── clean_transactions.csv           # Cleaned transaction data
│   ├── plot_elbow.png                   # KMeans elbow curve
│   ├── plot_clusters_pca.png            # 2D segment visualization
│   ├── plot_churn_roc.png               # Churn model ROC curve
│   └── plot_segment_churn.png           # Churn rate by segment
│
└── README.md                            # This file
```

## How It Works

### Data Input & Flexibility

**RetentionPro** supports multiple data formats and automatically detects column names:

| Field | Detected Column Names |
|-------|----------------------|
| **Customer ID** | `customer_id`, `CustomerID`, `cust_id` |
| **Date** | `invoicedate`, `invoice_date`, `order_date` |
| **Amount** | `order_amount` or `quantity × unit_price` |
| **Country** | `country` (optional) |
| **Product** | `description`, `stockcode` (optional) |

#### Supported Data Formats

1. **Transaction-Level Data** (recommended): `online_retail.csv` format
   - One row per transaction
   - Multiple transactions per customer

2. **Customer-Level Data** (optional): `customers.csv` format
   - One row per customer
   - Pre-aggregated metrics

### Data Transformation Pipeline

The analytics engine executes this workflow:

**1. Data Cleansing**
- Normalize column names and data types
- Parse dates and numeric fields
- Handle missing values (median for numeric, mode for categorical)
- Remove duplicates and invalid records
- Filter out cancellations and negative amounts

**2. Feature Engineering**
For each customer, calculate:
- **Recency** (R): Days since last purchase
- **Frequency** (F): Number of purchases
- **Monetary** (M): Total lifetime spend
- **Tenure**: Days since first purchase
- **AOV**: Average order value

**3. Customer Segmentation (K-Means)**
- Scale features to zero mean, unit variance
- Test multiple segment counts (k=2 to k=8)
- Select optimal k using elbow method
- Assign each customer to nearest segment
- Generate 2D visualization via PCA

**4. Segment Profiling**
Automatically label clusters using percentile-based rules:
- High-Value Loyal
- At-Risk High-Value
- Regular Active
- Low-Engagement

**5. Churn Prediction**
- Build predictive model (Logistic Regression or Random Forest)
- Train on 80% of data with stratification
- Evaluate on 20% holdout set
- Generate individual churn probability scores
- Export evaluation metrics (Accuracy, Precision, Recall, ROC-AUC)

### Dashboard & Analytics Interface

The Streamlit dashboard provides:

**📊 Dashboard Tab**
- Key metrics: Customer count, revenue, average churn risk
- Segment revenue distribution
- Risk distribution histogram
- Monthly revenue and customer activity trends

**🔍 Segmentation Tab**
- Scatter plot: Value vs. Recency (bubble size = churn risk)
- Risk heatmap: Segment × Risk Band cross-tabulation
- Segment summaries and country-level revenue breakdowns

**📈 Trends & Products Tab**
- Cohort retention heatmap (month-by-month retention rates)
- Top products by revenue
- Monthly activity trends

**🎯 Campaign Simulator**
- Adjust target audience size, risk reduction, and costs
- Real-time ROI calculation
- Financial impact visualization

**📋 Action Plan**
- Prioritized customer watchlist (risk × value)
- Downloadable CSV export
- Segment-level action recommendations

**🗂️ Data Explorer**
- Full customer table with filters applied
- Export to CSV for CRM upload

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Data Processing** | pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Visualization** | matplotlib, seaborn |
| **Dashboard** | Streamlit |

All dependencies are pinned in `requirements.txt` for reproducibility.

## Quick Examples

### Example 1: Run Full Pipeline
```bash
python customer_segmentation_retention.py
streamlit run streamlit_dashboard.py
```

### Example 2: Analyze Specific Segment
1. Open the dashboard
2. Use filters to select a segment
3. Navigate to **Campaign Simulator** to test retention strategies
4. Export watchlist from **Action Plan** tab

### Example 3: Custom Data
Replace `online_retail.csv` with your own transaction data (same schema):
- `customer_id`, `invoicedate`, `invoiceno`, `quantity`, `unitprice`
- The pipeline auto-detects column name variants

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Column not found** | Check `DATA_PATH` in `customer_segmentation_retention.py` and column names in your CSV |
| **Dashboard won't start** | Verify Streamlit is installed: `pip install streamlit==1.28.0` |
| **Clustering produces only 1 segment** | Ensure sufficient customer variety in data (recommend 100+ unique customers) |
| **Churn model has low AUC** | Add more transaction data or adjust label logic in `Step 4` |

## Next Steps

- **Export Results**: Download watchlists and action plans from the dashboard
- **Integrate with CRM**: Use exported CSVs in your customer relationship management system
- **A/B Testing**: Use the campaign simulator to validate retention strategy assumptions
- **Monitoring**: Re-run the pipeline monthly to track segment drift and churn trends
- **Customization**: Modify segment labels or churn logic in `customer_segmentation_retention.py` for your business

## License

This project is provided as-is for educational and business analytics purposes.

## Contact & Support

For questions or feature requests, please refer to the project documentation or logs.

---

**Built with ❤️ for customer retention professionals.**

*Last Updated: 2026*

## 7. Generated Artifacts and Their Purpose

### `clean_transactions.csv`

Transaction-level cleaned dataset used by the pipeline and dashboard.

Typical columns include:

- customer identifier
- transaction date
- order amount
- item and country metadata when available

### `customer_segments.csv`

Main customer intelligence table.

Key columns:

- `customer_id`
- `recency`, `frequency`, `monetary`
- `tenure_days`, `avg_order_value`
- `cluster`, `segment_name`
- `churn`, `churn_probability`

### `segment_summary.csv`

Cluster-level aggregate profile:

- segment name
- number of customers
- average recency
- average frequency
- average monetary

### `churn_predictions.csv`

Compact operational file with:

- `customer_id`
- `churn`
- `churn_probability`

### Plot Outputs

- `plot_elbow.png`: choose number of clusters.
- `plot_clusters_pca.png`: visual cluster separation.
- `plot_churn_roc.png`: churn model discrimination quality.
- `plot_segment_churn.png`: segment-wise churn comparison.

## 8. Setup Instructions (Windows PowerShell)

From this project folder:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
```

## 9. How to Run

### 9.1 Run the Analytics Pipeline

```powershell
python customer_segmentation_retention.py
```

This generates all CSV and PNG artifacts in the same folder.

### 9.2 Launch the Dashboard

```powershell
streamlit run streamlit_dashboard.py
```

If `streamlit` is not on PATH:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run streamlit_dashboard.py
```

## 10. Dashboard Guide

The Streamlit app is organized into six tabs.

### 1) Executive Overview

- Revenue by segment
- Risk distribution
- Monthly trends for revenue and active customers

### 2) Segment Studio

- Value vs recency scatter
- Segment insights
- Country contribution
- Segment summary table
- Segment risk heatmap

### 3) Cohort and Product Lab

- Monthly cohort retention heatmap
- Top products by revenue

### 4) Retention Simulator

- Campaign what-if analysis
- Adjustable assumptions: target size, risk reduction, margin, contact cost, incentive cost
- Outputs: expected customers saved, recovered revenue, cost, ROI

### 5) Action Console

- Prioritized watchlist by blended score
- Downloadable watchlist CSV
- Segment-level action planning table

### 6) Data Explorer

- Full filtered customer table
- Download filtered results

## 11. Prioritization Logic Used in Dashboard

Watchlist ranking blends risk and value:

- `retention_priority = (risk_weight * churn_probability) + ((1 - risk_weight) * normalized_monetary)`

Where:

- `risk_weight` is user-controlled in the sidebar.
- `normalized_monetary` scales customer value to `[0, 1]`.

This balances saving high-risk customers and protecting high-value customers.

## 12. Interpretation Guidance

- High `churn_probability` and high `monetary`: top retention candidates.
- High recency and low frequency segments: likely reactivation targets.
- Low recency and high frequency segments: loyalty protection candidates.
- Use cohort retention drops to identify when interventions should occur.

## 13. Assumptions and Limitations

- If explicit churn labels are missing, churn is heuristically defined from recency.
- Model quality depends on data completeness and representativeness.
- KMeans assumes roughly spherical clusters in scaled feature space.
- Generated labels are meant for prioritization and experimentation, not perfect truth.

## 14. Troubleshooting

### `customer_segments.csv was not found`

Run:

```powershell
python customer_segmentation_retention.py
```

### Streamlit command not found

Use:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run streamlit_dashboard.py
```

### No rows after filtering in dashboard

- Widen sidebar ranges for risk, monetary, recency, or frequency.
- Re-include segments and clusters.

## 15. Recommended Next Enhancements

- Add unit tests for transformation and feature-engineering functions.
- Persist model artifacts (`joblib`) for reproducible inference.
- Add calibration and precision-recall diagnostics.
- Add experiment logging and model version metadata.
- Add CI checks for linting and tests.

## 16. Quick Start Summary

1. Install dependencies from `requirements.txt`.
2. Run `customer_segmentation_retention.py` to generate artifacts.
3. Run `streamlit_dashboard.py` to explore insights and export action lists.
