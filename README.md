## Online Sales Visualization with AI Q&A (Streamlit)

This project lets you:
- Load online sales data from `.zip`, `.csv`, or `.xlsx`
- Explore with interactive charts and filters
- Ask natural-language questions about the data using OpenAI via PandasAI

### 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenAI API key (required for AI Q&A):
```bash
export OPENAI_API_KEY=sk-...
```

Alternatively, create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 2) Prepare a dataset

Option A: Use your local file  
- Place your dataset at `~/Downloads/online_sales.zip` (the app auto-tries this path), or  
- Use the sidebar to upload/select any `.zip`, `.csv`, or `.xlsx`

ZIPs may contain multiple CSV/XLSX files; the app will auto-pick a likely primary table (preferring names like `sales`, `orders`) or the largest table.

Option B: Download a Kaggle dataset (example)
- Recommended: Kaggle "Online Retail II" (Transactions of a UK-based non-store online retail)
  - Dataset: `https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci`

Using the Kaggle CLI:
```bash
# 1) Install and configure Kaggle
pip install kaggle
mkdir -p ~/.kaggle
cp /path/to/your/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 2) Download the dataset
kaggle datasets download -d mashlyn/online-retail-ii-uci -p ~/Downloads

# 3) You should now have something like:
# ~/Downloads/online-retail-ii-uci.zip
# Point the app at this ZIP, or rename to ~/Downloads/online_sales.zip for auto-load
```

### 3) Run the app
```bash
streamlit run app.py
```
Open the URL shown (usually `http://localhost:8501`).

### 4) Features
- Sidebar dataset selection (path input or file upload)
- Automatic parsing of CSV/TSV/XLSX and auto-datetime inference
- Filters:
  - Date range (if a datetime column is present)
  - Up to 3 categorical filters
- Charts:
  - Histogram for numeric columns
  - Bar aggregation by category (sum/mean/count/median)
  - Time series (count or metric) with daily/weekly/monthly frequency
- AI Q&A:
  - Ask natural questions like:
    - "What is total revenue by month?"
    - "Top 10 customers by total spend?"
    - "Average order value per country?"
  - Uses PandasAI + OpenAI API behind the scenes

### 5) Notes and Tips
- If the app shows "OPENAI_API_KEY is not set", provide it in the text box or via environment/.env
- Very large files may load faster if saved as CSV and compressed in a ZIP
- If your dataset has different column names, adjust your questions accordingly

### 6) Repository layout
```
app.py                         # Streamlit app entrypoint
src/
  ai_qna.py                    # PandasAI wrapper for Q&A
  charts.py                    # Plotly chart helpers
  data_loader.py               # Robust CSV/ZIP/XLSX loading + inference
requirements.txt               # Python dependencies
README.md
```
## Life Activity Dashboard

A dashboard-style visualization that combines Strava-like fitness data and weather patterns to reveal insights about your activity patterns in a single PNG.

### Setup

```bash
pip install -r requirements.txt
```

### Usage

1. **Generate sample data**

```bash
python generate_data.py
```

2. **Create the dashboard**

```bash
python create_dashboard.py
```

The dashboard will be saved as `life_dashboard.png` and includes:
- **Top metrics**: total distance, calories, number of activities, weather impact
- **Daily distance over time**: line + area chart
- **Activity type distribution**: pie chart
- **Weekly summary**: distance and activity counts
- **Temperature vs activity**: heatmap and insights panel

All key insights are contained visually in the dashboard PNG—no separate write‑up needed.
