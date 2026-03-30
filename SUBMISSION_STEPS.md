# 📋 EXACT STEPS FOR SUBMISSION

## Step 1: Verify Data Files
Ensure these files are in `/Users/neenubonny/life_visualization/`:
- ✅ `utterance_raw_chatgpt_like.csv`
- ✅ `user_dim_20k.csv`
- ✅ `model_user_snapshot_100k.csv`

## Step 2: Open Notebook
Open `Project_Data_Mining.ipynb` in Jupyter/VS Code/Cursor

## Step 3: Run All Cells (In Order)

### Section 1: ANALYSIS
- **Cell 0**: Title and project structure (markdown - no execution needed)
- **Cell 1**: Analysis section header (markdown)
- **Cell 2**: Pipeline diagram (markdown)

### Section 2: DATA PREPROCESSING AND TRANSFORMATION
- **Cell 3**: Data Preprocessing header (markdown)
- **Cell 4**: **RUN THIS** - Feature engineering pipeline
  - Reads raw data
  - Creates daily metrics
  - Creates snapshot features
  - Joins demographics
  - Saves intermediate files
  - **Expected output**: Shapes of datasets, confirmation messages

### Section 3: EXPLANATORY DATA ANALYSIS (EDA)
- **Cell 5**: EDA header (markdown)
- **Cell 6**: **RUN THIS** - Complete EDA + Statistical Tests
  - Schema checks
  - Missing values
  - Target distribution (if churned_30d exists)
  - Numeric distributions
  - Outlier detection
  - Correlation heatmap
  - Statistical tests
  - **Expected output**: Tables, plots, test results

- **Cell 7**: **RUN THIS** - Seaborn Visual EDA
  - KDE plots
  - Violin plots
  - Count plots
  - Pairplot
  - **Expected output**: Multiple visualizations

### Section 4: FEATURE ENGINEERING AND FEATURE SELECTION
- **Cell 8**: Feature Engineering header (markdown)
- **Cell 9**: **RUN THIS** - Modeling Data Prep
  - Loads `model_user_snapshot_100k.csv`
  - Creates additional features
  - Train/test split
  - Preprocessing pipeline
  - Feature selection
  - Model training
  - **Expected output**: 
    - Feature lists
    - Train/test shapes
    - Classification report
    - ROC-AUC score

### Final: Submission Checklist
- **Cell 10**: Checklist (markdown - review before submitting)

## Step 4: Verify Outputs

After running all cells, check:

1. **No errors** - All cells executed successfully
2. **Data files created**:
   - `user_daily_metrics.csv`
   - `user_snapshot_features.csv`
   - `model_ready_features_with_demographics.csv`
3. **Model metrics displayed**:
   - Classification report (precision, recall, F1)
   - ROC-AUC score
4. **Visualizations generated**:
   - Distribution plots
   - Correlation heatmap
   - Boxplots
   - KDE plots

## Step 5: Submit

1. Save the notebook
2. Ensure all outputs are visible (run "Run All" if needed)
3. Export as PDF or submit the .ipynb file
4. Include data files if required by your instructor

---

## Quick Run Command (if using terminal)

```bash
cd /Users/neenubonny/life_visualization
jupyter nbconvert --to notebook --execute Project_Data_Mining.ipynb --output Project_Data_Mining_EXECUTED.ipynb
```

This will run all cells and save an executed version.

---

## Troubleshooting

**If you see "Target column 'churned_30d' not found":**
- Make sure `model_user_snapshot_100k.csv` is in the project folder
- The modeling cell should use `model_user_snapshot_100k.csv` (already fixed)

**If EDA cells show warnings:**
- This is expected if `churned_30d` isn't in `model_ready_features_with_demographics.csv`
- The modeling cell uses `model_user_snapshot_100k.csv` which has the target

**If plots don't show:**
- Make sure matplotlib backend is set correctly
- In Jupyter, plots should display automatically
- In VS Code, you may need to configure plot display settings
