# Week 2: Data Collection, Cleaning & Preprocessing for Logistics Analysis

A documented data preprocessing pipeline for logistics/supply-chain transactional data — covering missing-value handling, duplicate removal, type standardization, outlier detection, and feature normalization, built with `pandas` and `scikit-learn`.

## Overview

Reliable logistics analytics starts with clean data. This project simulates ingesting a raw logistics export (modeled on the [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)) and walks through a full preprocessing pipeline, with reasoning for each technique chosen and its downstream impact on logistics KPIs and decisions.

## Contents

| File | Description |
|---|---|
| `Week2_Logistics_Data_Preprocessing.docx` | Full write-up: methodology, dataset characteristics, cleaning techniques, code walkthrough, and reflection on data quality's impact on logistics decision-making |
| `logistics_preprocessing_pipeline.py` | Standalone, reusable Python pipeline implementing the cleaning steps |

## Pipeline Steps

1. **Missing values** — median imputation for skewed numeric fields, mode/`"Unknown"` for categoricals, drop for unrecoverable critical dates
2. **Duplicates** — removed on a composite business key (Order ID + Product + Line Item)
3. **Type & format consistency** — date parsing, categorical casing standardization
4. **Outlier detection** — logical-bound checks (impossible shipping durations) treated as errors; IQR-based statistical outliers flagged for review rather than deleted
5. **Normalization** — Min-Max scaling fit on the training split only, to avoid data leakage

## Usage

```bash
pip install pandas numpy scikit-learn
python logistics_preprocessing_pipeline.py
```

Expects a raw CSV at `data/staging/logistics_orders_raw.csv`; outputs a cleaned CSV to `data/processed/logistics_orders_clean.csv`.

## Tools

`pandas` · `numpy` · `scikit-learn`

## Author

Anam Khan
