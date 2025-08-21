# Weekly Sales Predictor (Streamlit)

Upload a CSV / Excel / Parquet / JSON to forecast weekly **Orders** & **Revenue**.
- Modes: **Train on new file** OR **Update existing history**
- Only control: **prediction horizon (weeks)**
- Single plot: historical + forecast + occasion markers (Gregorian + Hijri)

## Run locally
```bash
pip install -r requirements.txt
streamlit run predictor.py