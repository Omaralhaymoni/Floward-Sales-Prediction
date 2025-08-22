# Weekly Sales Predictor (Streamlit)

Upload a CSV/XLSX/Parquet/JSON of your transactions and get weekly forecasts of
**Orders**, **Revenue**, and **AOV**. Includes occasion awareness (Gregorian + Hijri).

## Quick start (local)

```bash
python -m venv .venv && . .venv/Scripts/activate  # on mac/linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run predictor.py
