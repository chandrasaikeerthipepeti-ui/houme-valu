# HomeValue — House Price Prediction Website

A Flask website that estimates residential property prices using a Random Forest regression model.

## Features
- Responsive house price prediction interface
- Inputs: Indian city, built-up area, bedrooms, bathrooms, property age, furnishing, parking and locality type
- Random Forest model trained with scikit-learn
- Model information page with test metrics
- Demo dataset included at `data/house_prices_india_demo.csv`

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended):

   Windows:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the website:
   ```bash
   python app.py
   ```

6. Open `http://127.0.0.1:5000` in your browser.

## Machine-learning details
- Algorithm: `RandomForestRegressor`
- Training/test split: 80% / 20%
- Random seed: 42
- Demo training records: 7,000
- Test MAE: ₹1,546,364
- Test R²: 0.939

## Important: demo data only
The included dataset is **synthetic**, generated for an academic demonstration. Its city rates and sample prices are illustrative, not verified market prices. The test metrics measure performance against this synthetic data and do not establish real-world accuracy. Do not use predictions for actual buying, selling, lending or investment decisions. For a realistic project, replace the CSV with verified, recent property transaction data and retrain the model.

## Project structure
```text
house_price_prediction_website/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── house_prices_india_demo.csv
├── models/
│   └── house_price_model.joblib
├── static/
│   └── style.css
└── templates/
    ├── index.html
    └── about.html
```
