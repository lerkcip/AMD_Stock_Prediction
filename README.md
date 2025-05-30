# AMD Stock Prediction Dashboard

A Streamlit dashboard that visualizes Reddit sentiment analysis and predicts AMD stock price movements.

## Features

- Reddit sentiment analysis for AMD-related posts and comments
- Historical stock price visualization
- Sentiment trend analysis with hourly breakdown
- Next-day price movement prediction
- Feature importance visualization

## Data Sources

- Reddit posts and comments from relevant subreddits
- AMD stock price data from Yahoo Finance

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Plotly
- Scikit-learn
- XGBoost
- NLTK/VADER for sentiment analysis

## Setup and Deployment

This dashboard is deployed on Streamlit Cloud and automatically updates daily with the latest Reddit sentiment and stock data.

## Repository Structure

- `app.py`: Main Streamlit application
- `data/`: Contains datasets and model artifacts
- `scripts/`: Contains data update scripts
- `docs/`: Contains project documentation and white papers
- `notebooks/`: Contains Jupyter notebooks with analysis code
- `images/`: Contains visualization images used in reports
- `.github/workflows/`: Contains GitHub Actions workflow for daily updates (not currently working)
