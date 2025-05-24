import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import yfinance as yf
import praw
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import sys

# Add parent directory to path to import data_utils
sys.path.append('..')
sys.path.append('.')

# Try to import data_utils with different approaches
try:
    import data_utils
except ImportError:
    try:
        # Try relative import
        from .. import data_utils
    except ImportError:
        # Create a minimal version if import fails
        print("Could not import data_utils, using minimal implementation")
        class data_utils:
            @staticmethod
            def process_sentiment(posts_df, comments_df):
                # Simple placeholder implementation
                sentiment_data = pd.DataFrame()
                if not posts_df.empty:
                    sentiment_data = posts_df.groupby('date').agg({
                        'id': 'count'
                    }).reset_index()
                    sentiment_data.rename(columns={'id': 'post_count'}, inplace=True)
                    sentiment_data['compound_sentiment_mean'] = 0
                return sentiment_data

def fetch_reddit_data(subreddits=['AMD_Stock', 'StockMarket', 'stocks', 'investing'], 
                      days_back=0.25, limit=100):  # 0.25 days = 6 hours
    """
    Fetch recent Reddit posts and comments related to AMD
    """
    # Initialize Reddit API client
    reddit = praw.Reddit(
        client_id=os.environ.get('REDDIT_CLIENT_ID'),
        client_secret=os.environ.get('REDDIT_CLIENT_SECRET'),
        user_agent=os.environ.get('REDDIT_USER_AGENT')
    )
    
    # Get today's date
    today = datetime.now()
    since_date = today - timedelta(days=days_back)
    
    all_posts = []
    all_comments = []
    
    # For each subreddit, get posts
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Get posts
        for post in subreddit.new(limit=limit):
            created_time = datetime.fromtimestamp(post.created_utc)
            if created_time >= since_date:
                # Check if post is related to AMD
                title_lower = post.title.lower()
                selftext_lower = post.selftext.lower()
                if 'amd' in title_lower or 'amd' in selftext_lower:
                    post_data = {
                        'id': post.id,
                        'title': post.title,
                        'selftext': post.selftext,
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments,
                        'created_utc': post.created_utc,
                        'subreddit': subreddit_name
                    }
                    all_posts.append(post_data)
                    
                    # Get comments for this post
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        comment_data = {
                            'id': comment.id,
                            'body': comment.body,
                            'score': comment.score,
                            'created_utc': comment.created_utc,
                            'post_id': post.id,
                            'subreddit': subreddit_name
                        }
                        all_comments.append(comment_data)
    
    # Convert to DataFrames
    posts_df = pd.DataFrame(all_posts)
    comments_df = pd.DataFrame(all_comments)
    
    # Add date column
    if not posts_df.empty:
        posts_df['date'] = pd.to_datetime(posts_df['created_utc'], unit='s').dt.date
    if not comments_df.empty:
        comments_df['date'] = pd.to_datetime(comments_df['created_utc'], unit='s').dt.date
    
    return posts_df, comments_df

def fetch_stock_data(ticker='AMD', days=90):
    """
    Fetch recent stock data from Yahoo Finance
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days+10)  # Add buffer days for calculations
    
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Reset index to make date a column
    stock_data = stock_data.reset_index()
    stock_data['date'] = stock_data['Date'].dt.date
    
    # Calculate features needed for prediction
    stock_data['daily_return'] = stock_data['Close'].pct_change()
    stock_data['ma5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['ma10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['ma20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['ma_crossover'] = (stock_data['Close'] > stock_data['ma5']).astype(int)
    stock_data['volume_change'] = stock_data['Volume'].pct_change()
    stock_data['price_range'] = stock_data['High'] - stock_data['Low']
    stock_data['gap'] = stock_data['Open'] - stock_data['Close'].shift(1)
    
    # Shift daily_return to create target for next day prediction
    stock_data['next_day_return'] = stock_data['daily_return'].shift(-1)
    stock_data['target'] = (stock_data['next_day_return'] > 0).astype(str)
    stock_data.loc[stock_data['next_day_return'].abs() < 0.005, 'target'] = 'flat'
    stock_data.loc[stock_data['next_day_return'] < 0, 'target'] = 'down'
    
    return stock_data

def process_sentiment(posts_df, comments_df):
    """
    Process sentiment for posts and comments
    """
    # Use data_utils to process sentiment
    # This assumes your data_utils has functions for sentiment analysis
    try:
        sentiment_data = data_utils.process_sentiment(posts_df, comments_df)
    except:
        # Fallback if data_utils doesn't have the right functions
        # Simple placeholder - in a real implementation, use your actual sentiment processing
        sentiment_data = pd.DataFrame()
        if not posts_df.empty:
            sentiment_data = posts_df.groupby('date').agg({
                'id': 'count',
                'score': ['mean', 'median', 'sum', 'std'],
                'num_comments': ['mean', 'median', 'sum', 'std'],
                'upvote_ratio': ['mean', 'median']
            })
            sentiment_data.columns = ['post_count', 'post_score_mean', 'post_score_median', 
                                     'post_score_sum', 'post_score_std', 'post_comments_mean',
                                     'post_comments_median', 'post_comments_sum', 'post_comments_std',
                                     'post_upvote_ratio_mean', 'post_upvote_ratio_median']
            sentiment_data = sentiment_data.reset_index()
            
            # Add placeholder sentiment values
            sentiment_data['compound_sentiment_mean'] = 0
            sentiment_data['compound_sentiment_std'] = 0
            
    return sentiment_data

def merge_data(sentiment_data, stock_data):
    """
    Merge sentiment and stock data
    """
    # Merge on date
    merged_data = pd.merge(stock_data, sentiment_data, on='date', how='left')
    
    # Fill missing sentiment data
    for col in sentiment_data.columns:
        if col != 'date':
            merged_data[col] = merged_data[col].fillna(method='ffill')
    
    # Drop rows with NaN (usually at the beginning due to rolling calculations)
    merged_data = merged_data.dropna()
    
    return merged_data

def update_model(merged_data):
    """
    Update the prediction model with new data
    """
    try:
        # Load existing model
        with open("../data/model_results/best_model.pkl", "rb") as f:
            model = pickle.load(f)
            
        # Get feature names
        with open("../data/model_results/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
            
        # Get class mapping
        with open("../data/model_results/class_mapping.pkl", "rb") as f:
            class_mapping = pickle.load(f)
        
        # Extract features and target
        X = merged_data[feature_names]
        y = merged_data['target']
        
        # Encode target
        encoder = LabelEncoder()
        encoder.classes_ = np.array(list(class_mapping.keys()))
        y_encoded = encoder.transform(y)
        
        # Update model (partial fit or retrain)
        # This depends on your model type and strategy
        # For simplicity, we'll just retrain on all data
        if isinstance(model, xgb.XGBClassifier):
            model.fit(X, y_encoded)
        else:
            model.fit(X, y_encoded)
        
        # Calculate feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.zeros(len(feature_names))
            
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Save updated model and artifacts
        with open("../data/model_results/best_model.pkl", "wb") as f:
            pickle.dump(model, f)
            
        feature_importance.to_csv("../data/model_results/feature_importance.csv", index=False)
        
        with open("../data/model_results/feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
            
        with open("../data/model_results/class_mapping.pkl", "wb") as f:
            pickle.dump(class_mapping, f)
            
        print("Model updated successfully")
        
    except Exception as e:
        print(f"Error updating model: {str(e)}")
        # If model update fails, we'll continue with the existing model

def update_data():
    """
    Main function to update all data
    """
    print("Starting data update process...")
    
    # Fetch new Reddit data
    print("Fetching Reddit data...")
    posts_df, comments_df = fetch_reddit_data()
    
    # Fetch new stock data
    print("Fetching stock data...")
    stock_data = fetch_stock_data()
    
    # Process sentiment
    print("Processing sentiment...")
    sentiment_data = process_sentiment(posts_df, comments_df)
    
    # Merge data
    print("Merging data...")
    merged_data = merge_data(sentiment_data, stock_data)
    
    # Save data
    print("Saving data...")
    os.makedirs("../data", exist_ok=True)
    
    # Save new data
    if not posts_df.empty:
        posts_df.to_parquet("../data/AMD_posts.parquet")
    if not comments_df.empty:
        comments_df.to_parquet("../data/AMD_comments.parquet")
    stock_data.to_parquet("../data/AMD_stock_data.parquet")
    sentiment_data.to_parquet("../data/combined_sentiment.parquet")
    merged_data.to_csv("../data/merged_data.csv", index=False)
    
    # Update model
    print("Updating model...")
    update_model(merged_data)
    
    print("Data update completed successfully!")

if __name__ == "__main__":
    update_data()
