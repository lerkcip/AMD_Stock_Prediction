import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pickle
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

ticker = 'AMD'
days = 90
sentiment = 6

# Load model and artifacts
@st.cache_resource
def load_model_and_artifacts():
    try:
        with open("data/model_results/best_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load feature importance
        feature_importance = pd.read_csv("data/model_results/feature_importance.csv")
        
        # Load feature names if available
        try:
            with open("data/model_results/feature_names.pkl", "rb") as f:
                feature_names = pickle.load(f)
                # If model doesn't have feature_names_in_, add it
                if not hasattr(model, 'feature_names_in_'):
                    model.feature_names_in_ = feature_names
        except Exception as e:
            st.error(f"Error loading feature names: {str(e)}")
            st.stop()
        
        # Load class mapping if available
        try:
            with open("data/model_results/class_mapping.pkl", "rb") as f:
                class_mapping = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading class mapping: {str(e)}")
            st.error("Please ensure class_mapping.pkl exists in data/model_results/")
            st.stop()
        
        return model, feature_importance, class_mapping
    except Exception as e:
        st.error(f"Error loading model or artifacts: {str(e)}")
        st.error("Required files not found. Please ensure the following files exist:")
        st.error("- data/model_results/best_model.pkl")
        st.error("- data/model_results/feature_importance.csv")
        st.error("- data/model_results/feature_names.pkl")
        st.error("- data/model_results/class_mapping.pkl")
        st.stop()

# Load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("data/merged_data.csv")
        if data.empty:
            st.error("The merged_data.csv file is empty.")
            st.stop()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Required file not found: data/merged_data.csv")
        st.stop()

# Fetch recent sentiment data
@st.cache_data
def get_recent_sentiment():
    # In a real application, this would fetch new data from Reddit
    # For this demo, we'll use the most recent data from our dataset
    data = load_data()
    sentiment_data = data.sort_values('date', ascending=False).head(15)
    
    if 'compound_sentiment_mean' not in sentiment_data.columns:
        st.error("Required column 'compound_sentiment_mean' not found in sentiment data.")
        st.error("Available columns: " + ", ".join(sentiment_data.columns))
        st.stop()
    
    return sentiment_data

# Fetch recent stock data
@st.cache_data
def get_recent_stock_data(ticker=ticker, days=days):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if stock_data.empty:
            st.error(f"No stock data found for {ticker} from {start_date.date()} to {end_date.date()}")
            st.error("Please check your internet connection or try a different ticker symbol.")
            st.stop()
        
        # Calculate features needed for prediction
        stock_data['daily_return'] = stock_data['Close'].pct_change()
        stock_data['ma5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['ma10'] = stock_data['Close'].rolling(window=10).mean()
        stock_data['ma20'] = stock_data['Close'].rolling(window=20).mean()
        
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        st.error("Failed to download stock data from Yahoo Finance.")
        st.error("Please check your internet connection and try again.")
        st.stop()

# Make prediction
def predict_next_day(model, sentiment_data, stock_data, feature_importance, class_mapping):
    """
    Make prediction for the next trading day
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    sentiment_data : pandas.DataFrame
        Recent sentiment data
    stock_data : pandas.DataFrame
        Recent stock data
    feature_importance : pandas.DataFrame
        Feature importance DataFrame
    class_mapping : dict
        Mapping of class labels to encoded values
        
    Returns:
    --------
    tuple
        Prediction and probabilities
    """
    # Get the feature names used during training
    if hasattr(model, 'feature_names_in_'):
        # For sklearn 1.0+ models that store feature names
        model_features = model.feature_names_in_
        print(f"Using {len(model_features)} features from model.feature_names_in_")
    else:
        # Fall back to feature importance order
        model_features = feature_importance['feature'].tolist()
        print(f"Using {len(model_features)} features from feature importance")
    
    # Prepare features dictionary
    features = {}
    
    # Add sentiment features if available
    for col in model_features:
        if col in sentiment_data.columns:
            features[col] = sentiment_data[col].iloc[0]
        elif col in stock_data.columns:
            features[col] = stock_data[col].iloc[-1]
        else:
            # Feature not available, use 0
            features[col] = 0
            print(f"Feature '{col}' not found in input data, using 0")
    
    # Convert to DataFrame with features in the exact order used by the model
    features_df = pd.DataFrame([features])
    
    # Ensure columns are in the same order as during training
    features_df = features_df[model_features]
    
    # Handle missing values
    features_df = features_df.fillna(0)
    
    # Print feature values for debugging
    print("Features for prediction:")
    for col in model_features:
        print(f"  {col}: {features_df[col].iloc[0]}")
    
    # Make prediction
    try:
        prediction_encoded = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # Convert prediction back to original class
        inv_class_mapping = {v: k for k, v in class_mapping.items()}
        prediction = inv_class_mapping[prediction_encoded]
        
        # Create dictionary mapping class names to probabilities
        prob_dict = {inv_class_mapping[i]: prob for i, prob in enumerate(probabilities)}
        
        return prediction, prob_dict
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Model features: " + ", ".join(model_features))
        st.error("Available features: " + ", ".join(features_df.columns.tolist()))
        st.stop()

### Dashboard Setup and Data Loading

# Set up the dashboard title
st.title(f"Reddit Sentiment & {ticker} Stock Prediction")

# Load data and model
model, feature_importance, class_mapping = load_model_and_artifacts()
data = load_data()
recent_sentiment = get_recent_sentiment()
recent_stock = get_recent_stock_data(ticker=ticker)

# Make prediction
prediction, probabilities = predict_next_day(model, recent_sentiment, recent_stock, feature_importance, class_mapping)

### Prediction Display

# Display prediction
st.header("Next Day Prediction")

col1, col2, col3 = st.columns(3)

# Set color based on prediction
color = "green" if prediction == "up" else "red" if prediction == "down" else "gray"

with col1:
    st.markdown(f"<h3 style='text-align: center; color: {color};'>{prediction.upper()}</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Prediction</p>", unsafe_allow_html=True)

with col2:
    # Extract the value and convert to float
    current_price = float(recent_stock['Close'].iloc[-1])
    st.metric("Current Price", f"${current_price:.2f}")

with col3:
    # Extract values and convert to float
    prev_close = float(recent_stock['Close'].iloc[-2])
    current_close = float(recent_stock['Close'].iloc[-1])
    prev_day_return = ((current_close - prev_close) / prev_close) * 100
    delta_color = "normal" if abs(prev_day_return) < 0.5 else "off" if prev_day_return < 0 else "normal"
    st.metric("Previous Day Return", f"{prev_day_return:.2f}%", delta=f"{prev_day_return:.2f}%", delta_color=delta_color)

### Prediction Probability Visualization

# Display probability chart
st.subheader("Prediction Probabilities")

fig, ax = plt.subplots(figsize=(10, 5))

labels = list(probabilities.keys())
probs = [probabilities[label] * 100 for label in labels]
colors = ['red' if label == 'down' else 'gray' if label == 'flat' else 'green' for label in labels]

ax.bar(labels, probs, color=colors)
ax.set_ylabel('Probability (%)')
ax.set_ylim(0, 100)

for i, v in enumerate(probs):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center')

st.pyplot(fig)

# Display feature importance
st.subheader("Top 10 Most Important Features")

# Create a mapping of technical feature names to layperson terms
feature_name_mapping = {
    'ma5': '5-Day Moving Avg.',
    'ma10': '10-Day Moving Avg.',
    'ma20': '20-Day Moving Avg.',
    'ma_crossover': 'Price Above 5D Avg.',
    'daily_return': 'Daily Return',
    'prev_return': 'Previous Day Return',
    'compound_sentiment_mean': 'Average Sentiment',
    'compound_sentiment_median': 'Median Sentiment',
    'compound_sentiment_std': 'Sentiment Variability',
    'compound_sentiment_count': 'Sentiment Sample Size',
    'title_compound_mean': 'Post Title Sentiment',
    'title_compound_median': 'Median Title Sentiment',
    'title_compound_std': 'Title Sentiment Variability',
    'selftext_compound_mean': 'Post Content Sentiment',
    'selftext_compound_median': 'Median Content Sentiment',
    'selftext_compound_std': 'Content Sentiment Variability',
    'body_compound_mean': 'Comment Sentiment',
    'body_compound_median': 'Median Comment Sentiment',
    'body_compound_std': 'Comment Sentiment Variability',
    'post_count': 'Number of Posts',
    'post_score_mean': 'Average Post Score',
    'post_score_median': 'Median Post Score',
    'post_score_sum': 'Total Post Score',
    'post_score_std': 'Post Score Variability',
    'post_comments_mean': 'Average Comments per Post',
    'post_comments_median': 'Median Comments per Post',
    'post_comments_sum': 'Total Comments',
    'post_comments_std': 'Comment Count Variability',
    'post_upvote_ratio_mean': 'Average Upvote Ratio',
    'post_upvote_ratio_median': 'Median Upvote Ratio',
    'post_score_log': 'Log-Scaled Post Score',
    'comment_count': 'Number of Comments',
    'comment_score_mean': 'Average Comment Score',
    'comment_score_median': 'Median Comment Score',
    'comment_score_sum': 'Total Comment Score',
    'comment_score_std': 'Comment Score Variability',
    'comment_score_log': 'Log-Scaled Comment Score',
    'engagement_ratio': 'Comments per Post Ratio'
}

# Create a copy of the top 10 features with friendly names
top_10_features = feature_importance.head(10).copy()
top_10_features['friendly_name'] = top_10_features['feature'].map(
    lambda x: feature_name_mapping.get(x, x.replace('_', ' ').title())
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='importance_mean', y='friendly_name', data=top_10_features, ax=ax)
ax.set_title('Feature Importance')
ax.set_xlabel('Importance')
ax.set_ylabel('')  # Remove y-axis label as the friendly names are self-explanatory
st.pyplot(fig)

# Add a description of what the features mean - dynamically based on top 10 features
with st.expander("Feature Descriptions"):
    # Group features by category
    sentiment_features = []
    engagement_features = []
    technical_features = []
    other_features = []
    
    # Get the top 10 feature names
    top_features = top_10_features['friendly_name'].tolist()
    original_features = top_10_features['feature'].tolist()
    
    # Create descriptions dictionary
    feature_descriptions = {
        # Sentiment Features
        'compound_sentiment_mean': "Overall average sentiment score across all Reddit content",
        'compound_sentiment_std': "How much sentiment varies across posts and comments",
        'title_sentiment_mean': "Average sentiment in post titles",
        'comment_sentiment_mean': "Average sentiment in comments",
        'comment_sentiment_std': "Variability in comment sentiment",
        
        # Engagement Metrics
        'post_count': "Total Reddit posts in the time period",
        'post_score_mean': "Average score (upvotes minus downvotes) of posts",
        'post_score_median': "Median score of posts",
        'post_score_sum': "Total score across all posts",
        'post_score_std': "Variability in post scores",
        'post_comments_mean': "Average number of comments per post",
        'post_comments_median': "Median number of comments per post",
        'post_comments_sum': "Total comments across all posts",
        'post_comments_std': "Variability in comment counts",
        'post_upvote_ratio_mean': "Average percentage of upvotes out of all votes",
        'post_upvote_ratio_median': "Median percentage of upvotes",
        'post_score_log': "Log-scaled post score to reduce impact of outliers",
        'comment_count': "Total number of comments",
        'comment_score_mean': "Average score of comments",
        'comment_score_median': "Median score of comments",
        'comment_score_sum': "Total score across all comments",
        'comment_score_std': "Variability in comment scores",
        'comment_score_log': "Log-scaled comment score",
        'engagement_ratio': "Average number of comments per post",
        
        # Technical Indicators
        'ma5': "Average closing price over the past 5 trading days",
        'ma10': "Average closing price over the past 10 trading days",
        'ma20': "Average closing price over the past 20 trading days",
        'ma_crossover': "Whether price is above the 5-day moving average",
        'daily_return': "Percentage change in price from previous day",
        'prev_return': "Previous day's percentage change in price",
        'volume': "Number of shares traded",
        'volume_change': "Change in trading volume from previous day",
        'price_range': "Difference between high and low price",
        'gap': "Difference between today's open and previous day's close"
    }
    
    # Categorize the top features
    for i, feature in enumerate(original_features):
        friendly_name = top_features[i]
        description = feature_descriptions.get(feature, "Feature derived from Reddit data or stock price")
        
        # Determine category based on feature name
        if any(term in feature for term in ['sentiment', 'title', 'comment_sentiment']):
            sentiment_features.append((friendly_name, description))
        elif any(term in feature for term in ['post_', 'comment_', 'engagement', 'upvote', 'score']):
            engagement_features.append((friendly_name, description))
        elif any(term in feature for term in ['ma', 'return', 'volume', 'price', 'gap']):
            technical_features.append((friendly_name, description))
        else:
            other_features.append((friendly_name, description))
    
    # Display the categorized features
    if sentiment_features:
        st.markdown("#### Sentiment Features")
        for name, desc in sentiment_features:
            st.markdown(f"- **{name}**: {desc}")
        st.markdown("")
    
    if engagement_features:
        st.markdown("#### Engagement Metrics")
        for name, desc in engagement_features:
            st.markdown(f"- **{name}**: {desc}")
        st.markdown("")
    
    if technical_features:
        st.markdown("#### Technical Indicators")
        for name, desc in technical_features:
            st.markdown(f"- **{name}**: {desc}")
        st.markdown("")
    
    if other_features:
        st.markdown("#### Other Features")
        for name, desc in other_features:
            st.markdown(f"- **{name}**: {desc}")


### Sentiment Trend Visualization

# Display recent sentiment
st.header("Reddit Sentiment Analysis")

# Create a function to prepare date range data
def prepare_date_range_data(df, date_col='date', value_cols=None):
    # Get the last 7 days
    today = datetime.now().date()
    date_range = [(today - timedelta(days=i)) for i in range(sentiment, -1, -1)]
    
    # Create a template DataFrame with the date range
    template_df = pd.DataFrame({date_col: date_range})
    
    # Prepare the input data
    df_copy = df.copy()
    if df_copy.empty:
        # If no data, return the template with zeros
        for col in value_cols or []:
            template_df[col] = 0
        return template_df
    
    # Convert dates to datetime objects
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Convert to date objects for merging
    df_copy[date_col] = df_copy[date_col].dt.date
    
    # Merge with template
    result = pd.merge(template_df, df_copy, on=date_col, how='left')
    
    # Fill NaN values with 0
    if value_cols:
        for col in value_cols:
            if col in result.columns:
                result[col] = result[col].fillna(0)
    
    return result

# Get the data ready
value_cols = ['compound_sentiment_mean', 'post_count', 'comment_count', 'engagement_ratio']
plot_data = prepare_date_range_data(recent_sentiment, value_cols=value_cols)

# Convert dates to strings for display
plot_data['date_str'] = [d.strftime('%b %d') for d in plot_data['date']]

# Create a function to prepare hourly data for a smoother graph
def prepare_hourly_sentiment_data(df):
    # Get the original data with full datetime information
    # In a real app, this would use the actual timestamps from the data
    # For this demo, we'll simulate hourly data based on the daily data
    
    # First, get the unique dates sorted chronologically
    unique_dates = sorted(df['date'].unique())
    
    # Create a list to hold the hourly data
    hourly_data = []
    
    # Track the last valid sentiment value to use when no data is available
    last_valid_sentiment = 0  # Default starting value
    
    # For each date, create hourly data points
    for date in unique_dates:
        # Get the sentiment values and post counts for this date
        date_df = df[df['date'] == date]
        date_sentiments = date_df['compound_sentiment_mean'].tolist()
        
        # Check if this day has any posts/comments
        has_posts = False
        total_posts = 0
        if 'post_count' in date_df.columns:
            total_posts = date_df['post_count'].sum()
            has_posts = total_posts > 0
        
        # If this day has no posts/comments, use the last valid sentiment
        if not has_posts and date_sentiments:
            # No posts but we have a sentiment value (which must be 0)
            # This is not a legitimate 0, so use the previous value
            sentiment_to_use = last_valid_sentiment
        elif not date_sentiments:
            # No sentiment data at all for this day
            sentiment_to_use = last_valid_sentiment
        else:
            # We have posts and sentiment data, or the sentiment is legitimately 0
            sentiment_to_use = date_sentiments[0]
            # Update the last valid sentiment
            last_valid_sentiment = sentiment_to_use
        
        # Create hourly data points for this day
        for hour in [9, 12, 15, 18]:
            # Add very small variations to create a natural-looking curve
            variation = np.random.normal(0, 0.005) if has_posts else 0  # Only add variation if we have posts
            hourly_data.append({
                'datetime': pd.Timestamp(date).replace(hour=hour),
                'sentiment': sentiment_to_use + variation,
                'post_count': total_posts / 4  # Distribute posts across hours
            })
    
    # Convert to DataFrame and sort by datetime
    hourly_df = pd.DataFrame(hourly_data)
    hourly_df = hourly_df.sort_values('datetime')
    
    return hourly_df

# Post Activity section first
st.subheader("📈 Post Activity")

# Create a card-like container for metrics with enhanced styling
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: space-between;
    margin: 0px 0 20px 0;
}
.metric-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px 15px;
    text-align: center;
    margin-right: 15px;
    flex: 1;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:last-child {
    margin-right: 0;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
}
.metric-value {
    font-size: 48px;
    font-weight: 700;
    margin: 15px 0;
    line-height: 1.2;
}
.metric-label {
    font-size: 16px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    font-weight: 500;
}
.metric-sublabel {
    font-size: 14px;
    color: #777;
    font-style: italic;
    margin-top: 5px;
}
.sentiment-positive {
    color: #2ecc71;
    text-shadow: 0 0 15px rgba(46, 204, 113, 0.2);
}
.sentiment-negative {
    color: #e74c3c;
    text-shadow: 0 0 15px rgba(231, 76, 60, 0.2);
}
.sentiment-neutral {
    color: #7f8c8d;
}
</style>
""", unsafe_allow_html=True)

# Calculate metrics
total_posts = int(plot_data['post_count'].sum())
avg_sentiment = plot_data['compound_sentiment_mean'].mean()
sentiment_direction = "positive" if avg_sentiment > 0 else "negative" if avg_sentiment < 0 else "neutral"
sentiment_class = f"sentiment-{sentiment_direction}"

# Create a row of metrics
st.markdown("<div class='metric-container'>", unsafe_allow_html=True)

# Total Posts metric
st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">TOTAL POSTS</div>
    <div class="metric-value" style="color: #3498db;">{total_posts}</div>
</div>
""", unsafe_allow_html=True)

# Add additional metrics if available
if 'comment_count' in plot_data.columns:
    total_comments = int(plot_data['comment_count'].sum())
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TOTAL COMMENTS</div>
        <div class="metric-value" style="color: #3498db;">{total_comments}</div>
    </div>
    """, unsafe_allow_html=True)
    
# Average Sentiment metric
st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">AVERAGE SENTIMENT</div>
    <div class="metric-value {sentiment_class}">{avg_sentiment:.2f}</div>
    <div class="metric-sublabel">({sentiment_direction})</div>
</div>
""", unsafe_allow_html=True)

# Close the container
st.markdown("</div>", unsafe_allow_html=True)

# Sentiment Trend section second
st.subheader(f"📊 Sentiment Trend")

# Prepare hourly data for a smoother graph
hourly_data = prepare_hourly_sentiment_data(recent_sentiment)

# Create a modern, clean visualization
fig = plt.figure(figsize=(12, 6))
plt.style.use('seaborn-v0_8-whitegrid')  # Use a cleaner style

# Plot the sentiment line with hourly detail
plt.plot(
    hourly_data['datetime'],
    hourly_data['sentiment'],
    linewidth=3,
    color='#3498db',
    alpha=0.9,
    marker='o',
    markersize=6,
    markerfacecolor='white',
    markeredgecolor='#3498db',
    markeredgewidth=2
)

# Add a gradient fill below the line
# Use a colormap that transitions from red to green based on sentiment
for i in range(len(hourly_data) - 1):
    x_vals = [hourly_data['datetime'].iloc[i], hourly_data['datetime'].iloc[i+1]]
    y_vals = [hourly_data['sentiment'].iloc[i], hourly_data['sentiment'].iloc[i+1]]
    
    # Determine color based on sentiment value
    if y_vals[0] >= 0 and y_vals[1] >= 0:
        color = '#2ecc71'  # Green for positive
        alpha = 0.3
    elif y_vals[0] < 0 and y_vals[1] < 0:
        color = '#e74c3c'  # Red for negative
        alpha = 0.3
    else:
        # For crossing zero, use a gradient
        color = '#f39c12'  # Orange for mixed
        alpha = 0.2
    
    plt.fill_between(
        x_vals, y_vals, 0,
        color=color,
        alpha=alpha
    )

# Add a zero line
plt.axhline(y=0, color='#7f8c8d', linestyle='-', linewidth=1, alpha=0.5)

# Calculate appropriate date interval based on the date range
date_range = (hourly_data['datetime'].max() - hourly_data['datetime'].min()).days
date_interval = max(1, date_range // 10)  # Show about 10 date labels

# Format the x-axis to show only the date (not the hour)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Use DayLocator for the major ticks (dates)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=date_interval))

# Add minor ticks for hours but don't label them
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=12))
plt.gca().tick_params(axis='x', which='minor', length=4, color='gray')

# Rotate date labels for better readability
plt.xticks(ha='right')

# Add labels and grid
plt.ylabel('Sentiment Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)

# Set y-axis limits with some padding
max_sentiment = max(0.2, hourly_data['sentiment'].max() * 1.2)
min_sentiment = min(-0.1, hourly_data['sentiment'].min() * 1.2)
plt.ylim(min_sentiment, max_sentiment)

# Add annotations for significant sentiment changes
for i in range(1, len(hourly_data)):
    current = hourly_data['sentiment'].iloc[i]
    previous = hourly_data['sentiment'].iloc[i-1]
    change = current - previous
    
    # Only annotate significant changes
    if abs(change) > 0.1:
        direction = '↑' if change > 0 else '↓'
        color = '#2ecc71' if change > 0 else '#e74c3c'
        plt.annotate(
            f"{direction} {abs(change):.2f}",
            xy=(hourly_data['datetime'].iloc[i], current),
            xytext=(0, 10 if change > 0 else -20),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            fontweight='bold',
            color=color,
            arrowprops=dict(arrowstyle='->', color=color, alpha=0.7)
        )

# Tight layout
plt.tight_layout()

# Show the plot
st.pyplot(fig)

# Create a section for sentiment details
st.markdown("### 🔍 Sentiment Details")

# Create a table with daily sentiment data
detail_data = plot_data[['date_str', 'compound_sentiment_mean', 'post_count']].copy()
detail_data.columns = ['Date', 'Sentiment Score', 'Post Count']
detail_data['Sentiment Score'] = detail_data['Sentiment Score'].round(3)

# Add a sentiment indicator column
def get_sentiment_emoji(score):
    if score > 0.10: return "😀 Positive"
    elif score < -0.10: return "😟 Negative"
    else: return "😐 Neutral"

detail_data['Sentiment'] = detail_data['Sentiment Score'].apply(get_sentiment_emoji)

# Display the table
st.dataframe(detail_data, use_container_width=True)

# Display engagement metrics if available
if any(col in recent_sentiment.columns for col in ['comment_count', 'engagement_ratio']):
    st.markdown("### 📱 Engagement Metrics")
    
    # Filter for engagement metrics
    engagement_metrics = ['comment_count', 'engagement_ratio']
    available_metrics = [col for col in engagement_metrics if col in plot_data.columns]
    
    if available_metrics:
        # Create a clean visualization for engagement metrics
        fig3 = plt.figure(figsize=(10, 4))
        plt.style.use('ggplot')
        
        # Plot each metric with a different color
        colors = ['#9b59b6', '#2980b9', '#1abc9c']
        
        for i, metric in enumerate(available_metrics):
            plt.plot(
                plot_data['date'],
                plot_data[metric],
                marker='o',
                markersize=6,
                linewidth=2,
                color=colors[i % len(colors)],
                label=metric.replace('_', ' ').title()
            )
        
        # Clean up the plot
        plt.ylabel('Count/Ratio', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(plot_data['date'], plot_data['date_str'], rotation=0)
        
        # Add a legend
        plt.legend(loc='upper left', frameon=True)
        
        # Tight layout
        plt.tight_layout()
        
        # Show the plot
        st.pyplot(fig3)

# The engagement metrics are now handled in the section above

### Stock Price Visualization

# Display recent stock price
st.header(f"Recent {ticker} Stock Price")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(recent_stock.index, recent_stock['Close'], label='Close Price')

# Add moving averages
if 'ma5' in recent_stock.columns and 'ma10' in recent_stock.columns and 'ma20' in recent_stock.columns:
    ax.plot(recent_stock.index, recent_stock['ma5'], label='5-day MA', alpha=0.7)
    ax.plot(recent_stock.index, recent_stock['ma10'], label='10-day MA', alpha=0.7)
    ax.plot(recent_stock.index, recent_stock['ma20'], label='20-day MA', alpha=0.7)

ax.set_ylabel('Price ($)')
ax.set_title(f'{ticker} Stock Price (Last {days} Days)')
ax.grid(True, alpha=0.3)
ax.legend()

# Format x-axis dates to be more readable
date_format = mdates.DateFormatter('%b %d')  # Format as 'Apr 25', 'May 21', etc.
ax.xaxis.set_major_formatter(date_format)

# Set the number of ticks based on the date range
days_shown = (recent_stock.index[-1] - recent_stock.index[0]).days
tick_interval = max(1, days_shown // 8)  # Show around 8 ticks on the x-axis
ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))

# Rotate labels for better readability
plt.xticks(ha='right')

# Adjust layout to make room for rotated labels
plt.tight_layout()

st.pyplot(fig)

# Display daily returns
fig, ax = plt.subplots(figsize=(10, 5))

if 'daily_return' in recent_stock.columns:
    returns = recent_stock['daily_return'] * 100
    ax.bar(recent_stock.index, returns, color=returns.apply(lambda x: 'green' if x > 0 else 'red'))
    ax.set_ylabel('Daily Return (%)')
    ax.set_title(f'{ticker} Daily Returns (Last {days} Days)')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates to be more readable - same as stock price chart
    date_format = mdates.DateFormatter('%b %d')
    ax.xaxis.set_major_formatter(date_format)
    
    # Set the number of ticks based on the date range
    days_shown = (recent_stock.index[-1] - recent_stock.index[0]).days
    tick_interval = max(1, days_shown // 8)  # Show around 8 ticks on the x-axis
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=tick_interval))
    
    # Rotate labels for better readability
    plt.xticks(ha='right')
    
    # Adjust layout to make room for rotated labels
    plt.tight_layout()
    
    st.pyplot(fig)

### Raw Data Display

# Display raw data
with st.expander("View Raw Sentiment Data"):
    st.dataframe(recent_sentiment)

with st.expander("View Raw Stock Data"):
    st.dataframe(recent_stock)

# Display model information
with st.expander("View Model Information"):
    st.write(f"Model Type: {type(model).__name__}")
    
    if hasattr(model, 'get_params'):
        st.write("Model Parameters:")
        st.json(model.get_params())
    
    st.write("Class Mapping:")
    st.json(class_mapping)
    
    st.write("Top Features:")
    st.dataframe(feature_importance.head(10))
