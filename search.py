import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Function to fetch stock data using yfinance
def fetch_stock_data(ticker):
    df = yf.download(ticker, start="2010-01-01", progress=False)
    if df.empty:
        raise ValueError("Invalid ticker symbol or no data found.")
    return df[['Close']]

# Function to create features for prediction
def create_features(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    return df

# Train model and predict stock price
def predict_stock(ticker):
    df = fetch_stock_data(ticker)
    df = create_features(df)

    features = ['Close', 'MA10', 'MA50', 'MA100']
    X = df[features]
    y = df['Target']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    latest_data = df[features].iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(latest_data)[0]
    predictions = model.predict(X)

    return df, predicted_price, y, predictions

# Plotting functions
def plot_company_graph(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['MA10'], label='MA10', color='orange')
    plt.plot(df['MA50'], label='MA50', color='green')
    plt.plot(df['MA100'], label='MA100', color='red')
    plt.legend()
    plt.title('Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    st.pyplot(plt)

def plot_actual_vs_predicted(df, predicted_price, predictions):
    actual_prices = df['Close'].values

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, actual_prices, label='Actual Price', color='blue')
    plt.axvline(df.index[-1], color='gray', linestyle='--', label='Prediction Point')
    plt.plot(df.index, predictions, label='Predicted Price', color='red', linestyle='--')
    plt.legend()
    plt.title(f'Actual vs Predicted Price - Predicted Price: {predicted_price:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(plt)

# Recommendation logic
def generate_recommendation(current_price, predicted_price):
    change = predicted_price - current_price
    percent_change = (change / current_price) * 100

    if percent_change > 2:
        return "✅ Strong upward trend. Consider buying!"
    elif percent_change > 0:
        return "⚠️ Slight increase expected. Monitor carefully."
    elif percent_change < 0:
        return "❌ Potential decline. Reconsider investment."
    else:
        return "⚖️ No significant change. Hold position."

# Main app
def main():
    st.set_page_config(page_title="Stock Predictor", layout="wide")

    st.markdown("<h1 style='color:#4B8BBE;'>📈 Stock Price Predictor with Smart Recommendations</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Load company list CSV
    try:
        df_companies = pd.read_csv('company_list.csv')  # Columns: 'Company Name', 'Ticker'
    except FileNotFoundError:
        st.error("Company list file 'company_list.csv' not found.")
        return

    # Sidebar - Company Search with Default Full List
    st.sidebar.title("🏢 Company List")
    search_query = st.sidebar.text_input("🔍 Search Company")

    if search_query:
        filtered = df_companies[df_companies['Company Name'].str.contains(search_query, case=False, na=False)]
    else:
        filtered = df_companies

    for _, row in filtered.iterrows():
        st.sidebar.write(f"{row['Company Name']} ({row['Ticker']})")

    # Main - Ticker input
    ticker_input = st.text_input("🔎 Enter Ticker Symbol (e.g., AAPL, MSFT, TSLA)").upper().strip()

    if ticker_input:
        try:
            df, predicted_price, y_true, predictions = predict_stock(ticker_input)
            current_price = float(df['Close'].iloc[-1])
            predicted_price = float(predicted_price)

            recommendation = generate_recommendation(current_price, predicted_price)

            # Get company info
            company_info = yf.Ticker(ticker_input).info
            name = company_info.get("longName", ticker_input)
            logo = company_info.get("logo_url", "")

            cols = st.columns([1, 4])
            if logo:
                cols[0].image(logo, width=80)
            cols[1].markdown(f"### 🏢 {name} ({ticker_input})")

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Predicted Price", f"${predicted_price:.2f}")
            change = predicted_price - current_price
            percent_change = (change / current_price) * 100
            col3.metric("Change", f"${change:.2f}", delta=f"{percent_change:.2f}%")

            # Recommendation
            if "upward" in recommendation:
                st.success(recommendation)
            elif "Slight" in recommendation:
                st.warning(recommendation)
            else:
                st.error(recommendation)

            st.markdown("---")

            # Moving averages
            st.subheader("📊 Moving Averages")
            st.write(f"**MA10 (10-day):** {df['MA10'].iloc[-1]:.2f}")
            st.write(f"**MA50 (50-day):** {df['MA50'].iloc[-1]:.2f}")
            st.write(f"**MA100 (100-day):** {df['MA100'].iloc[-1]:.2f}")

            # Model evaluation
            st.subheader("📈 Model Evaluation Metrics")
            mae = mean_absolute_error(y_true, predictions)
            mse = mean_squared_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)

            st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**R-squared (R²):** {r2:.2f}")

            # Tabs for plots
            tab1, tab2 = st.tabs(["📉 Price + Averages", "📍 Actual vs Predicted"])
            with tab1:
                plot_company_graph(df)
            with tab2:
                plot_actual_vs_predicted(df, predicted_price, predictions)

        except ValueError as e:
            st.error(f"❌ Error: {e}")
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")
    else:
        st.info("ℹ️ Please enter a stock ticker symbol to get started.")

# Run app
if __name__ == "__main__":
    main()
