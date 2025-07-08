

## 📈 Stock Price Predictor with Smart Recommendations

A **Streamlit web app** that predicts the next day's stock prices using historical data and **Random Forest regression**, with smart recommendations like "Buy", "Hold", or "Sell". It also displays **moving averages**, price trends, and model evaluation metrics.

---

### 🔧 Features

* 📊 Predict next-day stock prices using machine learning
* 🏦 Fetch real-time stock data via **yFinance**
* 📈 Plot closing price with **MA10**, **MA50**, **MA100**
* 💡 Get dynamic **buy/sell/hold recommendations**
* 🔍 Search from a company list (`company_list.csv`)
* 🎨 Clean Streamlit UI with charts and evaluation metrics

---

### 🛠️ Technologies Used

* Python
* Streamlit
* yFinance
* scikit-learn (Random Forest Regressor)
* Matplotlib, Pandas
* HTML/CSS for Streamlit styling

---

### 🚀 Getting Started

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Stock-Prediction-App.git
cd Stock-Prediction-App
```

#### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

#### 3. Run the App

```bash
streamlit run search.py
```

---

### 📂 File Structure
```
├── search.py                # Main Streamlit application
├── company_list.csv         # List of companies and ticker symbols
├── README.md                # Project description
└── requirements.txt         # Python dependencies
```
---

### 📝 Example Input

> Enter a ticker symbol like `AAPL`, `TSLA`, or `MSFT` in the app input box.
> You’ll see:

* Current and predicted prices
* Moving averages
* Buy/Sell/Hold recommendation
* Evaluation metrics (MAE, MSE, R²)

---

### 📌 Note

Make sure your system has a stable internet connection — the app fetches live stock data and company info.
