import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import ta
from tabulate import tabulate
import requests
from typing import Dict, Tuple, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class StockAnalyzer:
    def __init__(self, ticker: str, benchmark_ticker: str = "^GSPC"):
        self.ticker = ticker.upper()
        self.benchmark_ticker = benchmark_ticker
        self.stock = yf.Ticker(self.ticker)
        self.benchmark = yf.Ticker(benchmark_ticker)
        self.data = None
        self.benchmark_data = None
        self.info = None
        self.financials = None
        self.balance_sheet = None
        self.cashflow = None
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=5*365 + 30)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.newsapi_key = "3f8f93bbf44a4ed19778bce70301776c"  # Replace with your NewsAPI key
        self.model = None
        self.scaler = StandardScaler()

    def get_data(self) -> bool:
        try:
            self.data = self.stock.history(start=self.start_date, end=self.end_date, interval="1d")
            self.benchmark_data = self.benchmark.history(start=self.start_date, end=self.end_date, interval="1d")
            if self.data.empty or self.benchmark_data.empty:
                raise ValueError("No historical data available.")
            self.info = self.stock.info
            self.financials = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.cashflow = self.stock.cash_flow
            return True
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {str(e)}")
            return False

    def compute_fundamentals(self) -> Dict:
        ratios = {}
        try:
            ratios['P/E'] = self.info.get('trailingPE', np.nan)
            ratios['PEG'] = self.info.get('pegRatio', np.nan)
            ratios['P/B'] = self.info.get('priceToBook', np.nan)
            ratios['EV/EBITDA'] = self.info.get('enterpriseToEbitda', np.nan)
            print("Financials Keys:", self.financials.index.tolist() if not self.financials.empty else "Empty")
            print("Balance Sheet Keys:", self.balance_sheet.index.tolist() if not self.balance_sheet.empty else "Empty")
            if not self.financials.empty and not self.balance_sheet.empty:
                net_income = self.financials.loc['Net Income'].iloc[0] if 'Net Income' in self.financials.index else np.nan
                revenue = self.financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in self.financials.index else np.nan
                equity = self.balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in self.balance_sheet.index else np.nan
                if pd.isna(equity):
                    for key in ['Total Equity', 'Common Stock Equity', 'Shareholders Equity']:
                        if key in self.balance_sheet.index:
                            equity = self.balance_sheet.loc[key].iloc[0]
                            break
                assets = self.balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in self.balance_sheet.index else np.nan
                ratios['Net Margin'] = (net_income / revenue * 100) if revenue != 0 and not pd.isna([net_income, revenue]).any() else np.nan
                ratios['ROE'] = (net_income / equity * 100) if equity != 0 and not pd.isna([net_income, equity]).any() else np.nan
                ratios['ROA'] = (net_income / assets * 100) if assets != 0 and not pd.isna([net_income, assets]).any() else np.nan
            if not self.balance_sheet.empty:
                debt = self.balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in self.balance_sheet.index else np.nan
                if pd.isna(debt):
                    for key in ['Long Term Debt', 'Short Long Term Debt', 'Total Liabilities']:
                        if key in self.balance_sheet.index:
                            debt = self.balance_sheet.loc[key].iloc[0]
                            break
                equity = self.balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in self.balance_sheet.index else np.nan
                if pd.isna(equity):
                    for key in ['Total Equity', 'Common Stock Equity', 'Shareholders Equity']:
                        if key in self.balance_sheet.index:
                            equity = self.balance_sheet.loc[key].iloc[0]
                            break
                current_assets = self.balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in self.balance_sheet.index else np.nan
                current_liabilities = self.balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in self.balance_sheet.index else np.nan
                ratios['Debt/Equity'] = debt / equity if equity != 0 and not pd.isna([debt, equity]).any() else np.nan
                ratios['Current Ratio'] = current_assets / current_liabilities if current_liabilities != 0 and not pd.isna([current_assets, current_liabilities]).any() else np.nan
            if len(self.financials.columns) >= 3:
                rev_start = self.financials.loc['Total Revenue'].iloc[-1] if 'Total Revenue' in self.financials.index else np.nan
                rev_end = self.financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in self.financials.index else np.nan
                earnings_start = self.financials.loc['Net Income'].iloc[-1] if 'Net Income' in self.financials.index else np.nan
                earnings_end = self.financials.loc['Net Income'].iloc[0] if 'Net Income' in self.financials.index else np.nan
                ratios['Revenue CAGR'] = ((rev_end / rev_start) ** (1/3) - 1) * 100 if rev_start != 0 and not pd.isna([rev_start, rev_end]).any() else np.nan
                ratios['Earnings CAGR'] = ((earnings_end / earnings_start) ** (1/3) - 1) * 100 if earnings_start != 0 and not pd.isna([earnings_start, earnings_end]).any() else np.nan
        except Exception as e:
            print(f"Error computing fundamentals: {str(e)}")
        return ratios

    def compute_technicals(self) -> Dict:
        df = self.data.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        return df

    def compare_to_benchmark(self) -> Dict:
        metrics = {}
        try:
            df = self.data['Close'].pct_change().dropna()
            bench_df = self.benchmark_data['Close'].pct_change().dropna()
            aligned = pd.concat([df, bench_df], axis=1).dropna()
            cov = aligned.cov().iloc[0, 1]
            var = aligned.iloc[:, 1].var()
            metrics['Beta'] = cov / var if var != 0 else np.nan
            metrics['Correlation'] = aligned.corr().iloc[0, 1]
        except Exception as e:
            print(f"Error in benchmark comparison: {str(e)}")
        return metrics

    def get_news_sentiment(self) -> str:
        try:
            company_name = self.info.get('longName', self.ticker).replace(' Inc.', '').replace(' Corporation', '')
            url = (f"https://newsapi.org/v2/everything?q={company_name}+stock+OR+{company_name}+technology+-crypto"
                f"&language=en&sortBy=publishedAt&apiKey={self.newsapi_key}")
            response = requests.get(url)
            response.raise_for_status()
            news = response.json().get('articles', [])
            print("Raw news data (NewsAPI):", [article.get('title', 'No title') for article in news[:10]])
            if not news:
                return "No recent news available from NewsAPI."
            sentiments = []
            news_summaries = []
            for article in news[:10]:
                title = article.get('title', 'No title')
                description = article.get('description', '')
                url = article.get('url', '#')
                if not title and not description:
                    continue
                text = title + " " + (description or '')
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                compound_score = sentiment['compound']
                sentiment_label = 'Positive' if compound_score > 0.05 else 'Negative' if compound_score < -0.05 else 'Neutral'
                sentiments.append(sentiment_label)
                desc_snippet = (description[:150] + '...') if description and len(description) > 150 else description
                news_summaries.append(f"- **[{title}]({url})** ({sentiment_label}): {desc_snippet or 'No description available.'}")
            if not sentiments:
                return "No valid news headlines or descriptions available for sentiment analysis."
            sentiment_counts = pd.Series(sentiments).value_counts()
            total = len(sentiments)
            positive = sentiment_counts.get('Positive', 0)
            negative = sentiment_counts.get('Negative', 0)
            neutral = sentiment_counts.get('Neutral', 0)
            if positive > negative and positive >= neutral:
                overall = "predominantly positive"
            elif negative > positive and negative >= neutral:
                overall = "predominantly negative"
            else:
                overall = "mixed or neutral"
            news_section = "\n".join(news_summaries)
            return (f"Analyzed {total} recent news items: {positive} positive, "
                    f"{negative} negative, {neutral} neutral. Overall sentiment is {overall}.\n\n"
                    f"### News Summaries\n{news_section}")
        except Exception as e:
            return f"Error analyzing news sentiment with NewsAPI: {str(e)}"
    def train_price_trend_model(self, df: pd.DataFrame) -> None:
        try:
            # Prepare features
            features = df[['RSI', 'MACD', 'MACD_Signal', 'ATR', 'OBV', 'VWAP', 'Volume']].copy()
            features['SMA_20_50'] = df['SMA_20'] / df['SMA_50']
            features['SMA_50_200'] = df['SMA_50'] / df['SMA_200']
            
            # Create target: 1 if price increases after 5 days, 0 otherwise
            df['Future_Price'] = df['Close'].shift(-5)
            df['Target'] = (df['Future_Price'] > df['Close']).astype(int)
            
            # Drop rows with NaN values
            data = pd.concat([features, df['Target']], axis=1).dropna()
            
            # Split features and target
            X = data.drop('Target', axis=1)
            y = data['Target']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Train Random Forest Classifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate model (optional, for logging)
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            print(f"Model Training Accuracy: {train_score:.2f}, Test Accuracy: {test_score:.2f}")
            
        except Exception as e:
            print(f"Error training price trend model: {str(e)}")
            self.model = None

    def predict_price_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        if self.model is None:
            return "No model trained.", 0.0
        try:
            # Prepare features for the latest day
            features = df[['RSI', 'MACD', 'MACD_Signal', 'ATR', 'OBV', 'VWAP', 'Volume']].copy()
            features['SMA_20_50'] = df['SMA_20'] / df['SMA_50']
            features['SMA_50_200'] = df['SMA_50'] / df['SMA_200']
            latest_features = features.iloc[-1:].dropna()
            if latest_features.empty:
                return "Insufficient data for prediction.", 0.0
            latest_scaled = self.scaler.transform(latest_features)
            
            # Predict trend
            prediction = self.model.predict(latest_scaled)[0]
            confidence = self.model.predict_proba(latest_scaled)[0][prediction]
            
            trend = "Up" if prediction == 1 else "Down"
            return f"Predicted 5-day price trend: {trend}", confidence
        except Exception as e:
            print(f"Error predicting price trend: {str(e)}")
            return "Prediction failed.", 0.0

    def plot_technicals(self, df: pd.DataFrame):
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(df.index, df['Close'], label='Close Price')
        ax1.plot(df.index, df['SMA_20'], label='SMA 20')
        ax1.plot(df.index, df['SMA_50'], label='SMA 50')
        ax1.plot(df.index, df['SMA_200'], label='SMA 200')
        ax1.fill_between(df.index, df['BB_High'], df['BB_Low'], alpha=0.1, color='gray', label='Bollinger Bands')
        ax1.set_title(f"{self.ticker} Price and Technicals")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
        ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
        ax2.set_title("RSI")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        import os
        output_path = os.path.join("static", f"{self.ticker}_technicals.png")
        plt.savefig(output_path)
        plt.close()

    def generate_report(self) -> str:
        if not self.get_data():
            return f"Error: Unable to fetch data for {self.ticker}."
        df = self.compute_technicals()
        fundamentals = self.compute_fundamentals()
        benchmark_metrics = self.compare_to_benchmark()
        news_sentiment = self.get_news_sentiment()
        self.plot_technicals(df)
        
        # Train and predict price trend
        self.train_price_trend_model(df)
        price_trend, confidence = self.predict_price_trend(df)
        
        latest_price = df['Close'].iloc[-1]
        price_change_1y = (latest_price / df['Close'].iloc[-252] - 1) * 100 if len(df) >= 252 else np.nan
        trend = "bullish" if df['Close'].iloc[-1] > df['SMA_200'].iloc[-1] else "bearish"
        rsi = df['RSI'].iloc[-1]
        rsi_status = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        macd_status = "bullish" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "bearish"
        pe_status = "overvalued" if fundamentals.get('P/E', np.nan) > 30 else "undervalued" if fundamentals.get('P/E', np.nan) < 15 else "fairly valued"
        debt_status = "high" if fundamentals.get('Debt/Equity', np.nan) > 1.5 else "manageable"
        dividend_yield = self.info.get('dividendYield', np.nan) * 100
        if not pd.isna(dividend_yield) and dividend_yield > 10:
            dividend_yield /= 100
        red_flags = []
        if fundamentals.get('Revenue CAGR', np.nan) < 0:
            red_flags.append("Declining revenue growth")
        if fundamentals.get('Debt/Equity', np.nan) > 2:
            red_flags.append("High leverage")
        if df['Close'].iloc[-1] < df['SMA_200'].iloc[-1]:
            red_flags.append("Bearish price trend")
        
        # Investment Recommendation Logic
        score = 0
        reasons = []
        
        # Technical Signals
        if rsi_status == "neutral":
            score += 1
            reasons.append("Neutral RSI indicates stable momentum.")
        elif rsi_status == "overbought":
            score -= 1
            reasons.append("Overbought RSI suggests caution.")
        elif rsi_status == "oversold":
            score += 1
            reasons.append("Oversold RSI suggests potential opportunity.")
            
        if macd_status == "bullish":
            score += 1
            reasons.append("Bullish MACD crossover is positive.")
        else:
            score -= 1
            reasons.append("Bearish MACD crossover indicates weakness.")
            
        if trend == "bullish":
            score += 1
            reasons.append("Price above 200-day SMA indicates bullish trend.")
        else:
            score -= 1
            reasons.append("Price below 200-day SMA indicates bearish trend.")
            
        # Fundamental Signals
        if not pd.isna(fundamentals.get('P/E')):
            if fundamentals['P/E'] < 15:
                score += 1
                reasons.append("Low P/E ratio suggests undervaluation.")
            elif fundamentals['P/E'] > 30:
                score -= 1
                reasons.append("High P/E ratio suggests overvaluation.")
                
        if not pd.isna(fundamentals.get('Debt/Equity')):
            if fundamentals['Debt/Equity'] < 1.5:
                score += 1
                reasons.append("Low Debt/Equity ratio indicates financial stability.")
            elif fundamentals['Debt/Equity'] > 2:
                score -= 1
                reasons.append("High Debt/Equity ratio indicates financial risk.")
                
        if not pd.isna(fundamentals.get('Revenue CAGR')):
            if fundamentals['Revenue CAGR'] > 0:
                score += 1
                reasons.append("Positive revenue growth is favorable.")
            else:
                score -= 1
                reasons.append("Negative revenue growth is a concern.")
                
        if not pd.isna(fundamentals.get('ROE')):
            if fundamentals['ROE'] > 10:
                score += 1
                reasons.append("High ROE indicates efficient use of equity.")
            elif fundamentals['ROE'] < 0:
                score -= 1
                reasons.append("Negative ROE indicates poor profitability.")
                
        # News Sentiment
        if "predominantly positive" in news_sentiment:
            score += 1
            reasons.append("Positive news sentiment supports investment.")
        elif "predominantly negative" in news_sentiment:
            score -= 1
            reasons.append("Negative news sentiment suggests caution.")
            
        # AI Prediction Signal
        if "Up" in price_trend:
            score += 1
            reasons.append("AI model predicts upward price trend.")
        elif "Down" in price_trend:
            score -= 1
            reasons.append("AI model predicts downward price trend.")
            
        # Determine Recommendation
        if score >= 3:
            recommendation = "Buy"
            rec_summary = "The stock shows strong technical, fundamental, and AI-predicted signals, suggesting a favorable investment opportunity."
        elif score >= 1:
            recommendation = "Hold"
            rec_summary = "The stock has mixed signals, indicating it may be prudent to hold existing positions."
        else:
            recommendation = "Sell"
            rec_summary = "The stock exhibits weak or negative signals, suggesting caution or potential divestment."
        
        report = f"""
# Stock Analysis Report: {self.ticker}
Generated on: {datetime.now().strftime('%Y-%m-%d')}

## Overview
- **Company**: {self.info.get('longName', 'N/A')}
- **Sector**: {self.info.get('sector', 'N/A')}
- **Industry**: {self.info.get('industry', 'N/A')}
- **Market Cap**: ${self.info.get('marketCap', 'N/A'):,}
- **Dividend Yield**: {dividend_yield:.2f}% if available
- **Latest Close Price**: ${latest_price:.2f}
- **1-Year Price Change**: {price_change_1y:.2f}% if available

## Price Trend
The stock is in a {trend} trend, trading {"above" if latest_price > df['SMA_200'].iloc[-1] else "below"} its 200-day SMA.

## Technical Analysis
- **RSI**: {rsi:.2f} ({rsi_status})
- **MACD**: {macd_status} crossover
- **ATR**: {df['ATR'].iloc[-1]:.2f} (volatility measure)
- **Price vs. Bollinger Bands**: {"near upper band" if latest_price > df['BB_High'].iloc[-1]*0.95 else "near lower band" if latest_price < df['BB_Low'].iloc[-1]*1.05 else "within bands"}

## Fundamental Analysis
- **P/E Ratio**: {fundamentals.get('P/E', 'N/A'):.2f}
- **PEG Ratio**: {fundamentals.get('PEG', 'N/A'):.2f}
- **P/B Ratio**: {fundamentals.get('P/B', 'N/A'):.2f}
- **EV/EBITDA**: {fundamentals.get('EV/EBITDA', 'N/A'):.2f}
- **ROE**: {fundamentals.get('ROE', 'N/A'):.2f}%
- **Net Margin**: {fundamentals.get('Net Margin', 'N/A'):.2f}%
- **Debt/Equity**: {fundamentals.get('Debt/Equity', 'N/A'):.2f} ({debt_status})
- **Revenue CAGR (3Y)**: {fundamentals.get('Revenue CAGR', 'N/A'):.2f}%
- **Earnings CAGR (3Y)**: {fundamentals.get('Earnings CAGR', 'N/A'):.2f}%

## Benchmark Comparison
- **Beta**: {benchmark_metrics.get('Beta', 'N/A'):.2f}
- **Correlation with {self.benchmark_ticker}**: {benchmark_metrics.get('Correlation', 'N/A'):.2f}

## News Sentiment
{news_sentiment}

## Price Trend Prediction (AI Model)
- **{price_trend} (Confidence: {confidence:.2%})**

## Red Flags
{', '.join(red_flags) if red_flags else 'None identified.'}

## Insights
- The stock's {rsi_status} RSI suggests {"caution for buyers" if rsi > 70 else "potential opportunity" if rsi < 30 else "stable momentum"}.
- {"High" if fundamentals.get('Debt/Equity', np.nan) > 1.5 else "Low"} leverage warrants attention for risk assessment.
- Revenue growth of {fundamentals.get('Revenue CAGR', 'N/A'):.2f}% indicates {"strong" if fundamentals.get('Revenue CAGR', np.nan) > 0 else "weak" if fundamentals.get('Revenue CAGR', np.nan) < 0 else "unknown"} performance.
- See {self.ticker}_technicals.png for price and technical charts.

## Recommendation
- **Decision**: {recommendation}
- **Rationale**: {rec_summary}
- **Key Factors**:
  {chr(10).join([f"  - {reason}" for reason in reasons])}

**Disclaimer**: This is not financial advice. Conduct your own research.
"""
        table_data = [[k, f"{v:.2f}" if isinstance(v, (int, float)) else str(v)] for k, v in fundamentals.items()]
        report += "\n\n## Fundamentals Table\n"
        report += tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid")
        return report

def analyze_stock(ticker: str):
    analyzer = StockAnalyzer(ticker)
    report = analyzer.generate_report()
    print(report)
    with open(f"{ticker}_report.md", "w") as f:
        f.write(report)
    print(f"Report saved as {ticker}_report.md")

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    analyze_stock(ticker)