import yfinance as yf

ticker = "MRNA"

def fetch_data(start="2020-01-01", end="2023-12-31"):
    df = yf.download(ticker, start=start, end=end)
    # Reset index so 'Date' becomes a proper column
    df.reset_index(inplace=True)
    df.to_csv(f"data/{ticker}.csv", index=False)  # no extra index column
    print(f"✅ Moderna stock data saved to data/{ticker}.csv")

if __name__ == "__main__":
    fetch_data()

