import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load stock price data
# ----------------------------
def load_stock(ticker="MRNA"):
    df = pd.read_csv(f"data/{ticker}.csv", parse_dates=[0])
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # Ensure prices are numeric
    for col in ["Close", "Adj Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ----------------------------
# Align event date to nearest trading day
# ----------------------------
def align_event_date(df, event_date):
    event_date = pd.to_datetime(event_date)
    # find closest date in 'Date' column
    closest_date = df.iloc[(df['Date'] - event_date).abs().argsort()[:1]]['Date'].values[0]
    return pd.to_datetime(closest_date)

# ----------------------------
# Run event study
# ----------------------------
def run_event_study(ticker, event_date, event_label="", window=10):
    df = load_stock(ticker)
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["Return"] = df[price_col].pct_change()

    original_date = pd.to_datetime(event_date)
    aligned_date = align_event_date(df, event_date)

    # Select window using boolean mask
    start_date = aligned_date - pd.Timedelta(days=window)
    end_date = aligned_date + pd.Timedelta(days=window)
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    event_window = df.loc[mask]

    if event_window.empty:
        print(f"⚠️ No data found for window around {aligned_date.date()}. Skipping event.")
        return

    # Plot cumulative return
    plt.figure(figsize=(10,5))
    plt.plot(event_window['Date'], event_window["Return"].cumsum(), label=f"{ticker} cumulative return")
    plt.axvline(aligned_date, linestyle="--", color="red", label="Event")
    plt.title(f"Event Study: {ticker} around {aligned_date.date()}\n{event_label}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    
    # Save file with both original and aligned dates
    filename = f"data/{ticker}_event_{original_date.date()}_aligned_{aligned_date.date()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved plot: {filename}")

    
# ----------------------------
# Load events CSV
# ----------------------------
def load_events(events_file="data/events.csv"):
    events = pd.read_csv(events_file)
    events["date"] = pd.to_datetime(events["date"])
    return events

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    events = load_events()
    for _, row in events.iterrows():
        run_event_study(row["ticker"], row["date"], row["event"])


