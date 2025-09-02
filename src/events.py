import pandas as pd

def load_events():
    return pd.read_csv("data/events.csv", parse_dates=["date"])

if __name__ == "__main__":
    events = load_events()
    print("✅ Loaded events:")
    print(events)
