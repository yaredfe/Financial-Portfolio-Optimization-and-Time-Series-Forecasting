import yfinance as yf


def download_data(start_date="2015-01-01",end_date="2024-10-31"):
    # Define the assets and date range

    TSLA=yf.download("TSLA",start=start_date,end=end_date)
    BND=yf.download("BND",start=start_date,end=end_date)
    SPY=yf.download("SPY",start=start_date,end=end_date)
    
    data=[TSLA,BND,SPY]
    for assets in data:
        assets.columns = assets.columns.get_level_values(0)
        if "price" in assets.columns:
            assets=assets.drop(columns="price",axis=1)
        assets.columns.name = None
    
    return TSLA,BND,SPY