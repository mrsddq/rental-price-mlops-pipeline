import pandas as pd


def add_price_per_sqft(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["price_per_sqft"] = enriched["price"] / enriched["sqft"]
    return enriched


def add_room_density(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["sqft_per_room"] = enriched["sqft"] / enriched["rooms"]
    return enriched


def enrich_housing_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_price_per_sqft(frame)
    return add_room_density(enriched)
