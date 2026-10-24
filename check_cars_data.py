
import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent / "cars.csv"

if not file_path.exists():
    print("ERROR: cars.csv was not found in the project folder.")
else:
    df = pd.read_csv(file_path)

    print("\n===== CARWISE DATA AUDIT =====")
    print("Total car records:", len(df))
    print("Total columns:", len(df.columns))
    print("\nColumns:", list(df.columns))

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATE RECORDS =====")
    print("Exact duplicate rows:", df.duplicated().sum())

    if "name" in df.columns:
        print(
            "Duplicate car names:",
            df["name"].astype(str).str.strip().str.lower().duplicated().sum()
        )

    if "price" in df.columns:
        prices = pd.to_numeric(df["price"], errors="coerce")
        print("\n===== PRICE CHECK =====")
        print("Invalid or missing prices:", prices.isna().sum())
        print("Zero or negative prices:", (prices <= 0).sum())
        print("Minimum price:", prices.min())
        print("Maximum price:", prices.max())

    if "mileage" in df.columns:
        mileage = pd.to_numeric(df["mileage"], errors="coerce")
        print("\n===== MILEAGE CHECK =====")
        print("Invalid or missing mileage:", mileage.isna().sum())
        print("Zero or negative mileage:", (mileage <= 0).sum())

    print("\n===== SAMPLE RECORDS =====")
    print(df.head(5).to_string(index=False))

    print("\nAudit complete. Review any warnings above.")
