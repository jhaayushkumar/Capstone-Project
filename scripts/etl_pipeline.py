"""End-to-end ETL pipeline for the Olist capstone project.

This script reads the raw CSV extracts from data/raw, applies the project
cleaning rules, writes standardized tables to data/processed, and builds a
analysis-ready master dataset for Tableau and reporting.

The pipeline keeps the original raw files untouched and only writes new outputs
to the processed directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("etl_pipeline")
DATE_COLUMNS_ORDERS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
DATE_COLUMNS_REVIEWS = ["review_creation_date", "review_answer_timestamp"]
NUMERIC_PRODUCT_COLUMNS = [
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]
RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "translation": "product_category_name_translation.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned capstone datasets.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the project root.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override the raw data directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override the processed data directory.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def resolve_paths(project_root: Path, raw_dir: Path | None, processed_dir: Path | None) -> tuple[Path, Path]:
    resolved_raw = raw_dir or project_root / "data" / "raw"
    resolved_processed = processed_dir or project_root / "data" / "processed"

    if not resolved_raw.exists():
        raise FileNotFoundError(f"Raw data directory not found: {resolved_raw}")

    resolved_processed.mkdir(parents=True, exist_ok=True)
    return resolved_raw, resolved_processed


def load_csv(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading %s", path.name)
    return pd.read_csv(path)


def standardize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def standardize_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def ensure_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def safe_mode(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode()
    if modes.empty:
        return non_null.iloc[0]
    return modes.iloc[0]


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_frame_columns(df).copy()
    ensure_columns(df, ["order_id", "customer_id"], "orders")

    for column in DATE_COLUMNS_ORDERS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if {"order_delivered_customer_date", "order_purchase_timestamp"}.issubset(df.columns):
        delivery_time = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
        df["delivery_time_days"] = delivery_time.where(delivery_time.ge(0))
    else:
        df["delivery_time_days"] = np.nan

    if "order_delivered_customer_date" in df.columns:
        df["is_delivered_clean"] = df["order_delivered_customer_date"].notna().astype(int)
    else:
        df["is_delivered_clean"] = 0

    df["order_stage"] = np.select(
        [
            df.get("order_delivered_customer_date", pd.Series(index=df.index)).notna(),
            df.get("order_delivered_carrier_date", pd.Series(index=df.index)).notna(),
            df.get("order_approved_at", pd.Series(index=df.index)).notna(),
        ],
        ["delivered", "shipped_not_delivered", "approved_not_shipped"],
        default="created_only",
    )
    df["is_completed_order"] = (df["order_stage"] == "delivered").astype(int)
    df["is_late_delivery"] = (
        df.get("order_delivered_customer_date", pd.Series(index=df.index))
        > df.get("order_estimated_delivery_date", pd.Series(index=df.index))
    ).fillna(False).astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def clean_items(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_frame_columns(df).copy()
    ensure_columns(df, ["order_id", "product_id", "price", "freight_value"], "items")

    if "shipping_limit_date" in df.columns:
        df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")

    numeric_columns = ["order_item_id", "price", "freight_value"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["item_total_value"] = df["price"].fillna(0) + df["freight_value"].fillna(0)
    threshold = df["price"].quantile(0.95)
    df["is_expensive_item"] = df["price"].gt(threshold).fillna(False).astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_frame_columns(df).copy()
    ensure_columns(df, ["product_id", "product_category_name"], "products")

    df["product_category_name"] = df["product_category_name"].fillna("unknown").astype("string").str.strip().str.lower()
    for column in NUMERIC_PRODUCT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].fillna(df[column].median())

    if {"product_length_cm", "product_height_cm", "product_width_cm"}.issubset(df.columns):
        df["product_volume_cm3"] = (
            df["product_length_cm"].fillna(0)
            * df["product_height_cm"].fillna(0)
            * df["product_width_cm"].fillna(0)
        )
    else:
        df["product_volume_cm3"] = np.nan

    return df.drop_duplicates().reset_index(drop=True)


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_frame_columns(df).copy()
    ensure_columns(df, ["customer_id", "customer_city", "customer_state"], "customers")
    df = df.drop_duplicates().reset_index(drop=True)
    df["customer_city"] = standardize_text(df["customer_city"])
    df["customer_state"] = standardize_text(df["customer_state"])
    df["state_city"] = df["customer_state"].fillna("") + "_" + df["customer_city"].fillna("")
    return df


def clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_frame_columns(df).copy()
    ensure_columns(df, ["order_id", "payment_type", "payment_installments", "payment_value"], "payments")

    df["payment_installments"] = pd.to_numeric(df["payment_installments"], errors="coerce")
    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce")
    df["payment_value_per_installment"] = df["payment_value"] / df["payment_installments"].replace(0, np.nan)
    df["payment_value_per_installment"] = df["payment_value_per_installment"].fillna(df["payment_value"])
    df["payment_type"] = standardize_text(df["payment_type"])
    df["is_credit_card"] = df["payment_type"].eq("credit_card").astype(int)
    df["is_full_payment"] = df["payment_installments"].eq(1).astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_frame_columns(df).copy()
    ensure_columns(df, ["review_id", "order_id", "review_score"], "reviews")

    for column in DATE_COLUMNS_REVIEWS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if "review_comment_title" in df.columns:
        df["review_comment_title"] = df["review_comment_title"].fillna("no_title")
    if "review_comment_message" in df.columns:
        df["review_comment_message"] = df["review_comment_message"].fillna("no_comment")

    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["sentiment"] = np.select(
        [df["review_score"].ge(4), df["review_score"].eq(3)],
        ["positive", "neutral"],
        default="negative",
    )
    df["is_negative_review"] = df["review_score"].le(2).fillna(False).astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def clean_translation(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_frame_columns(df).copy()
    if "\ufeffproduct_category_name" in df.columns:
        df = df.rename(columns={"\ufeffproduct_category_name": "product_category_name"})
    ensure_columns(df, ["product_category_name", "product_category_name_english"], "translation")
    df = df.drop_duplicates().reset_index(drop=True)
    df["product_category_name"] = standardize_text(df["product_category_name"])
    df["product_category_name_english"] = standardize_text(df["product_category_name_english"])
    return df


def clean_sellers(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_frame_columns(df).copy()
    ensure_columns(df, ["seller_id", "seller_city", "seller_state"], "sellers")
    df = df.drop_duplicates().reset_index(drop=True)
    df["seller_city"] = standardize_text(df["seller_city"])
    df["seller_state"] = standardize_text(df["seller_state"])
    df["state_city"] = df["seller_state"].fillna("") + "_" + df["seller_city"].fillna("")
    return df


def clean_geolocation(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_frame_columns(df).copy()
    ensure_columns(df, ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"], "geolocation")
    df = df.drop_duplicates().reset_index(drop=True)
    for column in ["geolocation_lat", "geolocation_lng"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "geolocation_city" in df.columns:
        df["geolocation_city"] = standardize_text(df["geolocation_city"])
    if "geolocation_state" in df.columns:
        df["geolocation_state"] = standardize_text(df["geolocation_state"])
    return df


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    return (
        payments.groupby("order_id", as_index=False)
        .agg(
            total_payment_value=("payment_value", "sum"),
            payment_row_count=("payment_type", "size"),
            payment_method_count=("payment_type", pd.Series.nunique),
            dominant_payment_type=("payment_type", safe_mode),
            average_payment_installments=("payment_installments", "mean"),
            average_payment_value_per_installment=("payment_value_per_installment", "mean"),
            credit_card_payment_rows=("is_credit_card", "sum"),
            full_payment_rows=("is_full_payment", "sum"),
        )
        .reset_index(drop=True)
    )


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    return (
        reviews.groupby("order_id", as_index=False)
        .agg(
            avg_review_score=("review_score", "mean"),
            min_review_score=("review_score", "min"),
            max_review_score=("review_score", "max"),
            review_row_count=("review_id", "size"),
            negative_review_count=("is_negative_review", "sum"),
            dominant_sentiment=("sentiment", safe_mode),
        )
        .reset_index(drop=True)
    )


def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    return (
        items.groupby("order_id", as_index=False)
        .agg(
            order_item_count=("order_item_id", "count"),
            distinct_products=("product_id", pd.Series.nunique),
            distinct_sellers=("seller_id", pd.Series.nunique),
            order_items_total_value=("item_total_value", "sum"),
            order_items_total_freight=("freight_value", "sum"),
            average_item_price=("price", "mean"),
            average_item_freight=("freight_value", "mean"),
            max_item_price=("price", "max"),
            min_item_price=("price", "min"),
        )
        .reset_index(drop=True)
    )


def build_master_dataset(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame,
    translation: pd.DataFrame,
) -> pd.DataFrame:
    order_item_level = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(items, on="order_id", how="inner", suffixes=("", "_item"))
        .merge(products, on="product_id", how="left", suffixes=("", "_product"))
        .merge(translation, on="product_category_name", how="left")
    )

    payment_summary = aggregate_payments(payments)
    review_summary = aggregate_reviews(reviews)
    item_summary = aggregate_items(items)

    master = (
        order_item_level.merge(item_summary, on="order_id", how="left")
        .merge(payment_summary, on="order_id", how="left")
        .merge(review_summary, on="order_id", how="left")
    )

    if {"order_delivered_customer_date", "order_purchase_timestamp"}.issubset(master.columns):
        master["delivery_delay_vs_estimate_days"] = (
            master["order_delivered_customer_date"] - master["order_estimated_delivery_date"]
        ).dt.days
        master["actual_delivery_days"] = (
            master["order_delivered_customer_date"] - master["order_purchase_timestamp"]
        ).dt.days

    if "order_delivered_customer_date" in master.columns and "order_estimated_delivery_date" in master.columns:
        master["is_late_delivery"] = master["order_delivered_customer_date"].gt(
            master["order_estimated_delivery_date"]
        ).fillna(False).astype(int)

    if "product_category_name_english" in master.columns:
        master["product_category_name_english"] = master["product_category_name_english"].fillna("unknown")

    master = master.sort_values(["order_purchase_timestamp", "order_id"], na_position="last")
    master.columns = master.columns.str.lower()
    master = master.drop_duplicates().reset_index(drop=True)
    return master


def write_csv(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s (%s rows, %s columns)", filename, len(df), len(df.columns))


def log_frame_summary(name: str, df: pd.DataFrame) -> None:
    LOGGER.info("%s: %s rows x %s columns", name, len(df), len(df.columns))
    missing_total = int(df.isna().sum().sum())
    LOGGER.info("%s missing values after cleaning: %s", name, missing_total)


def run_pipeline(raw_dir: Path, processed_dir: Path) -> dict[str, pd.DataFrame]:
    raw_frames = {
        name: load_csv(raw_dir / filename)
        for name, filename in RAW_FILES.items()
    }

    cleaned = {
        "orders": clean_orders(raw_frames["orders"]),
        "items": clean_items(raw_frames["items"]),
        "products": clean_products(raw_frames["products"]),
        "customers": clean_customers(raw_frames["customers"]),
        "payments": clean_payments(raw_frames["payments"]),
        "reviews": clean_reviews(raw_frames["reviews"]),
        "translation": clean_translation(raw_frames["translation"]),
        "sellers": clean_sellers(raw_frames["sellers"]),
        "geolocation": clean_geolocation(raw_frames["geolocation"]),
    }

    master_dataset = build_master_dataset(
        cleaned["orders"],
        cleaned["items"],
        cleaned["products"],
        cleaned["customers"],
        cleaned["payments"],
        cleaned["reviews"],
        cleaned["translation"],
    )
    cleaned["master_dataset"] = master_dataset

    output_map = {
        "orders": "orders_clean.csv",
        "items": "items_clean.csv",
        "products": "products_clean.csv",
        "customers": "customers_clean.csv",
        "payments": "payments_clean.csv",
        "reviews": "reviews_clean.csv",
        "translation": "translation_clean.csv",
        "sellers": "sellers_clean.csv",
        "geolocation": "geolocation_clean.csv",
        "master_dataset": "master_dataset.csv",
    }

    for name, filename in output_map.items():
        write_csv(cleaned[name], processed_dir, filename)
        log_frame_summary(name, cleaned[name])

    return cleaned


def main() -> int:
    setup_logging()
    args = parse_args()
    raw_dir, processed_dir = resolve_paths(args.project_root, args.raw_dir, args.processed_dir)

    run_pipeline(raw_dir, processed_dir)
    LOGGER.info("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())