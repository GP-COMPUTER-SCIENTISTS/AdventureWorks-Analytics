"""Carga de datos crudos y construcción de variables RFM por cliente."""
import numpy as np
import pandas as pd

from src.db import ROOT

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

INCOME_ORDER = ["0-25000", "25001-50000", "50001-75000", "75001-100000", "greater than 100000"]
EDUCATION_ORDER = ["Partial High School", "High School", "Partial College", "Bachelors", "Graduate Degree"]


def load_raw():
    """Devuelve (orders, order_details, demographics, sales_reasons) limpios."""
    orders = pd.read_parquet(RAW / "orders.parquet")
    details = pd.read_parquet(RAW / "order_details.parquet")
    demo = pd.read_parquet(RAW / "demographics.parquet")
    reasons = pd.read_parquet(RAW / "sales_reasons.parquet")

    orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])
    orders["Channel"] = np.where(orders["OnlineOrderFlag"] == 1, "Online", "Vendedor")

    for col in ["Education", "Occupation", "YearlyIncome", "Gender", "MaritalStatus"]:
        demo[col] = demo[col].str.strip()
    demo["YearlyIncome"] = pd.Categorical(demo["YearlyIncome"], INCOME_ORDER, ordered=True)
    demo["Education"] = pd.Categorical(demo["Education"], EDUCATION_ORDER, ordered=True)

    return orders, details, demo, reasons


def snapshot_date(orders: pd.DataFrame) -> pd.Timestamp:
    """Fecha de referencia para la recencia: el día siguiente al último pedido."""
    return orders["OrderDate"].max().normalize() + pd.Timedelta(days=1)


def build_customer_features(orders, details, demo) -> pd.DataFrame:
    """Una fila por cliente individual con RFM + variables de perfil."""
    ind = orders[orders["CustomerType"] == "Individual"].copy()
    snap = snapshot_date(orders)

    rfm = ind.groupby("CustomerID").agg(
        FirstOrder=("OrderDate", "min"),
        LastOrder=("OrderDate", "max"),
        Frequency=("SalesOrderID", "nunique"),
        Monetary=("SubTotal", "sum"),
    )
    # Territorio más frecuente de cada cliente (vectorizado, sin lambdas por grupo)
    terr = (ind.groupby(["CustomerID", "Territory", "TerritoryGroup"]).size()
            .reset_index(name="n").sort_values("n", ascending=False)
            .drop_duplicates("CustomerID").set_index("CustomerID")[["Territory", "TerritoryGroup"]])
    rfm = rfm.join(terr)
    rfm["Recency"] = (snap - rfm["LastOrder"]).dt.days
    rfm["Tenure"] = (snap - rfm["FirstOrder"]).dt.days
    rfm["AvgTicket"] = rfm["Monetary"] / rfm["Frequency"]

    # Mezcla de categorías por cliente (% del gasto)
    lines = details.merge(ind[["SalesOrderID", "CustomerID"]], on="SalesOrderID")
    cat = lines.pivot_table(index="CustomerID", columns="Category", values="LineTotal",
                            aggfunc="sum", fill_value=0)
    cat = cat.div(cat.sum(axis=1), axis=0).add_prefix("Share_")
    items = lines.groupby("CustomerID")["OrderQty"].sum().rename("Items")

    feats = rfm.join(cat).join(items)

    d = demo.set_index("CustomerID")
    d["Age"] = ((snap - d["BirthDate"]).dt.days / 365.25).astype(int)
    feats = feats.join(d[["Age", "Gender", "MaritalStatus", "YearlyIncome", "Education",
                          "Occupation", "HomeOwnerFlag", "NumberCarsOwned", "TotalChildren"]])
    return feats.reset_index()


def main():
    orders, details, demo, _ = load_raw()
    feats = build_customer_features(orders, details, demo)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(PROCESSED / "customer_features.parquet", index=False)
    print(f"{len(feats):,} clientes -> data/processed/customer_features.parquet")
    print(feats[["Recency", "Frequency", "Monetary", "AvgTicket"]].describe().round(1))


if __name__ == "__main__":
    main()
