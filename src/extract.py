"""Extrae del módulo Sales de AdventureWorks2022 los datos necesarios para el
EDA y el clustering, y los guarda en data/raw/*.parquet.

Uso:
    python -m src.extract
"""
from src.db import ROOT, read_sql

RAW = ROOT / "data" / "raw"

QUERIES = {
    # Cabecera de pedidos + territorio + tipo de cliente
    "orders": """
        SELECT
            h.SalesOrderID, h.OrderDate, h.DueDate, h.ShipDate, h.Status,
            CAST(h.OnlineOrderFlag AS int) AS OnlineOrderFlag,
            h.CustomerID, h.SalesPersonID, h.TerritoryID,
            t.Name AS Territory, t.CountryRegionCode AS Country, t.[Group] AS TerritoryGroup,
            h.SubTotal, h.TaxAmt, h.Freight, h.TotalDue,
            CASE WHEN c.StoreID IS NULL THEN 'Individual' ELSE 'Store' END AS CustomerType
        FROM Sales.SalesOrderHeader h
        JOIN Sales.Customer c        ON c.CustomerID = h.CustomerID
        LEFT JOIN Sales.SalesTerritory t ON t.TerritoryID = h.TerritoryID
    """,
    # Detalle de pedidos + producto + categoría
    "order_details": """
        SELECT
            d.SalesOrderID, d.SalesOrderDetailID, d.ProductID, d.OrderQty,
            d.UnitPrice, d.UnitPriceDiscount, d.LineTotal, d.SpecialOfferID,
            p.Name AS Product, p.StandardCost, p.ListPrice,
            ISNULL(ps.Name, 'Sin subcategoría') AS Subcategory,
            ISNULL(pc.Name, 'Sin categoría')    AS Category
        FROM Sales.SalesOrderDetail d
        JOIN Production.Product p                ON p.ProductID = d.ProductID
        LEFT JOIN Production.ProductSubcategory ps ON ps.ProductSubcategoryID = p.ProductSubcategoryID
        LEFT JOIN Production.ProductCategory pc    ON pc.ProductCategoryID = ps.ProductCategoryID
    """,
    # Demografía de clientes individuales (sin nombres ni correos)
    "demographics": """
        SELECT
            c.CustomerID, d.TotalPurchaseYTD, d.DateFirstPurchase, d.BirthDate,
            d.MaritalStatus, d.YearlyIncome, d.Gender, d.TotalChildren,
            d.NumberChildrenAtHome, d.Education, d.Occupation,
            d.HomeOwnerFlag, d.NumberCarsOwned
        FROM Sales.Customer c
        JOIN Sales.vPersonDemographics d ON d.BusinessEntityID = c.PersonID
        WHERE c.StoreID IS NULL
    """,
    "sales_reasons": """
        SELECT hr.SalesOrderID, r.Name AS Reason, r.ReasonType
        FROM Sales.SalesOrderHeaderSalesReason hr
        JOIN Sales.SalesReason r ON r.SalesReasonID = hr.SalesReasonID
    """,
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, sql in QUERIES.items():
        df = read_sql(sql)
        df.to_parquet(RAW / f"{name}.parquet", index=False)
        print(f"{name:15s} {len(df):>8,} filas -> data/raw/{name}.parquet")


if __name__ == "__main__":
    main()
