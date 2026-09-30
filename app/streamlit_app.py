"""Dashboard de AdventureWorks Sales: EDA + segmentación K-Means.

Ejecutar desde la raíz del proyecto:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering import RFM_COLS, evaluate_k, fit_kmeans, name_segments, pca_2d, prepare_matrix  # noqa: E402
from src.features import build_customer_features, load_raw  # noqa: E402
from src.viz import PALETTE, SEGMENT_COLORS, segment_order  # noqa: E402

st.set_page_config(page_title="AdventureWorks · Sales Analytics", page_icon="🚲", layout="wide")

MARKERS = {"Campeones": "circle", "Alto valor en riesgo": "square", "Leales recientes": "cross",
           "Nuevos": "triangle-up", "Bajo valor inactivos": "diamond"}


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
                      font=dict(size=13), hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False)
    return fig


def money(v):
    return f"${v / 1e6:,.1f} M" if abs(v) >= 1e6 else f"${v:,.0f}"


@st.cache_data(show_spinner="Cargando datos…")
def get_data():
    orders, details, demo, reasons = load_raw()
    feats = build_customer_features(orders, details, demo)
    return orders, details, reasons, feats


@st.cache_data(show_spinner="Evaluando k…")
def get_scores(X):
    return evaluate_k(X, range(2, 11))


@st.cache_data(show_spinner="Entrenando K-Means…")
def get_segments(feats, k):
    X, scaler = prepare_matrix(feats)
    km = fit_kmeans(X, k)
    out = feats.copy()
    out["Cluster"] = km.labels_
    names = name_segments(out)
    out["Segment"] = out["Cluster"].map(names)
    coords, pca = pca_2d(X)
    out["PC1"], out["PC2"] = coords[:, 0], coords[:, 1]
    centers = pd.DataFrame(np.expm1(scaler.inverse_transform(km.cluster_centers_)), columns=RFM_COLS)
    centers.index = centers.index.map(names)
    return out, centers, pca.explained_variance_ratio_


orders, details, reasons, feats = get_data()

st.title("AdventureWorks · Análisis de Ventas")
st.caption("Base AdventureWorks2022 · esquema Sales · mayo 2011 – junio 2014")

tab_eda, tab_k, tab_seg = st.tabs(["📊 EDA", "📐 Elección de k", "🎯 Segmentos"])

# ------------------------------------------------------------------ EDA
with tab_eda:
    f1, f2, f3 = st.columns(3)
    years = sorted(orders.OrderDate.dt.year.unique())
    yr = f1.select_slider("Años", options=years, value=(years[0], years[-1]))
    ctype = f2.multiselect("Tipo de cliente", ["Individual", "Store"], default=["Individual", "Store"])
    groups = sorted(orders.TerritoryGroup.dropna().unique())
    tgroup = f3.multiselect("Región", groups, default=groups)

    o = orders[orders.OrderDate.dt.year.between(*yr) & orders.CustomerType.isin(ctype)
               & orders.TerritoryGroup.isin(tgroup)]
    d = details[details.SalesOrderID.isin(o.SalesOrderID)]

    if o.empty:
        st.warning("No hay pedidos con esos filtros.")
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Ventas netas", money(o.SubTotal.sum()))
        k2.metric("Pedidos", f"{len(o):,}")
        k3.metric("Clientes", f"{o.CustomerID.nunique():,}")
        k4.metric("Ticket promedio", money(o.SubTotal.mean()))
        k5.metric("% online", f"{o.OnlineOrderFlag.mean():.0%}")

        # Trimestral: las tiendas registran sus pedidos en bloque a fin de mes
        m = (o.groupby([pd.Grouper(key="OrderDate", freq="QS"), "CustomerType"])["SubTotal"].sum()
             .reset_index())
        fig = px.line(m, x="OrderDate", y="SubTotal", color="CustomerType", markers=True,
                      color_discrete_map={"Store": PALETTE[0], "Individual": PALETTE[1]},
                      title="Ventas netas trimestrales por tipo de cliente",
                      labels={"OrderDate": "", "SubTotal": "Ventas (USD)", "CustomerType": ""})
        fig.update_traces(hovertemplate="$%{y:,.0f}", marker_size=8)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig), use_container_width=True)

        c1, c2 = st.columns(2)
        t = o.groupby("Territory")["SubTotal"].sum().sort_values().reset_index()
        fig = px.bar(t, x="SubTotal", y="Territory", orientation="h", title="Ventas por territorio",
                     labels={"SubTotal": "Ventas (USD)", "Territory": ""}, color_discrete_sequence=[PALETTE[0]])
        fig.update_traces(hovertemplate="%{y}: $%{x:,.0f}<extra></extra>")
        c1.plotly_chart(style(fig), use_container_width=True)

        cat = d.groupby("Category").agg(Ventas=("LineTotal", "sum"), Unidades=("OrderQty", "sum")).reset_index()
        cat = cat.melt(id_vars="Category", var_name="Medida", value_name="valor")
        cat["%"] = 100 * cat["valor"] / cat.groupby("Medida")["valor"].transform("sum")
        fig = px.bar(cat, x="%", y="Category", color="Medida", barmode="group", orientation="h",
                     title="Participación por categoría: dinero vs unidades",
                     color_discrete_sequence=PALETTE[:2], labels={"Category": "", "%": "% del total"})
        fig.update_traces(hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
        c2.plotly_chart(style(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        top = (d.groupby("Product")["LineTotal"].sum().nlargest(10).sort_values().reset_index())
        fig = px.bar(top, x="LineTotal", y="Product", orientation="h", title="Top 10 productos",
                     labels={"LineTotal": "Ventas (USD)", "Product": ""}, color_discrete_sequence=[PALETTE[0]])
        fig.update_traces(hovertemplate="%{y}: $%{x:,.0f}<extra></extra>")
        c3.plotly_chart(style(fig), use_container_width=True)

        r = (reasons[reasons.SalesOrderID.isin(o.SalesOrderID)].groupby("Reason").size()
             .sort_values().rename("Pedidos").reset_index())
        fig = px.bar(r, x="Pedidos", y="Reason", orientation="h", title="Motivos de compra declarados",
                     labels={"Reason": ""}, color_discrete_sequence=[PALETTE[0]])
        fig.update_traces(hovertemplate="%{y}: %{x:,} pedidos<extra></extra>")
        c4.plotly_chart(style(fig), use_container_width=True)

# ------------------------------------------------------------------ Elección de k
X, _ = prepare_matrix(feats)
with tab_k:
    st.markdown("K-Means sobre **Recency, Frequency, Monetary** de los "
                f"**{len(feats):,} clientes individuales** (log1p + estandarización).")
    scores = get_scores(X)
    specs = [("inertia", "Inercia (codo)"), ("silhouette", "Silhouette ↑"),
             ("calinski_harabasz", "Calinski-Harabasz ↑"), ("davies_bouldin", "Davies-Bouldin ↓")]
    cols = st.columns(4)
    for col, (metric, title) in zip(cols, specs):
        fig = px.line(scores, x="k", y=metric, markers=True, title=title, color_discrete_sequence=[PALETTE[0]])
        fig.update_traces(marker_size=9, hovertemplate="k=%{x}<br>%{y:,.3f}<extra></extra>")
        fig.update_layout(yaxis_title="", xaxis=dict(dtick=1))
        col.plotly_chart(style(fig, 300), use_container_width=True)
    st.dataframe(scores.set_index("k").style.format("{:,.3f}"), use_container_width=True)

# ------------------------------------------------------------------ Segmentos
with tab_seg:
    k = st.slider("Número de clusters (k)", 2, 8, 4, help="k = 4 es el valor recomendado en el notebook 02")
    seg, centers, evr = get_segments(feats, k)
    order = segment_order(seg.Segment)
    colors = {s: SEGMENT_COLORS.get(s, PALETTE[i % len(PALETTE)]) for i, s in enumerate(order)}

    prof = seg.groupby("Segment").agg(Clientes=("CustomerID", "count"), Recency=("Recency", "median"),
                                      Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                                      Ventas=("Monetary", "sum")).reindex(order)
    prof["% clientes"] = 100 * prof.Clientes / prof.Clientes.sum()
    prof["% ventas"] = 100 * prof.Ventas / prof.Ventas.sum()

    cols = st.columns(len(order))
    for c, s in zip(cols, order):
        p = prof.loc[s]
        c.markdown(f"<div style='border-left:4px solid {colors[s]};padding-left:10px'>"
                   f"<b>{s}</b><br><span style='font-size:1.6em'>{p['% clientes']:.0f}%</span> de clientes<br>"
                   f"<b>{p['% ventas']:.0f}%</b> de las ventas</div>", unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([3, 2])
    sample = seg.sample(min(6000, len(seg)), random_state=42)
    fig = px.scatter(sample, x="PC1", y="PC2", color="Segment", symbol="Segment",
                     category_orders={"Segment": order}, color_discrete_map=colors, symbol_map=MARKERS,
                     opacity=0.6, hover_data={"Recency": True, "Frequency": True, "Monetary": ":$,.0f",
                                              "PC1": False, "PC2": False, "Segment": True},
                     title=f"Clientes en espacio PCA (varianza explicada {evr.sum():.0%})",
                     labels={"PC1": f"PC1 ({evr[0]:.0%})", "PC2": f"PC2 ({evr[1]:.0%})"})
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    c1.plotly_chart(style(fig, 480), use_container_width=True)

    fig = px.scatter(sample, x="Recency", y="Monetary", color="Segment", symbol="Segment", log_y=True,
                     category_orders={"Segment": order}, color_discrete_map=colors, symbol_map=MARKERS,
                     opacity=0.6, title="Recency vs Monetary",
                     labels={"Recency": "Días desde la última compra", "Monetary": "Gasto total (USD, log)"})
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
    fig.update_layout(showlegend=False)
    c2.plotly_chart(style(fig, 480), use_container_width=True)

    c3, c4 = st.columns(2)
    mix = (seg.groupby("Segment")[["Share_Bikes", "Share_Accessories", "Share_Clothing"]].mean()
           .reindex(order) * 100).rename(columns=lambda c: c.replace("Share_", ""))
    fig = go.Figure(go.Heatmap(z=mix.values, x=mix.columns, y=mix.index, colorscale="Blues",
                               text=mix.round(0).values, texttemplate="%{text:.0f}%", xgap=3, ygap=3,
                               hovertemplate="%{y}<br>%{x}: %{z:.1f}% del gasto<extra></extra>"))
    fig.update_layout(title="¿Qué compra cada segmento? (% del gasto)")
    c3.plotly_chart(style(fig), use_container_width=True)

    reg = pd.crosstab(seg.Segment, seg.TerritoryGroup, normalize="index").reindex(order) * 100
    fig = go.Figure(go.Heatmap(z=reg.values, x=reg.columns, y=reg.index, colorscale="Blues",
                               text=reg.round(0).values, texttemplate="%{text:.0f}%", xgap=3, ygap=3,
                               hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>"))
    fig.update_layout(title="Región de los clientes (% del segmento)")
    c4.plotly_chart(style(fig), use_container_width=True)

    st.subheader("Perfil de los segmentos")
    st.dataframe(prof.style.format({"Clientes": "{:,.0f}", "Recency": "{:,.0f} días", "Frequency": "{:.2f}",
                                    "Monetary": "${:,.0f}", "Ventas": "${:,.0f}",
                                    "% clientes": "{:.1f}%", "% ventas": "{:.1f}%"}),
                 use_container_width=True)
    st.caption("Centroides en escala original")
    st.dataframe(centers.reindex(order).style.format({"Recency": "{:,.0f}", "Frequency": "{:.2f}",
                                                      "Monetary": "${:,.0f}"}), use_container_width=True)

    st.download_button("⬇️ Descargar clientes segmentados (CSV)",
                       seg[["CustomerID", "Segment", *RFM_COLS, "AvgTicket"]].to_csv(index=False).encode("utf-8"),
                       file_name=f"clientes_segmentados_k{k}.csv", mime="text/csv")
