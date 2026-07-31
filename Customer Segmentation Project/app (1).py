import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales & Revenue Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/sales_data.csv", parse_dates=["Date"])
    return df

df = load_data()

st.title("📊 Sales & Revenue Analysis Dashboard")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Region", df["Region"].unique(), default=list(df["Region"].unique()))
categories = st.sidebar.multiselect("Category", df["Category"].unique(), default=list(df["Category"].unique()))
date_range = st.sidebar.date_input("Date Range", [df["Date"].min(), df["Date"].max()])

mask = (
    df["Region"].isin(regions)
    & df["Category"].isin(categories)
    & (df["Date"] >= pd.to_datetime(date_range[0]))
    & (df["Date"] <= pd.to_datetime(date_range[1]))
)
fdf = df[mask]

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"₹{fdf['Sales'].sum():,.0f}")
col2.metric("Total Profit", f"₹{fdf['Profit'].sum():,.0f}")
col3.metric("Orders", f"{len(fdf):,}")
col4.metric("Avg Order Value", f"₹{fdf['Sales'].mean():,.0f}" if len(fdf) else "₹0")

st.divider()

# ---------- Revenue trend ----------
trend = fdf.groupby(pd.Grouper(key="Date", freq="ME"))["Sales"].sum().reset_index()
fig_trend = px.line(trend, x="Date", y="Sales", title="Monthly Revenue Trend", markers=True)
st.plotly_chart(fig_trend, use_container_width=True)

c1, c2 = st.columns(2)

# ---------- Top products ----------
top_products = fdf.groupby("Product")["Sales"].sum().nlargest(10).reset_index()
fig_top = px.bar(top_products, x="Sales", y="Product", orientation="h", title="Top 10 Products by Sales")
c1.plotly_chart(fig_top, use_container_width=True)

# ---------- Sales by region ----------
region_sales = fdf.groupby("Region")["Sales"].sum().reset_index()
fig_region = px.pie(region_sales, names="Region", values="Sales", title="Sales Share by Region")
c2.plotly_chart(fig_region, use_container_width=True)

# ---------- Category performance ----------
cat_perf = fdf.groupby("Category")[["Sales", "Profit"]].sum().reset_index()
fig_cat = px.bar(cat_perf, x="Category", y=["Sales", "Profit"], barmode="group", title="Category-wise Sales vs Profit")
st.plotly_chart(fig_cat, use_container_width=True)

st.divider()
st.subheader("Raw Data")
st.dataframe(fdf, use_container_width=True)
