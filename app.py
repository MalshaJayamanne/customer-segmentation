import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Customer Segmentation App")

# config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("data/customer_segments.csv")



# SIDEBAR
st.sidebar.title("Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

# Load uploaded or default dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Uploaded Dataset Loaded")
else:
    df = load_data()
    st.sidebar.success("Default Dataset Loaded")


required_columns = [
    'Recency',
    'Frequency',
    'Monetary',
    'CustomerSegment'
]

missing_cols = [
    col for col in required_columns
    if col not in df.columns
]

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()


# TITLE
st.title("🛍️ Customer Segmentation Dashboard")

st.markdown("""
Machine Learning powered customer segmentation using:

- RFM Analysis
- K-Means Clustering
- Business Intelligence Visualizations
""")


# FILTERS]
st.sidebar.header("🔍 Filters")

segment_options = df["CustomerSegment"].unique()

selected_segments = st.sidebar.multiselect(
    "Select Customer Segment",
    options=segment_options,
    default=segment_options
)

filtered_df = df[
    df["CustomerSegment"].isin(selected_segments)
]


# SEARCH customer ID
customer_id = st.sidebar.text_input(
    "Search Customer ID..."
)

if customer_id:

    if 'Customer ID' in filtered_df.columns:

        filtered_df = filtered_df[
            filtered_df['Customer ID']
            .astype(str)
            .str.contains(customer_id)
        ]


csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Segmented Data",
    data=csv,
    file_name='customer_segments.csv',
    mime='text/csv'
)


# KPI 
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    len(filtered_df)
)

col2.metric(
    "Average Spending",
    f"${filtered_df['Monetary'].mean():.2f}"
)

col3.metric(
    "Average Frequency",
    round(filtered_df['Frequency'].mean(), 2)
)

col4.metric(
    "Average Recency",
    round(filtered_df['Recency'].mean(), 2)
)


# SEGMENT DISTRIBUTION
st.subheader("Customer Segment Distribution")

segment_counts = (
    filtered_df['CustomerSegment']
    .value_counts()
    .reset_index()
)

segment_counts.columns = [
    'CustomerSegment',
    'Count'
]

fig_bar = px.bar(
    segment_counts,
    x='CustomerSegment',
    y='Count',
    color='CustomerSegment',
    text='Count'
)

fig_bar.update_layout(
    xaxis_title="Customer Segment",
    yaxis_title="Customer Count"
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)


# PIE CHART
st.subheader("Customer Segment Share")

fig_pie = px.pie(
    filtered_df,
    names='CustomerSegment',
    hole=0.4
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)


# RFM SCATTER PLOT
st.subheader("📈 RFM Customer Analysis")

fig_scatter = px.scatter(
    filtered_df,
    x='Frequency',
    y='Monetary',
    color='CustomerSegment',
    size='Monetary',
    hover_data=['Recency']
)

fig_scatter.update_layout(
    xaxis_title="Frequency",
    yaxis_title="Monetary Value"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# HEATMAP
st.subheader("🔥 RFM Correlation Heatmap")

corr = filtered_df[
    ['Recency', 'Frequency', 'Monetary']
].corr()

fig_heat = px.imshow(
    corr,
    text_auto=True,
    aspect='auto'
)

st.plotly_chart(
    fig_heat,
    use_container_width=True
)


# RAW DATA
st.subheader("Customer Data")

st.dataframe(
    filtered_df,
    use_container_width=True
)


# AI BUSINESS INSIGHTS
st.subheader("🧠 AI Business Insights")

vip_count = len(
    filtered_df[
        filtered_df['CustomerSegment']
        == 'VIP Customers'
    ]
)

regular_count = len(
    filtered_df[
        filtered_df['CustomerSegment']
        == 'Regular Customers'
    ]
)

total_revenue = filtered_df[
    'Monetary'
].sum()

avg_order = filtered_df[
    'Monetary'
].mean()

st.info(f"""
💎 VIP Customers: {vip_count}

VIP customers generate the highest business value and should receive loyalty rewards, premium offers, and personalized engagement.

🛍️ Regular Customers: {regular_count}

Regular customers represent the core customer base and can be targeted using upselling and retention strategies.

💰 Total Revenue: ${total_revenue:,.2f}

Average Customer Spending: ${avg_order:,.2f}
""")


# FOOTER
st.markdown("---")

st.caption(
    "Customer Segmentation Dashboard • Machine Learning + Data Analytics Project"
)