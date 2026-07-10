import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ─── PAGE CONFIG
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM STYLES ────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    .stAlert { border-radius: 10px; }
    .block-container { padding-top: 1.5rem; }
    .segment-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #1e3a5f;
        color: #60b4ff;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS 
DEFAULT_FILE = "data/customer_segments.csv"
def get_segment_names(n_clusters: int) -> dict[int, str]:
    if n_clusters == 2:
        return {0: "Inactive Customers", 1: "Active Customers"}
    elif n_clusters == 3:
        return {0: "Inactive Customers", 1: "Regular Customers", 2: "VIP Customers"}
    elif n_clusters == 4:
        return {0: "Inactive Customers", 1: "Regular Customers", 2: "Loyal Customers", 3: "VIP Customers"}
    elif n_clusters == 5:
        return {0: "Inactive Customers", 1: "At-Risk Customers", 2: "Regular Customers", 3: "Loyal Customers", 4: "VIP Customers"}
    else:
        return {i: f"Segment {i}" for i in range(n_clusters)}


def make_friendly_segments(series: pd.Series) -> pd.Series:
    """If the segment series contains numeric labels, convert them to friendly names."""
    # Convert series to string first for uniform string/number matching
    series_str = series.astype(str)
    unique_vals = sorted(series_str.dropna().unique())
    # Check if all unique values are digits (excluding empty/null values)
    is_numeric = all(val.replace('.0', '').isdigit() for val in unique_vals if val not in ["nan", "None", "<NA>", ""])
    if is_numeric and len(unique_vals) > 0:
        # Sort them numerically
        numeric_vals = sorted([int(float(val)) for val in unique_vals if val not in ["nan", "None", "<NA>", ""]])
        n_clusters = len(numeric_vals)
        names_map = get_segment_names(n_clusters)
        
        # Map values
        return series.map(lambda x: names_map.get(int(float(x)), f"Segment {x}") if pd.notna(x) else x)
    return series.astype(str)


PALETTE = px.colors.qualitative.Bold


# ─── HELPERS 
@st.cache_data
def load_default():
    return pd.read_csv(DEFAULT_FILE)


def get_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_data = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        unique_count = df[col].nunique()
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().head(3).tolist()
        sample_str = ", ".join([str(v) for v in sample_vals])
        summary_data.append({
            "Column Name": col,
            "Data Type": dtype,
            "Non-Null Count": len(df) - null_count,
            "Missing Values": f"{null_count} ({null_pct:.1f}%)",
            "Unique Values": unique_count,
            "Sample Preview": sample_str
        })
    return pd.DataFrame(summary_data)


def run_kmeans(df: pd.DataFrame, feature_cols: list[str], n_clusters: int) -> pd.DataFrame:
    """Scale + cluster and return df with ClusterLabel & ClusterName columns."""
    data = df[feature_cols].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(scaled)
    df = df.copy()
    df.loc[data.index, "ClusterLabel"] = labels.astype(int)
    # Sort clusters by mean of first feature so label 0 = lowest
    cluster_means = df.groupby("ClusterLabel")[feature_cols[0]].mean().sort_values()
    rank_map = {old: new for new, old in enumerate(cluster_means.index)}
    df["ClusterLabel"] = df["ClusterLabel"].map(rank_map)
    names_map = get_segment_names(n_clusters)
    df["CustomerSegment"] = df["ClusterLabel"].map(
        lambda x: names_map.get(int(x), f"Segment {int(x)}")
        if pd.notna(x) else np.nan
    )
    return df


def pca_2d(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    data = df[feature_cols].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)
    out = df.loc[data.index].copy()
    out["PC1"] = coords[:, 0]
    out["PC2"] = coords[:, 1]
    return out, pca.explained_variance_ratio_


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Dashboard Controls")
    st.markdown("---")

    # File Upload
    st.subheader("📂 Dataset")
    uploaded_file = st.file_uploader("Upload any CSV file", type=["csv"])

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded **{uploaded_file.name}**  \n`{raw_df.shape[0]:,}` rows · `{raw_df.shape[1]}` columns")
        using_default = False
    else:
        raw_df = load_default()
        st.info("📌 Using default dataset  \n`data/customer_segments.csv`")
        using_default = True

    st.markdown("---")

    # ── Column Mapping ────────────────────────────────────────────────────────
    all_cols = list(raw_df.columns)
    num_cols = list(raw_df.select_dtypes(include="number").columns)

    # Pre-fill guesses
    def _guess(keywords, cols, default_idx=0):
        for k in keywords:
            for c in cols:
                if k.lower() in c.lower():
                    return c
        return cols[default_idx] if cols else None

    with st.expander("🗺️ Column Mapping", expanded=False):
        col_recency   = st.selectbox("Recency column",   num_cols, index=num_cols.index(_guess(["recency","days","last"],   num_cols)) if _guess(["recency","days","last"],   num_cols) in num_cols else 0)
        col_frequency = st.selectbox("Frequency column", num_cols, index=num_cols.index(_guess(["freq","count","order","purchase"], num_cols)) if _guess(["freq","count","order","purchase"], num_cols) in num_cols else min(1, len(num_cols)-1))
        col_monetary  = st.selectbox("Monetary column",  num_cols, index=num_cols.index(_guess(["monetary","spend","revenue","amount","value","sales"], num_cols)) if _guess(["monetary","spend","revenue","amount","value","sales"], num_cols) in num_cols else min(2, len(num_cols)-1))

        # Optional: pre-existing segment column
        seg_cols = ["None (run K-Means clustering)"] + all_cols
        seg_default = "None (run K-Means clustering)"
        guessed_col = None
        for keyword in ["segment", "cluster", "label", "group", "class"]:
            for c in all_cols:
                if keyword in c.lower():
                    guessed_col = c
                    break
            if guessed_col:
                break
        if guessed_col:
            seg_default = guessed_col
        seg_col_choice = st.selectbox("Segment column (optional)", seg_cols,
                                      index=seg_cols.index(seg_default))

        # Optional: customer ID column
        id_cols = ["None"] + all_cols
        id_default = next((c for c in all_cols if "id" in c.lower() or "customer" in c.lower()), "None")
        col_id = st.selectbox("Customer ID column (optional)", id_cols,
                              index=id_cols.index(id_default) if id_default in id_cols else 0)

        currency_symbol = st.text_input("Currency symbol", value="$")

    st.markdown("---")

    # K-Means settings (shown only when no segment col)
    run_clustering = (seg_col_choice == "None (run K-Means clustering)")
    if run_clustering:
        st.subheader("⚙️ Clustering Settings")
        n_clusters = st.slider("Number of segments (K)", 2, 8, 4)
        extra_features = st.multiselect(
            "Extra numeric features for clustering",
            [c for c in num_cols if c not in [col_recency, col_frequency, col_monetary]],
            default=[]
        )

    st.markdown("---")

    # Filters (populated after processing)
    filter_header_placeholder = st.empty()
    filter_placeholder = st.empty()
    search_placeholder = st.empty()


# ─── PROCESS DATA ────────────────────────────────────────────────────────────────
feature_cols = list(dict.fromkeys([col_recency, col_frequency, col_monetary]))

if run_clustering:
    feature_cols_km = feature_cols + extra_features
    with st.spinner("🔄 Running K-Means clustering…"):
        df = run_kmeans(raw_df, feature_cols_km, n_clusters)
else:
    df = raw_df.copy()
    df["CustomerSegment"] = make_friendly_segments(df[seg_col_choice])

# Drop rows where any mapped column is NaN
df = df.dropna(subset=feature_cols + ["CustomerSegment"])

# ─── SIDEBAR FILTERS (now we have segments) ───────────────────────────────────
segment_options = sorted(df["CustomerSegment"].unique())

show_filter_section = (len(segment_options) > 1) or (col_id != "None")
if show_filter_section:
    filter_header_placeholder.subheader("🔍 Filters")

selected_segments = segment_options
if len(segment_options) > 1:
    with filter_placeholder:
        selected_segments = st.multiselect(
            "Customer Segments",
            options=segment_options,
            default=segment_options
        )

customer_id_query = ""
if col_id != "None":
    with search_placeholder:
        customer_id_query = st.text_input("🔎 Search Customer ID…")

# Apply filters
filtered_df = df[df["CustomerSegment"].isin(selected_segments)].copy()
if customer_id_query and col_id != "None":
    filtered_df = filtered_df[
        filtered_df[col_id].astype(str).str.contains(customer_id_query, case=False, na=False)
    ]


# ─── HEADER ──────────────────────────────────────────────────────────────────────
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown(f"""
Analysing **{filtered_df.shape[0]:,}** customers across **{len(selected_segments)}** segments  
using **{'K-Means Clustering' if run_clustering else 'pre-labelled segments'}** on columns:
`{col_recency}` · `{col_frequency}` · `{col_monetary}`
""")

if run_clustering:
    st.caption(f"K = {n_clusters} segments | Features: {', '.join(feature_cols_km)}")

st.markdown("---")


# ─── DATASET SUMMARY ─────────────────────────────────────────────────────────────
with st.expander("📋 Dataset Summary & Feature Profile", expanded=False):
    st.markdown("### 📊 Dataset Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{raw_df.shape[0]:,}")
    m2.metric("Total Columns", f"{raw_df.shape[1]:,}")
    m3.metric("Missing Cells", f"{raw_df.isnull().sum().sum():,}")
    m4.metric("Duplicate Rows", f"{raw_df.duplicated().sum():,}")
    
    st.markdown("### 🧬 Feature Profile Table")
    st.dataframe(get_dataset_summary(raw_df), use_container_width=True)


# ─── KPI METRICS ─────────────────────────────────────────────────────────────────
st.subheader("📊 Key Performance Indicators")

monetary_vals = pd.to_numeric(filtered_df[col_monetary], errors="coerce")
freq_vals     = pd.to_numeric(filtered_df[col_frequency], errors="coerce")
recency_vals  = pd.to_numeric(filtered_df[col_recency], errors="coerce")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers",     f"{len(filtered_df):,}")
k2.metric("Total Revenue",       f"{currency_symbol}{monetary_vals.sum():,.2f}")
k3.metric("Avg. Spend",          f"{currency_symbol}{monetary_vals.mean():,.2f}")
k4.metric("Avg. Frequency",      f"{freq_vals.mean():.1f}")
k5.metric("Avg. Recency (days)", f"{recency_vals.mean():.1f}")

st.markdown("---")


# ─── ROW 1: Distribution charts ───────────────────────────────────────────────────
st.subheader("Customer Segment Distribution")
c1, c2 = st.columns([3, 2])

with c1:
    _vc = filtered_df["CustomerSegment"].value_counts()
    seg_counts = pd.DataFrame({
        "CustomerSegment": _vc.index,
        "Count": _vc.values
    })

    fig_bar = px.bar(
        seg_counts,
        x="CustomerSegment", y="Count",
        color="CustomerSegment",
        text="Count",
        color_discrete_sequence=PALETTE
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        showlegend=False,
        xaxis_title="Segment",
        yaxis_title="Customer Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    fig_pie = px.pie(
        filtered_df,
        names="CustomerSegment",
        hole=0.45,
        color_discrete_sequence=PALETTE
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ─── ROW 2: RFM Scatter ─────────────────────────────────────────────────────────
st.subheader("📈 RFM Scatter — Frequency vs. Monetary")

hover_extra = []
if col_id != "None":
    hover_extra.append(col_id)
hover_extra.append(col_recency)

# Create safe size column (Plotly requires size >= 0)
filtered_df["_scatter_size"] = pd.to_numeric(filtered_df[col_monetary], errors="coerce").abs().fillna(0)

fig_scatter = px.scatter(
    filtered_df,
    x=col_frequency,
    y=col_monetary,
    color="CustomerSegment",
    size="_scatter_size",
    size_max=25,
    hover_data=hover_extra,
    color_discrete_sequence=PALETTE,
    opacity=0.75,
)
fig_scatter.update_layout(
    xaxis_title="Frequency",
    yaxis_title="Monetary Value",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_scatter, use_container_width=True)


# ─── ROW 3: Box plots per segment ───────────────────────────────────────────────
st.subheader("📦 RFM Distribution per Segment")
b1, b2, b3 = st.columns(3)

for col_widget, rfm_col, label in [
    (b1, col_recency,   "Recency"),
    (b2, col_frequency, "Frequency"),
    (b3, col_monetary,  "Monetary"),
]:
    fig_box = px.box(
        filtered_df,
        x="CustomerSegment",
        y=rfm_col,
        color="CustomerSegment",
        color_discrete_sequence=PALETTE,
        points="outliers",
    )
    fig_box.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title=label,
        title=f"{label} by Segment",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    col_widget.plotly_chart(fig_box, use_container_width=True)


# ─── ROW 4: Correlation heatmap ──────────────────────────────────────────────────
# st.subheader("🔥 Feature Correlation Heatmap")

# all_numeric_for_corr = [c for c in [col_recency, col_frequency, col_monetary] + (extra_features if run_clustering else [])]
# corr_df = filtered_df[all_numeric_for_corr].apply(pd.to_numeric, errors="coerce").corr()

# fig_heat = px.imshow(
#     corr_df,
#     text_auto=".2f",
#     color_continuous_scale="RdBu_r",
#     aspect="auto",
#     zmin=-1, zmax=1,
# )
# fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)")
# st.plotly_chart(fig_heat, use_container_width=True)


# ─── ROW 5: PCA 2-D cluster map ──────────────────────────────────────────────────
if len(feature_cols) >= 2:
    st.subheader("🧬 PCA Cluster Map (2-D Projection)")
    try:
        pca_df, evr = pca_2d(filtered_df, feature_cols)
        fig_pca = px.scatter(
            pca_df,
            x="PC1", y="PC2",
            color="CustomerSegment",
            opacity=0.7,
            color_discrete_sequence=PALETTE,
        )
        fig_pca.update_layout(
            xaxis_title=f"PC1 ({evr[0]*100:.1f}% var)",
            yaxis_title=f"PC2 ({evr[1]*100:.1f}% var)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pca, use_container_width=True)
    except Exception:
        st.info("PCA projection requires at least 2 numeric features with no missing values.")


# ─── ROW 6: Segment-level stats table ───────────────────────────────────────────
st.subheader("📋 Segment Summary Statistics")

agg = (
    filtered_df
    .groupby("CustomerSegment")[[col_recency, col_frequency, col_monetary]]
    .agg(["mean", "median", "std", "count"])
    .round(2)
)
st.dataframe(agg, use_container_width=True)


# ─── ROW 7: Export & Dataset Converter ───────────────────────────────────────────
with st.expander("📥 Dataset Converter & Export Options", expanded=False):
    st.markdown("### 🗃️ Raw Filtered Data View")
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("### 🔄 Export Formats")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.markdown("**Option 1: Standardized `customer_segments.csv` Format**")
        st.caption("Converts your file columns to match the standard schema: ID, Recency, Frequency, Monetary, Cluster, CustomerSegment.")
        
        std_df = pd.DataFrame()
        std_df["Customer ID"] = filtered_df[col_id] if col_id != "None" else range(1, len(filtered_df) + 1)
        std_df["Recency"] = filtered_df[col_recency]
        std_df["Frequency"] = filtered_df[col_frequency]
        std_df["Monetary"] = filtered_df[col_monetary]
        
        if "ClusterLabel" in filtered_df.columns:
            std_df["Cluster"] = filtered_df["ClusterLabel"].astype(int)
        else:
            std_df["Cluster"] = filtered_df["CustomerSegment"].astype("category").cat.codes
            
        std_df["CustomerSegment"] = filtered_df["CustomerSegment"]
        
        std_csv = std_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download customer_segments.csv",
            data=std_csv,
            file_name="customer_segments.csv",
            mime="text/csv",
            key="dl_std_csv"
        )
        
    with col_dl2:
        st.markdown("**Option 2: Full Original Dataset with Segment Info**")
        st.caption("Keeps all your original columns and adds the computed segment classification columns at the end.")
        
        raw_csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Dataset with Segments",
            data=raw_csv,
            file_name="segmented_customers_full.csv",
            mime="text/csv",
            key="dl_full_csv"
        )


# ─── ROW 8: AI Insights ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧠 AI Business Insights")

top_seg = (
    filtered_df
    .groupby("CustomerSegment")[col_monetary]
    .mean()
    .idxmax()
)
top_seg_count = len(filtered_df[filtered_df["CustomerSegment"] == top_seg])
total_revenue = monetary_vals.sum()
avg_order     = monetary_vals.mean()

insight_cols = st.columns(len(segment_options) if len(segment_options) <= 5 else 5)

for i, seg in enumerate(segment_options):
    seg_df = filtered_df[filtered_df["CustomerSegment"] == seg]
    seg_rev  = pd.to_numeric(seg_df[col_monetary], errors="coerce").sum()
    seg_pct  = (len(seg_df) / len(filtered_df) * 100) if len(filtered_df) else 0
    rev_pct  = (seg_rev / total_revenue * 100) if total_revenue else 0
    col = insight_cols[i % len(insight_cols)]
    col.metric(
        label=seg,
        value=f"{len(seg_df):,} customers",
        delta=f"{currency_symbol}{seg_rev:,.0f} ({rev_pct:.1f}% rev)"
    )

st.markdown(f"""
**Top Revenue Segment:** {top_seg} — {top_seg_count:,} customers  
💰 **Total Revenue:** {currency_symbol}{total_revenue:,.2f}  
🛍️ **Average Customer Spend:** {currency_symbol}{avg_order:,.2f}  
📊 **Segments Analysed:** {len(selected_segments)}
""")


# ─── FOOTER ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Customer Segmentation Dashboard • Machine Learning + Data Analytics Project")