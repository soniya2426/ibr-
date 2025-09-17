import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, RocCurveDisplay, r2_score,
    mean_absolute_error, mean_squared_error, classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="IBR 2 Dashboard — Shiffa & Estée Lauder", layout="wide")

# ------------------- Utilities -------------------
@st.cache_data
def load_excel(path: str, sheet_name: str):
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df

@st.cache_data
def read_uploaded_file(uploaded_file, excel_sheet: str | None):
    """Read uploaded file. If Excel, allow sheet selection."""
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        # If sheet is provided, read that; else read first sheet
        if excel_sheet is None:
            xls = pd.ExcelFile(uploaded_file)
            first = xls.sheet_names[0]
            return pd.read_excel(xls, sheet_name=first), xls.sheet_names, first
        else:
            xls = pd.ExcelFile(uploaded_file)
            return pd.read_excel(xls, sheet_name=excel_sheet), xls.sheet_names, excel_sheet
    elif name.endswith(".csv"):
        return pd.read_csv(uploaded_file), None, None
    else:
        return pd.read_parquet(uploaded_file), None, None

@st.cache_data
def detect_column_types(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols

def brand_filter_ui(df: pd.DataFrame):
    options = None
    for col in df.columns:
        if col.lower() in ["brand", "brands"]:
            options = df[col].dropna().unique().tolist()
            break
    if options is not None and len(options) > 0:
        selected = st.multiselect("Filter by Brand", sorted(map(str, options)), default=sorted(map(str, options)))
        if selected:
            brand_col = [c for c in df.columns if c.lower() in ["brand", "brands"]][0]
            df = df[df[brand_col].astype(str).isin(selected)]
    else:
        st.info("No `brand` column detected. Skipping brand filter.")
    return df

def download_dataframe_button(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")

def describe_fig(caption: str):
    st.caption(f"**What this shows:** {caption}")

# ------------------- Data Loading -------------------
with st.sidebar:
    st.title("IBR 2 Controls")
    st.write("Upload a dataset (Excel/CSV/Parquet). If not provided, the default packaged dataset is used.")

    uploaded = st.file_uploader("Upload file", type=["xlsx", "xls", "csv", "parquet"], help="Upload Excel directly here; you'll be able to choose the sheet.")

    excel_sheet = None
    sheet_name_default = "ibr final responses for dashboard 2.xlsx"

    if uploaded is not None and (uploaded.name.lower().endswith(".xlsx") or uploaded.name.lower().endswith(".xls")):
        # First pass to get sheet names
        _df_tmp, sheet_names, current = read_uploaded_file(uploaded, None)
        excel_sheet = st.selectbox("Excel sheet", sheet_names, index=sheet_names.index(current) if current in sheet_names else 0)
        df, _, _ = read_uploaded_file(uploaded, excel_sheet)
        st.success(f"Loaded uploaded Excel: sheet '{excel_sheet}' — shape {df.shape}.")
    elif uploaded is not None:
        df, _, _ = read_uploaded_file(uploaded, None)
        st.success(f"Loaded uploaded file — shape {df.shape}.")
    else:
        # Fallback to default packaged dataset
        try:
            df = load_excel("data/ibr_final_responses_for_dashboard_2.xlsx", sheet_name=sheet_name_default)
            st.success(f"Loaded default dataset with sheet '{sheet_name_default}' — shape {df.shape}.")
        except Exception as e:
            st.error(f"Failed to load default dataset: {e}")
            st.stop()

    # Global Filters
    st.subheader("Global Filters")
    df = brand_filter_ui(df)

    # Generic column filters
    col_to_filter = st.multiselect("Choose columns to filter", df.columns.tolist())
    for c in col_to_filter:
        if pd.api.types.is_numeric_dtype(df[c]):
            min_v, max_v = float(df[c].min()), float(df[c].max())
            lo, hi = st.slider(f"Range for {c}", min_value=min_v, max_value=max_v, value=(min_v, max_v))
            df = df[(df[c] >= lo) & (df[c] <= hi)]
        else:
            vals = sorted(df[c].dropna().astype(str).unique().tolist())
            picks = st.multiselect(f"Values for {c}", vals, default=vals[: min(10, len(vals))])
            if picks:
                df = df[df[c].astype(str).isin(picks)]

    st.write("After filtering, your working data shape is:", df.shape)
    download_dataframe_button(df, "filtered_dataset.csv", "Download current filtered data")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Visualization", "Classification", "Clustering", "Association Rules", "Regression", "About"
])

# ------------------- Tab 1: Data Visualization -------------------
with tab1:
    st.header("Data Visualization & Descriptive Insights")
    num_cols, cat_cols = detect_column_types(df)

    # 1. Missingness heatmap
    st.subheader("1) Missingness Heatmap")
    if len(df.columns) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(df.isna(), cbar=False, ax=ax)
        st.pyplot(fig, use_container_width=True)
        describe_fig("Highlights where features have missing values (if any). Dense bands indicate systematic gaps.")
    else:
        st.info("No columns to plot.")

    # 2. Summary stats
    st.subheader("2) Summary Statistics")
    st.dataframe(df.describe(include="all").transpose())
    describe_fig("Overview of central tendency, spread, and unique value counts across columns.")

    # 3. Correlation heatmap (numeric)
    num_only = df.select_dtypes(include=[np.number])
    if num_only.shape[1] >= 2:
        st.subheader("3) Correlation Heatmap (Numeric)")
        corr = num_only.corr(numeric_only=True)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax2)
        st.pyplot(fig2, use_container_width=True)
        describe_fig("Pairwise linear relationships between numeric variables; strong positive/negative blocks indicate potential drivers.")

    # 4. Top categories
    if cat_cols:
        st.subheader("4) Top Categories by Frequency")
        cat_col = st.selectbox("Choose a categorical column", cat_cols, key="viz_cat_col")
        top_counts = df[cat_col].astype(str).value_counts().head(15).reset_index()
        top_counts.columns = [cat_col, "count"]
        fig3 = px.bar(top_counts, x=cat_col, y="count")
        st.plotly_chart(fig3, use_container_width=True)
        describe_fig("Displays the most frequent categories which can influence segment sizes or sampling.")

    # 5. Numeric distribution
    if num_cols:
        st.subheader("5) Numeric Distribution")
        ncol = st.selectbox("Choose a numeric column", num_cols, key="viz_num_col")
        fig4 = px.histogram(df, x=ncol, nbins=30, marginal="box")
        st.plotly_chart(fig4, use_container_width=True)
        describe_fig("Histogram with box plot reveals skewness, outliers, and multi-modality.")

    # 6. Scatter explorer
    if len(num_cols) >= 2:
        st.subheader("6) Scatter Explorer")
        x = st.selectbox("X-axis", num_cols, key="viz_scatter_x")
        y = st.selectbox("Y-axis", [c for c in num_cols if c != x], key="viz_scatter_y")
        color = st.selectbox("Color by (optional)", ["(none)"] + cat_cols, key="viz_scatter_color")
        fig5 = px.scatter(df, x=x, y=y, color=None if color=="(none)" else color, trendline="ols")
        st.plotly_chart(fig5, use_container_width=True)
        describe_fig("Shows bivariate relationships; the trendline hints at direction/strength of association.")

    # 7. Box by category
    if num_cols and cat_cols:
        st.subheader("7) Box Plot by Category")
        numc = st.selectbox("Numeric", num_cols, key="viz_box_num")
        catc = st.selectbox("Category", cat_cols, key="viz_box_cat")
        fig6 = px.box(df, x=catc, y=numc)
        st.plotly_chart(fig6, use_container_width=True)
        describe_fig("Contrasts distribution of a metric across categories; wide boxes/outliers suggest heterogeneous segments.")

    # 8. Pareto chart (80/20)
    if cat_cols:
        st.subheader("8) Pareto (Cumulative %)")
        pareto_col = st.selectbox("Categorical column", cat_cols, key="viz_pareto")
        counts = df[pareto_col].astype(str).value_counts()
        pareto_df = counts.reset_index()
        pareto_df.columns = ["category", "count"]
        pareto_df["cum_pct"] = pareto_df["count"].cumsum() / pareto_df["count"].sum() * 100
        fig7 = px.bar(pareto_df, x="category", y="count")
        st.plotly_chart(fig7, use_container_width=True)
        st.line_chart(pareto_df.set_index("category")["cum_pct"])
        describe_fig("Helps identify the few categories contributing to the majority share (80/20 rule).")

    # 9. Time trend (if datetime column)
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if dt_cols:
        st.subheader("9) Time Trend")
        dtc = st.selectbox("Datetime column", dt_cols, key="viz_time_col")
        freq = st.selectbox("Resample frequency", ["D","W","M","Q","Y"], index=2)
        ts = df.set_index(dtc).sort_index()
        series_col = st.selectbox("Value to aggregate", num_cols, key="viz_time_val")
        agg = ts[series_col].resample(freq).mean().dropna()
        fig8 = px.line(agg, y=series_col)
        st.plotly_chart(fig8, use_container_width=True)
        describe_fig("Shows how a key metric evolves over time to spot seasonality or shifts.")

    # 10. Pivot heatmap
    if num_cols and cat_cols:
        st.subheader("10) Pivot Heatmap")
        r = st.selectbox("Rows", cat_cols, key="viz_pivot_r")
        c = st.selectbox("Columns", cat_cols, key="viz_pivot_c")
        v = st.selectbox("Value", num_cols, key="viz_pivot_v")
        pv = pd.pivot_table(df, index=r, columns=c, values=v, aggfunc="mean")
        fig9, ax9 = plt.subplots(figsize=(8,6))
        sns.heatmap(pv, cmap="viridis", ax=ax9)
        st.pyplot(fig9, use_container_width=True)
        describe_fig("Cross-category average intensities; bright/dark cells reveal standout combinations.")

# ------------------- Tab 2: Classification -------------------
with tab2:
    st.header("Classification")
    num_cols, cat_cols = detect_column_types(df)
    target = st.selectbox("Select target (classification)", options=df.columns.tolist())
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
    random_state = st.number_input("Random state", value=42, step=1)
    models_to_run = st.multiselect(
        "Choose algorithms", 
        ["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"],
        default=["KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]
    )

    # Build preprocessing
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    # ensure classification
    if pd.api.types.is_numeric_dtype(y):
        # if too many unique values, coerce to categorical by binning
        if y.nunique() > 20:
            st.warning("Target seems numeric with many unique values; converting to categorical by quantiles (4 bins).")
            y = pd.qcut(y, q=4, labels=[f"Q{i}" for i in range(1,5)])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, num_cols), ("cat", categorical_transformer, cat_cols)]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)

    def make_model(name):
        if name == "KNN":
            return KNeighborsClassifier(n_neighbors=5)
        if name == "Decision Tree":
            return DecisionTreeClassifier(random_state=int(random_state))
        if name == "Random Forest":
            return RandomForestClassifier(n_estimators=300, random_state=int(random_state))
        if name == "Gradient Boosting":
            return GradientBoostingClassifier(random_state=int(random_state))
        raise ValueError(name)

    results = []
    trained = {}
    for mname in models_to_run:
        pipe = Pipeline(steps=[("pre", preprocessor), ("clf", make_model(mname))])
        pipe.fit(X_train, y_train)
        trained[mname] = pipe
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        acc_tr = accuracy_score(y_train, y_pred_train)
        acc_te = accuracy_score(y_test, y_pred_test)
        # macro metrics for multi-class
        prec = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
        results.append([mname, acc_tr, acc_te, prec, rec, f1])

    res_df = pd.DataFrame(results, columns=["Model","Train Acc","Test Acc","Precision (macro)","Recall (macro)","F1 (macro)"])
    st.subheader("Performance Summary")
    st.dataframe(res_df.style.format({"Train Acc":"{:.3f}","Test Acc":"{:.3f}","Precision (macro)":"{:.3f}","Recall (macro)":"{:.3f}","F1 (macro)":"{:.3f}"}))
    st.caption("**What this shows:** Comparative generalization; large train-test gaps may indicate overfitting.")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    model_for_cm = st.selectbox("Choose model", models_to_run, key="cm_model")
    if model_for_cm:
        best = trained[model_for_cm]
        y_pred = best.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(pd.Series(y_test).unique()))
        cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in sorted(pd.Series(y_test).unique())], columns=[f"Pred {l}" for l in sorted(pd.Series(y_test).unique())])
        st.dataframe(cm_df)
        st.caption("**What this shows:** Correct vs misclassified counts per class; strong diagonal indicates good performance.")

    # ROC Curves
    st.subheader("ROC Curves (All Selected Models)")
    fig_roc, ax_roc = plt.subplots(figsize=(7,5))
    classes = sorted(pd.Series(y_test).unique())
    for mname in models_to_run:
        model = trained[mname]
        try:
            y_score = model.predict_proba(X_test)
            if y_score.ndim == 1:
                y_score = np.vstack([1 - y_score, y_score]).T
        except Exception:
            try:
                y_score = model.decision_function(X_test)
            except Exception:
                lb = LabelBinarizer().fit(y_train)
                y_score = lb.transform(model.predict(X_test))

        lb2 = LabelBinarizer().fit(y_train)
        Y_test = lb2.transform(y_test)
        if Y_test.ndim == 1:
            Y_test = np.column_stack([1-Y_test, Y_test])
        fpr, tpr, _ = roc_curve(Y_test.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{mname} (AUC={roc_auc:.3f})")
    ax_roc.plot([0,1], [0,1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Micro-averaged ROC Curves")
    ax_roc.legend()
    st.pyplot(fig_roc, use_container_width=True)
    st.caption("**What this shows:** Overall ranking ability across classes; higher AUC indicates better separability.")

    # Batch Predict New Data
    st.subheader("Batch Predict on New Data")
    model_for_pred = st.selectbox("Model to use for prediction", models_to_run, key="pred_model")
    file_new = st.file_uploader("Upload new data for prediction (without target column)", type=["xlsx","xls","csv","parquet"], key
