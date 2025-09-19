import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from typing import Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, r2_score,
    mean_absolute_error, mean_squared_error
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
def read_uploaded_file(uploaded_file, excel_sheet: Optional[str]) -> Tuple[pd.DataFrame, Optional[List[str]], Optional[str]]:
    """
    Read uploaded file. If Excel, allow sheet selection.
    Parquet is attempted but may fail in hosted environments without pyarrow/fastparquet.
    Returns: (df, sheet_names_or_None, selected_sheet_or_None)
    """
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(uploaded_file)
            if excel_sheet is None:
                first = xls.sheet_names[0]
                return pd.read_excel(xls, sheet_name=first), xls.sheet_names, first
            else:
                return pd.read_excel(xls, sheet_name=excel_sheet), xls.sheet_names, excel_sheet
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel file: {e}")
    elif name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file), None, None
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")
    elif name.endswith(".parquet"):
        # Parquet requires pyarrow or fastparquet. If not present, provide actionable message.
        try:
            df = pd.read_parquet(uploaded_file)
            return df, None, None
        except Exception:
            raise RuntimeError(
                "Reading parquet failed. The hosted environment may not have pyarrow/fastparquet installed. "
                "Please upload CSV or Excel (XLSX)."
            )
    else:
        # fallback attempt
        try:
            df = pd.read_parquet(uploaded_file)
            return df, None, None
        except Exception as e:
            raise RuntimeError(f"Unsupported filetype or read error: {e}")

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
        try:
            _df_tmp, sheet_names, current = read_uploaded_file(uploaded, None)
            excel_sheet = st.selectbox("Excel sheet", sheet_names, index=sheet_names.index(current) if current in sheet_names else 0)
            df, _, _ = read_uploaded_file(uploaded, excel_sheet)
            st.success(f"Loaded uploaded Excel: sheet '{excel_sheet}' — shape {df.shape}.")
        except Exception as e:
            st.error(str(e))
            st.stop()
    elif uploaded is not None:
        try:
            df, _, _ = read_uploaded_file(uploaded, None)
            st.success(f"Loaded uploaded file — shape {df.shape}.")
        except Exception as e:
            st.error(str(e))
            st.stop()
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
        labels_sorted = sorted(pd.Series(y_test).unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels_sorted], columns=[f"Pred {l}" for l in labels_sorted])
        st.dataframe(cm_df)
        st.caption("**What this shows:** Correct vs misclassified counts per class; strong diagonal indicates good performance.")

    # ROC Curves
    st.subheader("ROC Curves (All Selected Models)")
    fig_roc, ax_roc = plt.subplots(figsize=(7,5))
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
    file_new = st.file_uploader("Upload new data for prediction (without target column)", type=["xlsx","xls","csv","parquet"], key="pred_upload")
    if file_new is not None:
        try:
            if file_new.name.lower().endswith((".xlsx", ".xls")):
                xls_pred = pd.ExcelFile(file_new)
                sheet_pred = st.selectbox("Sheet for prediction file", xls_pred.sheet_names)
                new_df = pd.read_excel(xls_pred, sheet_name=sheet_pred)
            elif file_new.name.lower().endswith(".csv"):
                new_df = pd.read_csv(file_new)
            else:
                # parquet attempt; may raise RuntimeError if pyarrow is missing
                new_df = pd.read_parquet(file_new)
            preds = trained[model_for_pred].predict(new_df)
            out = new_df.copy()
            out["prediction"] = preds
            st.dataframe(out.head(20))
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions", csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))
            st.info("If you uploaded a parquet file and see this error, upload CSV/XLSX instead or install pyarrow/fastparquet in your environment.")

# ------------------- Tab 3: Clustering -------------------
with tab3:
    st.header("K-Means Clustering")
    num_cols, cat_cols = detect_column_types(df)
    features = st.multiselect("Select features for clustering (numeric recommended)", df.columns.tolist(), default=[c for c in num_cols][: min(6, len(num_cols))])
    if not features:
        st.warning("Select at least one feature to proceed.")
    else:
        work = df[features].copy()
        work = work.select_dtypes(include=[np.number]).dropna()
        if work.empty:
            st.error("No numeric data available for clustering after dropping NA.")
        else:
            st.subheader("Elbow Plot")
            inertias = []
            K_range = range(2, 11)
            for k in K_range:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                km.fit(work)
                inertias.append(km.inertia_)
            fig_elb = px.line(x=list(K_range), y=inertias, labels={"x":"K (clusters)","y":"Inertia"})
            st.plotly_chart(fig_elb, use_container_width=True)
            st.caption("**What this shows:** How within-cluster variance decreases with K; look for an 'elbow' where improvements taper.")

            k_sel = st.slider("Number of clusters", min_value=2, max_value=10, value=3, step=1)
            km2 = KMeans(n_clusters=k_sel, n_init=10, random_state=42).fit(work)
            labels = km2.labels_
            df_clusters = df.copy()
            df_clusters["cluster"] = labels
            st.subheader("Cluster Personas (Numeric feature means)")
            persona = df_clusters.groupby("cluster")[features].mean(numeric_only=True).round(2)
            st.dataframe(persona)
            st.caption("**What this shows:** Average profile per cluster; use to label customer personas.")

            download_dataframe_button(df_clusters, "data_with_clusters.csv", "Download full data with cluster labels")

# ------------------- Tab 4: Association Rules -------------------
with tab4:
    st.header("Association Rule Mining (Apriori)")
    _, cat_cols = detect_column_types(df)
    cat_choices = st.multiselect("Choose 2 or more categorical columns", cat_cols, default=cat_cols[:2] if len(cat_cols)>=2 else cat_cols)
    min_support = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)
    min_lift = st.slider("Min Lift", 0.5, 5.0, 1.0, 0.1)

    if len(cat_choices) >= 2:
        basket = pd.DataFrame()
        for c in cat_choices:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c)
            basket = pd.concat([basket, dummies], axis=1)
        freq = apriori(basket, min_support=min_support, use_colnames=True)
        if not freq.empty:
            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
            rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)
            rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
            rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
            show_cols = ["antecedents","consequents","support","confidence","lift"]
            st.subheader("Top 10 Associations")
            st.dataframe(rules[show_cols].head(10).round(3))
            st.caption("**What this shows:** Strong co-occurrence patterns among selected categorical attributes.")
        else:
            st.warning("No frequent itemsets found at current support threshold. Try lowering it.")
    else:
        st.info("Select at least two categorical columns to run Apriori.")

# ------------------- Tab 5: Regression -------------------
with tab5:
    st.header("Regression (Quick Insights)")
    num_cols, cat_cols = detect_column_types(df)
    numeric_targets = [c for c in num_cols if df[c].nunique() > 10]
    if not numeric_targets:
        st.warning("No suitable numeric targets detected (with >10 unique values).")
    else:
        target_r = st.selectbox("Select numeric target", numeric_targets)
        features_r = st.multiselect("Select features (exclude target)", [c for c in df.columns if c != target_r], default=[c for c in num_cols if c != target_r][: min(6, len(num_cols))])
        test_size_r = st.slider("Test size", 0.1, 0.5, 0.2, step=0.05, key="reg_ts")

        X = df[features_r].copy()
        y = df[target_r].copy()

        num_c = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_c = X.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])
        categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, num_c), ("cat", categorical_transformer, cat_c)]
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_r, random_state=42)

        regs = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=42),
            "Lasso": Lasso(alpha=0.001, max_iter=10000, random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }

        rows = []
        trained_r = {}
        for name, model in regs.items():
            pipe = Pipeline(steps=[("pre", preprocessor), ("reg", model)])
            pipe.fit(X_train, y_train)
            trained_r[name] = pipe
            pred_tr = pipe.predict(X_train)
            pred_te = pipe.predict(X_test)
            r2_tr = r2_score(y_train, pred_tr)
            r2_te = r2_score(y_test, pred_te)
            mae = mean_absolute_error(y_test, pred_te)
            rmse = mean_squared_error(y_test, pred_te, squared=False)
            rows.append([name, r2_tr, r2_te, mae, rmse])

        met = pd.DataFrame(rows, columns=["Model","R2 (train)","R2 (test)","MAE","RMSE"])
        st.subheader("Model Comparison")
        st.dataframe(met.round(3))
        st.caption("**What this shows:** Fit quality and error magnitudes; compare for bias-variance trade-offs.")

        st.subheader("Quick Insights")
        st.write("- Coefficients/feature importance (where available)")
        for name in ["Linear", "Ridge", "Lasso", "Decision Tree"]:
            model = trained_r[name]
            pre = model.named_steps["pre"]
            ohe = pre.named_transformers_["cat"].named_steps["onehot"] if cat_c else None
            num_feats = num_c
            cat_feats = list(ohe.get_feature_names_out(cat_c)) if ohe is not None else []
            feat_names = np.array(num_feats + cat_feats)
            try:
                reg = model.named_steps["reg"]
                if hasattr(reg, "coef_"):
                    coefs = np.ravel(reg.coef_)
                    imp = pd.Series(coefs, index=feat_names).sort_values(key=lambda x: np.abs(x), ascending=False).head(10)
                    st.write(f"**Top drivers — {name}**")
                    st.dataframe(imp.round(4))
                elif hasattr(reg, "feature_importances_"):
                    fi = pd.Series(reg.feature_importances_, index=feat_names).sort_values(ascending=False).head(10)
                    st.write(f"**Top drivers — {name}**")
                    st.dataframe(fi.round(4))
            except Exception as e:
                st.info(f"{name}: could not compute feature contributions ({e}).")

# ------------------- Tab 6: About -------------------
with tab6:
    st.header("About this App")
    st.markdown("""
    This dashboard was generated for the IBR 2 project to support analysis for **Shiffa** and **Estée Lauder**.
    Use the sidebar to upload Excel directly (choose the sheet), apply global filters, and explore each tab for tailored analytics.
    """)
    st.markdown("""
    **Credits:** Streamlit, scikit-learn, mlxtend, Plotly, Seaborn, Matplotlib.
    """)
