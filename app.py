import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bike Rental Predictor", page_icon="🚲", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0B0F19 0%, #111827 100%); color: #E5E7EB !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0D1117; border-right: 1px solid #1F2937; }
[data-testid="stSidebar"] * { color: #D1D5DB !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #9CA3AF !important; }
[data-testid="stSidebar"] h2 { color: #34D399 !important; }

/* ── Slider: make value label visible ── */
[data-testid="stSlider"] { color: #E5E7EB !important; }
[data-testid="stSlider"] label { color: #D1D5DB !important; font-weight: 500; }
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #9CA3AF !important; }
div[data-baseweb="slider"] div[role="slider"] { background: #34D399 !important; }
div[data-baseweb="slider"] div { border-color: #34D399 !important; }
.stSlider p { color: #E5E7EB !important; font-size: 0.85rem; }

/* ── Selectbox ── */
[data-testid="stSidebar"] .stSelectbox label { color: #D1D5DB !important; }
[data-baseweb="select"] { background-color: #1F2937 !important; }
[data-baseweb="select"] * { color: #E5E7EB !important; }

/* ── All body text ── */
p, span, label, li { color: #E5E7EB !important; }
h1 { background: linear-gradient(90deg,#34D399,#60A5FA); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:700; font-size:2rem; }
h2, h3 { color: #34D399 !important; font-weight: 600; }
h4 { color: #93C5FD !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: rgba(31,41,55,0.8); border: 1px solid #374151;
    border-radius: 14px; padding: 18px 22px;
    transition: transform .25s, border-color .25s, box-shadow .25s;
}
div[data-testid="metric-container"]:hover { transform: translateY(-4px); border-color: #34D399; box-shadow: 0 8px 30px rgba(52,211,153,0.15); }
div[data-testid="metric-container"] label { color: #9CA3AF !important; font-size:.8rem; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #34D399 !important; font-size:2rem !important; font-weight:700 !important; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #60A5FA !important; font-size:.85rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { background:rgba(31,41,55,.9); color:#9CA3AF !important; border-radius:8px; padding:6px 18px; }
.stTabs [aria-selected="true"] { background:#34D399 !important; color:#000 !important; font-weight:600; }

/* ── st.table ── */
table { color: #E5E7EB !important; background: rgba(31,41,55,0.8) !important; border-radius: 8px; }
th { color: #34D399 !important; background: rgba(52,211,153,0.1) !important; }
td { color: #E5E7EB !important; border-color: #1F2937 !important; }

/* ── Info / success / warning boxes ── */
div[data-testid="stAlert"] p { color: #E5E7EB !important; }
.stInfo { background: rgba(96,165,250,.12) !important; border-left: 3px solid #60A5FA; border-radius:8px; }
.stSuccess { background: rgba(52,211,153,.12) !important; border-left: 3px solid #34D399; border-radius:8px; }
.stWarning { background: rgba(245,158,11,.12) !important; border-left: 3px solid #F59E0B; border-radius:8px; }

/* ── Multiselect ── */
[data-baseweb="tag"] { background: #1F2937 !important; color: #34D399 !important; }
[data-baseweb="tag"] span { color: #34D399 !important; }

/* ── Explanation cards ── */
.explain-box {
    background: rgba(31,41,55,0.6);
    border-left: 3px solid #60A5FA;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0 16px 0;
    color: #CBD5E1 !important;
    font-size: 0.88rem;
    line-height: 1.6;
}
.explain-box b { color: #93C5FD !important; }
.gauge-label { text-align:center; color:#9CA3AF !important; font-size:.8rem; margin-top:-12px; }

hr { border-color: #1F2937 !important; }
</style>
""", unsafe_allow_html=True)

# ── helpers ────────────────────────────────────────────────
SEASON_MAP       = {1:"🌸 Spring", 2:"☀️ Summer", 3:"🍂 Fall", 4:"❄️ Winter"}
SEASON_MAP_SHORT = {1:"Spring",    2:"Summer",    3:"Fall",   4:"Winter"}
PALETTE = ["#34D399","#60A5FA","#F59E0B","#F472B6"]

CHART_LAYOUT = dict(
    plot_bgcolor ="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E5E7EB", size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB")),
    margin=dict(t=45, b=10, l=10, r=10),
)
GRID = dict(showgrid=True, gridcolor="#1F2937", color="#E5E7EB", tickfont=dict(color="#E5E7EB"))

def apply_chart(fig):
    fig.update_layout(**CHART_LAYOUT)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig

def explain(text):
    st.markdown(f"<div class='explain-box'>{text}</div>", unsafe_allow_html=True)

# ── data & model ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("bike_rental_100_rows.csv")
    df["season_name"] = df["season"].map(SEASON_MAP_SHORT)
    return df

@st.cache_data
def train_model(df: pd.DataFrame):
    X = df[["temp","humidity","windspeed","season"]]
    y = df["count"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    return rf, {"r2": round(r2_score(y_te, y_pred), 3),
                "mae": round(mean_absolute_error(y_te, y_pred), 1)}

try:
    df = load_data()
    model, metrics = train_model(df)
except FileNotFoundError:
    st.error("❌ `bike_rental_100_rows.csv` not found — place it in the same folder as `app.py`.")
    st.stop()

# ── sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛠️ Prediction Controls")
    st.markdown("Adjust the sliders and season below. The prediction updates **instantly** on every change.")
    st.markdown("---")
    temp      = st.slider("🌡️ Temperature (normalised)", 0.0, 1.0, 0.50, 0.01, help="0 = freezing cold, 1 = hottest recorded")
    humidity  = st.slider("💧 Humidity (normalised)",    0.0, 1.0, 0.50, 0.01, help="0 = bone dry, 1 = fully saturated")
    windspeed = st.slider("💨 Wind Speed (normalised)",  0.0, 1.0, 0.20, 0.01, help="0 = still, 1 = storm-force wind")
    season    = st.selectbox("📅 Season", [1,2,3,4], format_func=lambda x: SEASON_MAP[x])
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    st.markdown(f"**R² Score** (test set): `{metrics['r2']}`")
    st.markdown(f"**MAE** (test set): `{metrics['mae']} bikes`")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Model: **Random Forest** (200 trees)\nDataset: 100 historical records\nTarget: bike rentals per hour")

# ── live prediction ────────────────────────────────────────
input_df   = pd.DataFrame({"temp":[temp],"humidity":[humidity],"windspeed":[windspeed],"season":[season]})
prediction = int(model.predict(input_df)[0])

# ── header ─────────────────────────────────────────────────
st.title("🚲 Intelligent Bike Rental Predictor")
st.markdown("*A real-time ML dashboard — adjust the sidebar sliders to forecast bike demand instantly.*")
st.markdown("---")

# ── KPI row ────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
k1.metric("🎯 Predicted Rentals", f"{prediction}",        "live forecast")
k2.metric("📏 R² Score",          f"{metrics['r2']}",     "test accuracy")
k3.metric("📉 MAE",               f"{metrics['mae']} bikes","mean abs. error")
k4.metric("📦 Training Samples",  f"{len(df)}",           "dataset rows")
explain("These four cards give a snapshot of the model's live prediction and its quality. "
        "<b>R²</b> (closer to 1 = better fit) and <b>MAE</b> (lower = fewer rental errors) are computed on a held-out 20 % test split — "
        "never seen during training — so they reflect real-world accuracy.")
st.markdown("---")

# ── tabs ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Data Explorer", "🔍 Model Insights"])

# ════════════════ TAB 1 – PREDICTION ════════════════
with tab1:
    left, right = st.columns([1,1], gap="large")

    with left:
        st.subheader("Your Input Parameters")
        explain("This table shows the values you chose on the left sidebar. "
                "Each feature is <b>normalised to [0, 1]</b> so the model treats all inputs on an equal scale.")
        display_df = pd.DataFrame({
            "Parameter": ["Temperature","Humidity","Wind Speed","Season"],
            "Value":     [f"{temp:.2f}", f"{humidity:.2f}", f"{windspeed:.2f}", SEASON_MAP[season]],
        })
        st.table(display_df)

        st.markdown("**Demand Level Gauge**")
        explain("The bar below shows where the current prediction sits relative to the <b>highest recorded demand</b> "
                "in the dataset. Green = high demand, Yellow = moderate, Red = low demand.")
        max_count = int(df["count"].max())
        gauge_pct = min(prediction / max_count, 1.0)
        level     = "🔴 Low" if gauge_pct < .33 else ("🟡 Moderate" if gauge_pct < .66 else "🟢 High")
        st.progress(gauge_pct)
        st.markdown(f"<p class='gauge-label'>{level} demand &nbsp;|&nbsp; {prediction} of {max_count} peak rentals</p>",
                    unsafe_allow_html=True)

    with right:
        st.subheader("🎯 Live Prediction")
        explain("The model predicts <b>how many bikes will be rented</b> given the current conditions. "
                "Change any slider on the left and this number updates immediately — no button needed.")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(52,211,153,.15),rgba(96,165,250,.1));
                    border:1px solid #34D399;border-radius:16px;padding:40px;text-align:center;margin:10px 0 24px;">
            <div style="font-size:4rem;font-weight:700;color:#34D399;">{prediction}</div>
            <div style="color:#9CA3AF;font-size:1rem;margin-top:6px;">estimated bike rentals</div>
            <div style="color:#60A5FA;font-size:.85rem;margin-top:4px;">{SEASON_MAP[season]}</div>
        </div>""", unsafe_allow_html=True)

        # Scenario comparison
        scenarios = {
            "❄️ Cold & Windy":   model.predict(pd.DataFrame({"temp":[0.1],"humidity":[0.6],"windspeed":[0.7],"season":[season]}))[0],
            "🌤️ Mild & Calm":    model.predict(pd.DataFrame({"temp":[0.5],"humidity":[0.5],"windspeed":[0.1],"season":[season]}))[0],
            "☀️ Hot & Dry":      model.predict(pd.DataFrame({"temp":[0.9],"humidity":[0.2],"windspeed":[0.05],"season":[season]}))[0],
            "🌧️ Humid & Stormy": model.predict(pd.DataFrame({"temp":[0.4],"humidity":[0.9],"windspeed":[0.6],"season":[season]}))[0],
        }
        sc_df = pd.DataFrame({"Scenario": list(scenarios.keys()),
                              "Predicted Rentals": [int(v) for v in scenarios.values()]})
        fig_sc = px.bar(sc_df, x="Scenario", y="Predicted Rentals",
                        color="Predicted Rentals",
                        color_continuous_scale=["#EF4444","#F59E0B","#34D399"],
                        title=f"Scenario Comparison — {SEASON_MAP[season]}",
                        text="Predicted Rentals")
        fig_sc.update_traces(textposition="outside", textfont_color="#E5E7EB")
        fig_sc.update_layout(coloraxis_showscale=False, showlegend=False, **CHART_LAYOUT)
        fig_sc.update_xaxes(**GRID)
        fig_sc.update_yaxes(**GRID)
        st.plotly_chart(fig_sc, use_container_width=True)
        explain("These bars compare four <b>preset weather scenarios</b> for the selected season. "
                "It shows how sensitive rental demand is to temperature and wind — "
                "hot & dry days consistently outperform cold & stormy ones.")

# ════════════════ TAB 2 – DATA EXPLORER ════════════════
with tab2:
    st.subheader("📊 Explore Historical Rental Data")
    explain("Use the filter below to explore the 100 historical rental records. "
            "Each chart reveals how a different environmental factor relates to rental demand. "
            "Points are colour-coded by season. Hover over any point for exact values.")

    selected_seasons = st.multiselect("Filter by Season", [1,2,3,4], default=[1,2,3,4],
                                      format_func=lambda x: SEASON_MAP_SHORT[x])
    if not selected_seasons:
        st.warning("Please select at least one season.")
        st.stop()

    fdf = df[df["season"].isin(selected_seasons)].copy()

    c1,c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(fdf, x="temp", y="count", color="season_name",
                          color_discrete_sequence=PALETTE,
                          title="🌡️ Temperature vs Rentals",
                          labels={"temp":"Temperature","count":"Rentals","season_name":"Season"},
                          hover_data=["humidity","windspeed"])
        # Manual trendline via numpy (no statsmodels needed)
        z = np.polyfit(fdf["temp"], fdf["count"], 1)
        x_line = np.linspace(fdf["temp"].min(), fdf["temp"].max(), 100)
        y_line = np.polyval(z, x_line)
        fig1.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                  line=dict(color="#F9FAFB", dash="dash", width=2),
                                  name="Trend", showlegend=False))
        apply_chart(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        explain("<b>Key insight:</b> Higher temperature strongly correlates with more rentals. "
                "The dashed trendline shows this upward relationship across all seasons.")

    with c2:
        fig2 = px.box(fdf, x="season_name", y="count", color="season_name",
                      color_discrete_sequence=PALETTE,
                      title="📦 Rental Distribution by Season",
                      labels={"season_name":"Season","count":"Rentals"})
        fig2.update_layout(showlegend=False, **CHART_LAYOUT)
        fig2.update_xaxes(**GRID)
        fig2.update_yaxes(**GRID)
        st.plotly_chart(fig2, use_container_width=True)
        explain("<b>Key insight:</b> The box plot shows median, spread, and outliers for each season. "
                "Summer and Fall tend to have higher and more consistent demand.")

    c3,c4 = st.columns(2)
    with c3:
        fig3 = px.scatter(fdf, x="humidity", y="count", color="season_name",
                          color_discrete_sequence=PALETTE,
                          title="💧 Humidity vs Rentals",
                          labels={"humidity":"Humidity","count":"Rentals","season_name":"Season"})
        apply_chart(fig3)
        st.plotly_chart(fig3, use_container_width=True)
        explain("<b>Key insight:</b> Very high humidity tends to reduce rentals. "
                "People prefer cycling in moderately dry conditions.")

    with c4:
        fig4 = px.scatter(fdf, x="windspeed", y="count", color="season_name",
                          color_discrete_sequence=PALETTE,
                          title="💨 Wind Speed vs Rentals",
                          labels={"windspeed":"Wind Speed","count":"Rentals","season_name":"Season"})
        apply_chart(fig4)
        st.plotly_chart(fig4, use_container_width=True)
        explain("<b>Key insight:</b> High wind speeds discourage cycling. "
                "Most high-rental days occur when wind speed is below 0.3 (normalised).")

    st.markdown("**Raw Data Table**")
    st.dataframe(fdf.drop(columns=["season_name"]), use_container_width=True, height=250)

# ════════════════ TAB 3 – MODEL INSIGHTS ════════════════
with tab3:
    st.subheader("🔍 Model Internals & Evaluation")
    explain("This tab opens the 'black box'. It shows <b>which features the model relies on most</b>, "
            "how well its predictions match the real data, and whether the errors are randomly distributed.")

    lm, rm = st.columns(2)

    with lm:
        feat_names  = ["Temperature","Humidity","Wind Speed","Season"]
        importances = model.feature_importances_
        idx         = np.argsort(importances)
        fig_imp = go.Figure(go.Bar(
            x=importances[idx], y=[feat_names[i] for i in idx], orientation="h",
            marker=dict(color=importances[idx],
                        colorscale=[[0,"#1E3A5F"],[0.5,"#60A5FA"],[1,"#34D399"]]),
        ))
        fig_imp.update_layout(title="🎯 Feature Importances",
                              xaxis_title="Importance Score", yaxis_title="",
                              **CHART_LAYOUT)
        fig_imp.update_xaxes(**GRID)
        fig_imp.update_yaxes(tickfont=dict(color="#E5E7EB"), gridcolor="#1F2937")
        st.plotly_chart(fig_imp, use_container_width=True)
        explain("<b>Temperature</b> is by far the most important predictor — it alone explains the majority of demand variation. "
                "<b>Season</b> comes second (which makes sense as it encodes temperature trends). "
                "Humidity and Wind Speed have a smaller but still meaningful role.")

    with rm:
        X_full = df[["temp","humidity","windspeed","season"]]
        y_full = df["count"]
        y_pred = model.predict(X_full)
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(x=y_full, y=y_pred, mode="markers",
                                     marker=dict(color="#60A5FA", size=7, opacity=.8), name="Predictions"))
        mn, mx = y_full.min(), y_full.max()
        fig_avp.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                     line=dict(color="#34D399", dash="dash", width=2), name="Perfect Fit"))
        fig_avp.update_layout(title="✅ Actual vs Predicted Rentals",
                              xaxis_title="Actual Rentals", yaxis_title="Predicted Rentals",
                              **CHART_LAYOUT)
        fig_avp.update_xaxes(**GRID)
        fig_avp.update_yaxes(**GRID)
        st.plotly_chart(fig_avp, use_container_width=True)
        explain("Points close to the <b>green dashed line</b> (perfect fit) mean the model predicted almost exactly right. "
                "The tighter the cluster around that line, the better the model. "
                "Outliers show cases where the model over- or under-predicted.")

    residuals = y_pred - y_full.values
    fig_res = px.histogram(x=residuals, nbins=20,
                           title="📉 Residual Distribution (Predicted − Actual)",
                           labels={"x":"Prediction Error","y":"Frequency"},
                           color_discrete_sequence=["#F59E0B"])
    fig_res.add_vline(x=0, line_dash="dash", line_color="#34D399", annotation_text="Zero Error",
                      annotation_font_color="#34D399")
    apply_chart(fig_res)
    st.plotly_chart(fig_res, use_container_width=True)
    explain("A good model has errors <b>centred around zero</b> and roughly bell-shaped. "
            "This means the model neither consistently over-predicts nor under-predicts. "
            "If the histogram leans left or right, the model has a systematic bias.")

# ── footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align:center;color:#4B5563;font-size:.8rem;'>"
            "🚲 Bike Rental Predictor &nbsp;|&nbsp; Random Forest · 200 Trees &nbsp;|&nbsp; Streamlit + Plotly"
            "</div>", unsafe_allow_html=True)
