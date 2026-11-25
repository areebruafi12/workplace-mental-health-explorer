from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "mental_health_treatment_model.pkl"
DATA_PATH = BASE_DIR / "mental_health_tech_clean.csv"

FEATURE_LABELS = {
    "self_employed": "Self-employed",
    "company_size": "Company size",
    "remote_work": "Works remotely at least 50% of the time",
    "tech_company": "Employer is primarily a tech company",
    "mental_health_benefits": "Employer provides mental-health benefits",
    "care_options": "Knows available mental-health care options",
    "wellness_program": "Employer discusses mental health in a wellness program",
    "seek_help_resources": "Employer provides resources to seek help",
    "anonymity_protection": "Anonymity is believed to be protected",
    "mental_health_leave_ease": "Ease of taking mental-health medical leave",
    "mental_health_discussion_consequence": "Negative consequence expected for mental-health discussion",
    "physical_health_discussion_consequence": "Negative consequence expected for physical-health discussion",
    "coworker_discussion_comfort": "Comfort discussing mental health with coworkers",
    "supervisor_discussion_comfort": "Comfort discussing mental health with a supervisor",
    "mental_health_interview": "Would raise a mental-health issue in an interview",
    "physical_health_interview": "Would raise a physical-health issue in an interview",
    "mental_vs_physical_parity": "Employer treats mental and physical health equally",
    "observed_negative_consequences": "Observed negative consequences for coworkers",
}

FEATURE_GROUPS = {
    "Workplace setup": [
        "self_employed",
        "company_size",
        "remote_work",
        "tech_company",
    ],
    "Support and privacy": [
        "mental_health_benefits",
        "care_options",
        "wellness_program",
        "seek_help_resources",
        "anonymity_protection",
        "mental_health_leave_ease",
    ],
}

st.set_page_config(
    page_title="Workplace Mental Health Survey Explorer",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
            radial-gradient(950px 520px at 8% -12%, rgba(124,92,255,.25), transparent 62%),
            radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,.16), transparent 58%),
            #0b1020 !important;
        color: #f8fafc !important;
    }
    header[data-testid="stHeader"] { background: rgba(11,16,32,.82) !important; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3.5rem; }

    h1, h2, h3, p, span, li, [data-testid="stMarkdownContainer"] { color: #f8fafc; }
    [data-testid="stCaptionContainer"] { color: #b9c2ea !important; }

    .hero {
        position: relative; overflow: hidden;
        padding: 2.25rem 2.4rem; border-radius: 24px; margin-bottom: 1.25rem;
        background: linear-gradient(120deg, #7c5cff 0%, #3b82f6 55%, #22d3ee 100%);
        box-shadow: 0 20px 52px rgba(59,130,246,.25);
    }
    .hero::after {
        content: "🧠"; position: absolute; right: 2.1rem; top: 50%;
        transform: translateY(-50%); font-size: 6.4rem; opacity: .18;
    }
    .eyebrow { font-size:.75rem; letter-spacing:.16em; font-weight:700; opacity:.8; }
    .hero h1 { font-size: 2.35rem; line-height:1.15; margin:.35rem 0 .55rem; color:#fff !important; }
    .hero p { max-width:770px; margin:0; color:rgba(255,255,255,.92) !important; font-size:1.02rem; }
    .chips { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.15rem; }
    .chip { border:1px solid rgba(255,255,255,.25); background:rgba(8,15,40,.18);
        border-radius:999px; padding:.28rem .75rem; color:white !important; font-size:.78rem; font-weight:600; }

    .notice {
        border: 1px solid rgba(245,158,11,.35); border-left:4px solid #f59e0b;
        background: rgba(120,77,5,.28); border-radius:14px; padding:1rem 1.1rem;
        color:#fde68a !important; margin:0 0 1.15rem;
    }
    .notice b { color:#fbbf24 !important; }

    button[data-baseweb="tab"] {
        color:#9aa7d5 !important; font-weight:650; font-size:.96rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] { color:#ffffff !important; }
    [data-baseweb="tab-highlight"] { background:#7c5cff !important; height:3px !important; }
    [data-baseweb="tab-border"] { background:rgba(124,140,255,.32) !important; }

    div[data-testid="stForm"] {
        background:rgba(20,27,52,.82); border:1px solid rgba(124,140,255,.28);
        border-radius:20px; padding:1.45rem 1.35rem 1.1rem;
        box-shadow:0 12px 34px rgba(0,0,0,.28);
    }
    .form-section {
        color:#8be9fd !important; font-size:.79rem; font-weight:800; letter-spacing:.12em;
        text-transform:uppercase; margin:1rem 0 .55rem; padding-bottom:.42rem;
        border-bottom:1px solid rgba(124,140,255,.22);
    }
    label, label p, [data-testid="stWidgetLabel"] p {
        color:#e5e7eb !important; font-weight:600 !important; font-size:.87rem !important;
    }
    div[data-baseweb="select"] > div {
        background:#0f1630 !important; border:1px solid rgba(124,140,255,.34) !important;
        border-radius:12px !important; min-height:45px;
    }
    div[data-baseweb="select"] *, [data-baseweb="select"] span {
        color:#f8fafc !important;
    }
    div[role="listbox"], div[data-baseweb="menu"] {
        background:#111a35 !important; border:1px solid rgba(124,140,255,.36) !important;
    }
    div[role="option"] { color:#f8fafc !important; background:#111a35 !important; }
    div[role="option"]:hover { background:#25345f !important; }

    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        border-radius:12px !important; font-weight:700 !important; transition:all .2s ease;
    }
    div[data-testid="stFormSubmitButton"] > button {
        width:100%; border:none !important; padding:.78rem 1rem !important; font-size:1.02rem !important;
        color:#fff !important; background:linear-gradient(90deg,#7c5cff,#3b82f6) !important;
        box-shadow:0 10px 25px rgba(124,92,255,.35);
    }
    div[data-testid="stFormSubmitButton"] > button:hover, .stButton > button:hover {
        transform:translateY(-2px); filter:brightness(1.12);
    }
    .stButton > button {
        background:#172448 !important; border:1px solid rgba(124,140,255,.35) !important; color:#fff !important;
    }

    [data-testid="stMetric"] {
        background:rgba(20,27,52,.84); border:1px solid rgba(124,140,255,.22);
        padding:1rem 1.05rem; border-radius:16px;
    }
    [data-testid="stMetricLabel"] p { color:#aeb9e7 !important; }
    [data-testid="stMetricValue"] { color:#ffffff !important; }

    .prediction-card {
        padding:1.5rem; border-radius:20px; text-align:center; color:#fff;
        background:linear-gradient(135deg,#17224a,#293f8f);
        border:1px solid rgba(167,139,250,.45); box-shadow:0 16px 36px rgba(0,0,0,.3);
    }
    .prediction-card .small { color:#b9c2ea !important; font-size:.78rem; letter-spacing:.13em; text-transform:uppercase; }
    .prediction-card .value { color:#fff !important; font-size:2.4rem; font-weight:800; margin:.35rem 0; }
    .prediction-card .detail { color:#dbeafe !important; font-size:.92rem; }

    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#10173a 0%,#0b1020 100%) !important;
        border-right:1px solid rgba(124,140,255,.18);
    }
    .side-stat { display:flex; justify-content:space-between; gap:.5rem; padding:.48rem 0;
        border-bottom:1px dashed rgba(124,140,255,.18); color:#bbc5ee; font-size:.87rem; }
    .side-stat b { color:#fff; text-align:right; }

    [data-testid="stDataFrame"] { border:1px solid rgba(124,140,255,.24); border-radius:12px; overflow:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_value(value: str) -> str:
    special = {
        "dont_know": "Don't know",
        "not_sure": "Not sure",
        "some_of_them": "Some of them",
        "somewhat_difficult": "Somewhat difficult",
        "somewhat_easy": "Somewhat easy",
        "very_difficult": "Very difficult",
        "very_easy": "Very easy",
        "more_than_1000": "More than 1000",
        "1_5": "1-5",
        "6_25": "6-25",
        "26_100": "26-100",
        "100_500": "100-500",
        "500_1000": "500-1000",
    }
    return special.get(value, value.replace("_", " ").title())


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


if not MODEL_PATH.exists() or not DATA_PATH.exists():
    missing = [path.name for path in (MODEL_PATH, DATA_PATH) if not path.exists()]
    st.error("Missing project files: " + ", ".join(missing))
    st.stop()

artifact = load_artifact()
df = load_data()
model = artifact["pipeline"]
feature_names = artifact["feature_names"]
options = artifact["options"]
defaults = artifact["defaults"]
metrics = artifact["metrics"]

SAMPLE_PROFILE = {
    "self_employed": "no",
    "company_size": "26_100",
    "remote_work": "yes",
    "tech_company": "yes",
    "mental_health_benefits": "yes",
    "care_options": "yes",
    "wellness_program": "yes",
    "seek_help_resources": "yes",
    "anonymity_protection": "yes",
    "mental_health_leave_ease": "somewhat_easy",
}


def load_sample_profile() -> None:
    for feature, value in SAMPLE_PROFILE.items():
        if value in options[feature]:
            st.session_state[f"input_{feature}"] = value


with st.sidebar:
    st.markdown("### 🧠 Project dashboard")
    st.markdown(
        f"""
        <div class="side-stat"><span>Algorithm</span><b>Logistic Regression</b></div>
        <div class="side-stat"><span>Survey records</span><b>{len(df):,}</b></div>
        <div class="side-stat"><span>Input features</span><b>{len(feature_names)}</b></div>
        <div class="side-stat"><span>Validation F1</span><b>{metrics["f1"]:.1%}</b></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🎲 Quick start")
    st.caption("Load one realistic workplace-policy example into the form.")
    st.button(
        "Load sample profile",
        on_click=load_sample_profile,
        use_container_width=True,
    )
    st.markdown("### 🔎 Use it correctly")
    st.caption(
        "Explore historical survey patterns only. This is not a clinical, hiring, "
        "or employee-screening tool."
    )

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">2014 OSMI DATASET · CLASSIFICATION PROJECT</div>
        <h1>Workplace Mental Health Survey Explorer</h1>
        <p>Explore workplace support patterns and interact with an educational
        treatment-seeking response classifier.</p>
        <div class="chips">
            <span class="chip">⚡ Instant model demo</span>
            <span class="chip">📊 Survey insights</span>
            <span class="chip">🔒 No demographic inputs</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="notice">
        <b>Important:</b> This is a class-project demo using anonymized survey data.
        It estimates a survey-response pattern, not anyone's mental health, diagnosis,
        or need for treatment. Do not use it for screening, employment, insurance,
        or any decision about a person.
    </div>
    """,
    unsafe_allow_html=True,
)

predict_tab, insights_tab, data_tab = st.tabs(
    ["Model demo", "Survey insights", "Dataset"]
)

with predict_tab:
    st.subheader("Educational survey-response demo")
    st.caption(
        "Ten focused workplace-support inputs, arranged in two simple sections."
    )

    with st.form("survey_form"):
        row = {}
        for section_title, section_features in FEATURE_GROUPS.items():
            active_features = [feature for feature in section_features if feature in feature_names]
            if not active_features:
                continue
            st.markdown(
                f'<div class="form-section">{section_title}</div>',
                unsafe_allow_html=True,
            )
            columns = st.columns(2)
            for index, feature in enumerate(active_features):
                with columns[index % 2]:
                    choices = options[feature]
                    state_key = f"input_{feature}"
                    saved_value = st.session_state.get(state_key, defaults[feature])
                    default_index = choices.index(saved_value)
                    row[feature] = st.selectbox(
                        FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                        choices,
                        index=default_index,
                        format_func=display_value,
                        key=state_key,
                    )

        submitted = st.form_submit_button(
            "Analyse workplace survey pattern",
            use_container_width=True,
        )

    if submitted:
        # Keep the app compatible if an older saved model is still present in a
        # deployment. The visible form supplies the current 10 inputs; any
        # older model-only columns receive their saved default value.
        model_row = {
            feature: defaults.get(feature, "unknown")
            for feature in feature_names
        }
        model_row.update(row)
        input_data = pd.DataFrame([model_row], columns=feature_names)
        probabilities = model.predict_proba(input_data)[0]
        class_probabilities = dict(zip(model.classes_, probabilities))
        probability_yes = float(class_probabilities[1])
        predicted_label = "Treatment sought (1)" if probability_yes >= 0.5 else "No treatment sought (0)"

        left, right = st.columns([1, 1.25], gap="large")
        with left:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="small">Estimated survey-response probability</div>
                    <div class="value">{probability_yes:.1%}</div>
                    <div class="detail">{predicted_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "This is a pattern in historical survey data, not a statement "
                "about any individual."
            )
        with right:
            chart_data = pd.DataFrame(
                {
                    "Survey response": ["No treatment sought", "Treatment sought"],
                    "Probability": [1 - probability_yes, probability_yes],
                }
            ).set_index("Survey response")
            st.bar_chart(chart_data, color="#0f766e", height=220)

with insights_tab:
    st.subheader("Quick dataset view")
    first, second, third, fourth = st.columns(4)
    first.metric("Survey responses", f"{len(df):,}")
    second.metric("Features in app", len(feature_names))
    third.metric("Validation accuracy", f"{metrics['accuracy']:.1%}")
    fourth.metric("Validation F1 score", f"{metrics['f1']:.1%}")

    st.caption(
        "Metrics are from one stratified 80/20 validation split, not proof of "
        "real-world clinical or employment usefulness."
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Treatment-seeking response balance")
        response_counts = df["treatment_sought"].value_counts().sort_index()
        response_counts.index = ["No (0)", "Yes (1)"]
        st.bar_chart(response_counts, color="#2563eb")
    with right:
        st.markdown("#### Reported treatment-seeking by benefit availability")
        benefit_view = pd.crosstab(
            df["mental_health_benefits"],
            df["treatment_sought"],
            normalize="index",
        ).reindex(columns=[0, 1], fill_value=0)
        benefit_view.columns = ["No (0)", "Yes (1)"]
        st.bar_chart(benefit_view, color=["#94a3b8", "#0f766e"])

    st.markdown("#### Average survey pattern by company size")
    company_view = pd.crosstab(
        df["company_size"],
        df["treatment_sought"],
        normalize="index",
    ).reindex(columns=[1], fill_value=0)
    company_view.columns = ["Treatment sought rate"]
    st.bar_chart(company_view, color="#0f766e")

with data_tab:
    st.subheader("Clean project dataset")
    st.caption(
        "The cleaned file retains all 1,259 survey responses. Timestamps, granular "
        "state data, and free-text comments were removed."
    )
    selected_columns = st.multiselect(
        "Columns to display",
        df.columns.tolist(),
        default=df.columns.tolist(),
    )
    st.dataframe(
        df[selected_columns],
        use_container_width=True,
        hide_index=True,
        height=460,
    )
    st.download_button(
        "Download cleaned CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="mental_health_tech_clean.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "Source: OSMI Mental Health in Tech Survey (2014), distributed via Kaggle. "
    "For educational use only."
)

