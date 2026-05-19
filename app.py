"""
Government Medical Supply Chain & Fund Management System
Premium Streamlit Dashboard with AI-Powered Recommendations
============================================================
5-Tab Interface:
  1. National Dashboard
  2. State Deep Dive
  3. 2028-29 Predictions
  4. AI Recommendations
  5. Policy Chat (SLM + RAG)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import joblib
import requests

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="GovHealth AI - Medical Supply Chain & Fund Management",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Main title gradient */
    .main-title {
        background: linear-gradient(135deg, #0077B6, #00B4D8, #48CAE4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0;
        padding: 0.5rem 0;
    }
    
    .subtitle {
        color: #90E0EF;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(0, 119, 182, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 180, 216, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 0.5rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 119, 182, 0.2);
        border-color: rgba(0, 180, 216, 0.5);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #48CAE4;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #ADE8F4;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-delta {
        font-size: 0.75rem;
        color: #00B4D8;
        margin-top: 0.2rem;
    }

    /* Priority badges */
    .badge-critical {
        background: linear-gradient(135deg, #d63031, #ff7675);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .badge-high {
        background: linear-gradient(135deg, #e17055, #fab1a0);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .badge-moderate {
        background: linear-gradient(135deg, #fdcb6e, #f9ca24);
        color: #2d3436;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .badge-low {
        background: linear-gradient(135deg, #00b894, #55efc4);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Recommendation cards */
    .rec-card {
        background: rgba(0, 119, 182, 0.08);
        border-left: 4px solid #00B4D8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0, 119, 182, 0.1);
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 180, 216, 0.3);
    }

    /* Section headers */
    .section-header {
        color: #48CAE4;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid rgba(0, 180, 216, 0.3);
    }

    /* Chat styling */
    .stChatMessage {
        border-radius: 12px;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #023E8A 0%, #0077B6 100%);
    }
    
    div[data-testid="stSidebar"] .stMarkdown {
        color: #CAF0F8;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Load Data & Models
# -------------------------
@st.cache_data
def load_healthcare_data():
    return pd.read_csv("data/india_healthcare_data.csv")

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv("data/predictions_2028_29.csv")
    except:
        return None

@st.cache_data
def load_predictions_json():
    try:
        with open("data/predictions_2028_29.json") as f:
            return json.load(f)
    except:
        return None

@st.cache_resource
def load_recommendation_engine():
    from recommendation_engine import HealthcareRecommendationEngine
    return HealthcareRecommendationEngine()

@st.cache_resource
def load_vectorstore():
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.warning(f"Vector store not loaded: {e}")
        return None

@st.cache_resource
def load_budget_model():
    try:
        return joblib.load("budget_predictor.pkl")
    except:
        return None

@st.cache_resource
def load_llm():
    model_name = os.getenv("OLLAMA_MODEL", "mistral:7b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        available_models = {
            model.get("name")
            for model in response.json().get("models", [])
        }
        if model_name not in available_models:
            st.warning(
                f"Ollama is running, but `{model_name}` is not installed. "
                f"Run `ollama pull {model_name}` once, then reload this page."
            )
            return None, None
        return None, {"model": model_name, "base_url": base_url}
    except Exception as e:
        st.warning(
            "Ollama is not available. Install/start Ollama and run "
            f"`ollama pull {model_name}`. Details: {e}"
        )
        return None, None


def stream_ollama_chat(llm_model, messages, max_tokens):
    payload = {
        "model": llm_model["model"],
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_predict": max_tokens,
        },
    }

    with requests.post(
        f"{llm_model['base_url']}/api/chat",
        json=payload,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            content = chunk.get("message", {}).get("content")
            if content:
                yield content


# -------------------------
# Helper Functions
# -------------------------
def render_metric_card(label, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """

def get_priority_badge(category):
    badge_class = f"badge-{category.lower()}"
    return f'<span class="{badge_class}">{category}</span>'

def format_number(n):
    if n >= 10000000:
        return f"{n/10000000:.1f} Cr"
    elif n >= 100000:
        return f"{n/100000:.1f} L"
    elif n >= 1000:
        return f"{n/1000:.1f} K"
    return str(int(n))


# -------------------------
# Load All Data
# -------------------------
df = load_healthcare_data()
predictions_df = load_predictions()
engine = load_recommendation_engine()
states_list = sorted(df['state'].unique().tolist())
latest_year = min(df['year'].max(), 2026)  # Current real year is 2026
latest_data = df[df['year'] == latest_year]

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown('<p class="main-title" style="font-size:1.5rem;">🏥 GovHealth AI</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #CAF0F8; font-size: 0.85rem; margin-top:-10px;">
    Government Medical Supply Chain<br>& Fund Management System
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # National Stats
    total_pop = latest_data['population_crore'].sum()
    total_budget = latest_data['health_budget_crore'].sum()
    total_hospitals = int(latest_data['hospitals_total'].sum())
    total_doctors = int(latest_data['doctors_total'].sum())
    
    st.markdown("##### 📊 National Overview")
    st.metric("Population", f"{total_pop:.1f} Cr")
    st.metric("Total Budget", f"₹{total_budget:,.0f} Cr")
    st.metric("Hospitals", f"{total_hospitals:,}")
    st.metric("Doctors", f"{total_doctors:,}")
    
    st.markdown("---")
    st.markdown("""
    <p style="color: #90E0EF; font-size: 0.75rem;">
    🤖 Powered by Ollama Mistral 7B<br>
    📊 ML Predictions: GBR + RF<br>
    🔍 RAG: FAISS + MiniLM<br>
    💡 100% Offline Operation
    </p>
    """, unsafe_allow_html=True)


# -------------------------
# Main Title
# -------------------------
st.markdown('<p class="main-title">Government Medical Supply Chain & Fund Management</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Decision Support for Healthcare Resource Allocation & Budget Planning</p>', unsafe_allow_html=True)

# -------------------------
# Tab Layout
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 National Dashboard",
    "🏥 State Deep Dive",
    "🔮 2028-29 Predictions",
    "💡 AI Recommendations",
    "💬 Policy Chat"
])


# =============================
# TAB 1: National Dashboard
# =============================
with tab1:
    st.markdown('<div class="section-header">National Health KPIs</div>', unsafe_allow_html=True)
    
    avg_vaccine = latest_data['vaccine_coverage_pct'].mean()
    avg_beds = latest_data['hospital_beds_per_1000'].mean()
    avg_doctors = latest_data['doctor_per_1000'].mean()
    avg_life = latest_data['life_expectancy'].mean()
    total_nurses = int(latest_data['nurses_total'].sum())
    
    # KPI Row
    cols = st.columns(6)
    with cols[0]:
        st.markdown(render_metric_card("Total Population", f"{total_pop:.1f} Cr", "36 States/UTs"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(render_metric_card("Health Budget", f"₹{total_budget/1000:.0f}K Cr", f"~{total_budget/total_pop/100:.0f}₹/capita"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(render_metric_card("Hospitals", f"{total_hospitals:,}", f"+ Clinics & PHCs"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(render_metric_card("Doctors", f"{format_number(total_doctors)}", f"Avg {avg_doctors:.2f}/1000"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(render_metric_card("Vaccine Coverage", f"{avg_vaccine:.1f}%", "National Average"), unsafe_allow_html=True)
    with cols[5]:
        st.markdown(render_metric_card("Life Expectancy", f"{avg_life:.1f} yr", "National Average"), unsafe_allow_html=True)
    
    st.markdown("")

    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">State-wise Health Budget (₹ Crore)</div>', unsafe_allow_html=True)
        budget_data = latest_data.sort_values('health_budget_crore', ascending=True).tail(15)
        fig = px.bar(
            budget_data, x='health_budget_crore', y='state',
            orientation='h',
            color='health_budget_crore',
            color_continuous_scale='Tealgrn',
            labels={'health_budget_crore': 'Budget (₹ Cr)', 'state': ''}
        )
        fig.update_layout(
            height=450, showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#CAF0F8'),
            xaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
            yaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-header">State-wise Infrastructure Gap Score</div>', unsafe_allow_html=True)
        gap_data = latest_data.sort_values('infra_gap_score', ascending=False).head(15)
        fig = px.bar(
            gap_data, x='infra_gap_score', y='state',
            orientation='h',
            color='infra_gap_score',
            color_continuous_scale='RdYlGn_r',
            labels={'infra_gap_score': 'Gap Score (1-10)', 'state': ''}
        )
        fig.update_layout(
            height=450, showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#CAF0F8'),
            xaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
            yaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, width='stretch')

    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Doctor Density vs WHO Standard</div>', unsafe_allow_html=True)
        doc_data = latest_data.sort_values('doctor_per_1000', ascending=True)
        colors = ['#ff7675' if x < 0.5 else '#fdcb6e' if x < 1.0 else '#00b894' 
                  for x in doc_data['doctor_per_1000']]
        fig = go.Figure(data=[go.Bar(
            y=doc_data['state'], x=doc_data['doctor_per_1000'],
            orientation='h', marker_color=colors
        )])
        fig.add_vline(x=1.0, line_dash="dash", line_color="#48CAE4",
                      annotation_text="WHO Min: 1.0")
        fig.update_layout(
            height=600,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#CAF0F8', size=10),
            xaxis=dict(title='Doctors per 1,000 Population', gridcolor='rgba(144,224,239,0.1)'),
            yaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-header">Vaccine Coverage by State (%)</div>', unsafe_allow_html=True)
        vax_data = latest_data.sort_values('vaccine_coverage_pct', ascending=True)
        colors = ['#ff7675' if x < 75 else '#fdcb6e' if x < 85 else '#00b894' 
                  for x in vax_data['vaccine_coverage_pct']]
        fig = go.Figure(data=[go.Bar(
            y=vax_data['state'], x=vax_data['vaccine_coverage_pct'],
            orientation='h', marker_color=colors
        )])
        fig.add_vline(x=95, line_dash="dash", line_color="#48CAE4",
                      annotation_text="Target: 95%")
        fig.update_layout(
            height=600,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#CAF0F8', size=10),
            xaxis=dict(title='Vaccination Coverage (%)', gridcolor='rgba(144,224,239,0.1)', range=[50, 100]),
            yaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, width='stretch')

    # National Budget Trend
    st.markdown('<div class="section-header">National Health Budget Trend (2020-2027)</div>', unsafe_allow_html=True)
    yearly_budget = df.groupby('year')['health_budget_crore'].sum().reset_index()
    fig = px.area(
        yearly_budget, x='year', y='health_budget_crore',
        labels={'health_budget_crore': 'Total Budget (₹ Cr)', 'year': 'Year'},
        color_discrete_sequence=['#00B4D8']
    )
    fig.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#CAF0F8'),
        xaxis=dict(gridcolor='rgba(144,224,239,0.1)', dtick=1),
        yaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    fig.update_traces(fill='tozeroy', line=dict(width=3))
    st.plotly_chart(fig, width='stretch')


# =============================
# TAB 2: State Deep Dive
# =============================
with tab2:
    st.markdown('<div class="section-header">State Healthcare Analysis</div>', unsafe_allow_html=True)
    
    col_select, col_compare = st.columns([1, 1])
    with col_select:
        selected_state = st.selectbox("Select State", states_list, key="state_dive")
    with col_compare:
        compare_state = st.selectbox("Compare with (optional)", ["None"] + states_list, key="state_compare")

    state_data = latest_data[latest_data['state'] == selected_state].iloc[0]
    state_history = df[df['state'] == selected_state].sort_values('year')

    # State Metrics
    st.markdown(f'<div class="section-header">{selected_state} — Key Indicators ({latest_year})</div>', unsafe_allow_html=True)
    
    cols = st.columns(5)
    with cols[0]:
        st.markdown(render_metric_card("Population", f"{state_data['population_crore']:.2f} Cr",
                    f"Urban: {state_data['urban_pct']:.1f}%"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(render_metric_card("Health Budget", f"₹{state_data['health_budget_crore']:,.0f} Cr",
                    f"₹{state_data['budget_per_capita_inr']:,.0f}/capita"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(render_metric_card("Hospitals", f"{state_data['hospitals_total']:,}",
                    f"{state_data['hospital_beds_per_1000']:.2f} beds/1000"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(render_metric_card("Doctors", f"{state_data['doctors_total']:,}",
                    f"{state_data['doctor_per_1000']:.2f}/1000 pop"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(render_metric_card("Vaccines", f"{state_data['vaccine_coverage_pct']:.1f}%",
                    f"{state_data['cold_chain_facilities']} cold chain"), unsafe_allow_html=True)

    cols2 = st.columns(5)
    with cols2[0]:
        st.markdown(render_metric_card("ICU Beds", f"{state_data['icu_beds']:,}",
                    f"Gap Score: {state_data['infra_gap_score']}"), unsafe_allow_html=True)
    with cols2[1]:
        st.markdown(render_metric_card("Nurses", f"{state_data['nurses_total']:,}",
                    f"PHCs: {int(state_data.get('phc_count', 0))}"), unsafe_allow_html=True)
    with cols2[2]:
        st.markdown(render_metric_card("MMR", f"{state_data['maternal_mortality_ratio']:.0f}",
                    f"Target: <70"), unsafe_allow_html=True)
    with cols2[3]:
        st.markdown(render_metric_card("IMR", f"{state_data['infant_mortality_rate']:.0f}",
                    f"Target: <15"), unsafe_allow_html=True)
    with cols2[4]:
        st.markdown(render_metric_card("Life Expectancy", f"{state_data['life_expectancy']:.1f} yr",
                    f"Disease Index: {state_data['disease_index']:.2f}"), unsafe_allow_html=True)

    # Radar Chart: State vs National Average
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">State vs National Average</div>', unsafe_allow_html=True)
        categories = ['Doctor Ratio', 'Beds/1000', 'Vaccine %', 'Life Expect.', 'Budget/Capita']
        
        # Normalize to 0-100 scale for radar
        national_avg = latest_data[['doctor_per_1000', 'hospital_beds_per_1000', 'vaccine_coverage_pct', 'life_expectancy', 'budget_per_capita_inr']].mean()
        state_vals = [
            min(100, state_data['doctor_per_1000'] / 2 * 100),
            min(100, state_data['hospital_beds_per_1000'] / 3 * 100),
            state_data['vaccine_coverage_pct'],
            min(100, (state_data['life_expectancy'] - 60) / 20 * 100),
            min(100, state_data['budget_per_capita_inr'] / 7000 * 100)
        ]
        national_vals = [
            min(100, national_avg['doctor_per_1000'] / 2 * 100),
            min(100, national_avg['hospital_beds_per_1000'] / 3 * 100),
            national_avg['vaccine_coverage_pct'],
            min(100, (national_avg['life_expectancy'] - 60) / 20 * 100),
            min(100, national_avg['budget_per_capita_inr'] / 7000 * 100)
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=state_vals + [state_vals[0]], theta=categories + [categories[0]],
            fill='toself', name=selected_state,
            fillcolor='rgba(0, 180, 216, 0.3)', line=dict(color='#00B4D8', width=2)
        ))
        fig.add_trace(go.Scatterpolar(
            r=national_vals + [national_vals[0]], theta=categories + [categories[0]],
            fill='toself', name='National Avg',
            fillcolor='rgba(144, 224, 239, 0.15)', line=dict(color='#90E0EF', width=2, dash='dot')
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(144,224,239,0.2)'),
                angularaxis=dict(gridcolor='rgba(144,224,239,0.2)')
            ),
            height=400, showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#CAF0F8'),
            legend=dict(x=0.8, y=1.15),
            margin=dict(l=30, r=30, t=30, b=30)
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-header">Gap Analysis</div>', unsafe_allow_html=True)
        gaps = engine.calculate_gap_analysis(selected_state)
        if gaps:
            gap_items = [
                ("Doctor Gap", gaps['doctor_gap'], f"{gaps['doctor_ratio']:.2f} → 1.0/1000"),
                ("Bed Gap", gaps['additional_beds_needed'], f"{gaps['current_beds_per_1000']:.2f} → 3.0/1000"),
                ("ICU Gap", gaps['icu_gap'], f"{gaps['current_icu_beds']:,} → {gaps['required_icu_beds']:,}"),
                ("Vaccine Gap", f"{gaps['vaccine_coverage_gap']:.1f}%", f"{gaps['current_vaccine_coverage']:.1f}% → 95%"),
                ("Budget Gap", f"₹{gaps['budget_gap_crore']:,.0f} Cr", f"₹{gaps['current_per_capita']:,.0f} → ₹3,000/capita"),
            ]
            
            for item_name, gap_val, detail in gap_items:
                gap_str = f"{gap_val:,}" if isinstance(gap_val, int) else str(gap_val)
                if isinstance(gap_val, (int, float)) and gap_val > 0:
                    st.error(f"**{item_name}**: {gap_str} — {detail}")
                elif isinstance(gap_val, str) and gap_val.replace('.', '').replace('%', '') != '0':
                    st.warning(f"**{item_name}**: {gap_str} — {detail}")
                else:
                    st.success(f"**{item_name}**: Met benchmark ✓ — {detail}")

    # Historical Trends
    st.markdown(f'<div class="section-header">{selected_state} — Historical Trends</div>', unsafe_allow_html=True)
    
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=state_history['year'], y=state_history['health_budget_crore'],
            name='Budget (₹ Cr)', fill='tozeroy', line=dict(color='#00B4D8', width=2),
            fillcolor='rgba(0, 180, 216, 0.2)'
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=state_history['year'], y=state_history['hospitals_total'],
            name='Hospitals', line=dict(color='#48CAE4', width=2, dash='dot')
        ), secondary_y=True)
        fig.update_layout(
            title="Budget & Hospitals", height=300,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#CAF0F8'),
            xaxis=dict(dtick=1, gridcolor='rgba(144,224,239,0.1)'),
            yaxis=dict(title='Budget (₹ Cr)', gridcolor='rgba(144,224,239,0.1)'),
            yaxis2=dict(title='Hospitals', gridcolor='rgba(144,224,239,0.1)'),
            legend=dict(x=0, y=1.15, orientation='h'),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, width='stretch')

    with trend_col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=state_history['year'], y=state_history['vaccine_coverage_pct'],
            name='Vaccine Coverage', line=dict(color='#00B894', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=state_history['year'], y=state_history['doctor_per_1000'] * 50,
            name='Doctor Ratio (×50)', line=dict(color='#FDCB6E', width=2, dash='dot')
        ))
        fig.add_hline(y=95, line_dash="dash", line_color="rgba(255,118,117,0.5)",
                      annotation_text="95% Target")
        fig.update_layout(
            title="Vaccine Coverage & Doctor Ratio", height=300,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#CAF0F8'),
            xaxis=dict(dtick=1, gridcolor='rgba(144,224,239,0.1)'),
            yaxis=dict(gridcolor='rgba(144,224,239,0.1)', range=[0, 100]),
            legend=dict(x=0, y=1.15, orientation='h'),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, width='stretch')

    # State Comparison
    if compare_state != "None":
        st.markdown(f'<div class="section-header">{selected_state} vs {compare_state}</div>', unsafe_allow_html=True)
        comp = engine.compare_states(selected_state, compare_state)
        if comp:
            comp_data = []
            nice_names = {
                'population_crore': 'Population (Cr)', 'health_budget_crore': 'Budget (₹ Cr)',
                'budget_per_capita_inr': 'Per Capita (₹)', 'hospitals_total': 'Hospitals',
                'hospital_beds_per_1000': 'Beds/1000', 'doctors_total': 'Doctors',
                'doctor_per_1000': 'Doctor Ratio', 'nurses_total': 'Nurses',
                'vaccine_coverage_pct': 'Vaccine Coverage(%)', 'infra_gap_score': 'Infra Gap',
                'maternal_mortality_ratio': 'MMR', 'infant_mortality_rate': 'IMR',
                'life_expectancy': 'Life Expectancy', 'icu_beds': 'ICU Beds',
                'cold_chain_facilities': 'Cold Chain'
            }
            for metric, vals in comp.items():
                comp_data.append({
                    'Metric': nice_names.get(metric, metric),
                    selected_state: vals[selected_state],
                    compare_state: vals[compare_state],
                    'Better': '✅ ' + vals['better']
                })
            st.dataframe(pd.DataFrame(comp_data), width='stretch', hide_index=True)


# =============================
# TAB 3: 2028-29 Predictions
# =============================
with tab3:
    st.markdown('<div class="section-header">🔮 Future Predictions — 2028 & 2029</div>', unsafe_allow_html=True)
    
    if predictions_df is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            pred_state = st.selectbox("Select State", states_list, key="pred_state")
        with col2:
            pred_year = st.selectbox("Prediction Year", [2028, 2029], key="pred_year")

        pred = predictions_df[
            (predictions_df['state'] == pred_state) & 
            (predictions_df['year'] == pred_year)
        ]
        
        current = latest_data[latest_data['state'] == pred_state]

        if not pred.empty and not current.empty:
            pred_row = pred.iloc[0]
            curr_row = current.iloc[0]

            st.markdown(f'<div class="section-header">{pred_state} — Current ({latest_year}) vs Predicted ({pred_year})</div>', unsafe_allow_html=True)
            
            # Prediction metric cards
            cols = st.columns(4)
            metrics_pred = [
                ("Health Budget", f"₹{curr_row['health_budget_crore']:,.0f} Cr", f"₹{pred_row['predicted_budget_crore']:,.0f} Cr"),
                ("Hospitals", f"{curr_row['hospitals_total']:,}", f"{pred_row['predicted_hospitals']:,.0f}"),
                ("Doctors", f"{curr_row['doctors_total']:,}", f"{pred_row['predicted_doctors']:,.0f}"),
                ("ICU Beds", f"{curr_row['icu_beds']:,}", f"{pred_row['predicted_icu_beds']:,.0f}"),
            ]
            for i, (label, current_val, predicted_val) in enumerate(metrics_pred):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div style="display:flex; justify-content:space-around; margin-top:0.5rem;">
                            <div>
                                <div style="color:#90E0EF; font-size:0.7rem;">CURRENT</div>
                                <div style="color:#ADE8F4; font-size:1.1rem; font-weight:600;">{current_val}</div>
                            </div>
                            <div style="color:#48CAE4; font-size:1.5rem; align-self:center;">→</div>
                            <div>
                                <div style="color:#00B894; font-size:0.7rem;">PREDICTED</div>
                                <div style="color:#00B894; font-size:1.1rem; font-weight:600;">{predicted_val}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            cols2 = st.columns(4)
            metrics_pred2 = [
                ("Vaccine Doses", f"{curr_row.get('vaccine_doses_required_cr', 'N/A')} Cr", f"{pred_row['predicted_vaccine_doses_cr']:.2f} Cr"),
                ("Beds/1000", f"{curr_row['hospital_beds_per_1000']:.2f}", f"{pred_row['predicted_beds_per_1000']:.2f}"),
                ("Nurses", f"{curr_row['nurses_total']:,}", f"{pred_row['predicted_nurses']:,.0f}"),
                ("Vaccine Coverage", f"{curr_row['vaccine_coverage_pct']:.1f}%", f"{pred_row['projected_vaccine_coverage_pct']:.1f}%"),
            ]
            for i, (label, current_val, predicted_val) in enumerate(metrics_pred2):
                with cols2[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div style="display:flex; justify-content:space-around; margin-top:0.5rem;">
                            <div>
                                <div style="color:#90E0EF; font-size:0.7rem;">CURRENT</div>
                                <div style="color:#ADE8F4; font-size:1.1rem; font-weight:600;">{current_val}</div>
                            </div>
                            <div style="color:#48CAE4; font-size:1.5rem; align-self:center;">→</div>
                            <div>
                                <div style="color:#00B894; font-size:0.7rem;">PREDICTED</div>
                                <div style="color:#00B894; font-size:1.1rem; font-weight:600;">{predicted_val}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Growth Trajectory Chart
            st.markdown(f'<div class="section-header">{pred_state} — Growth Trajectory</div>', unsafe_allow_html=True)
            
            state_hist = df[df['state'] == pred_state].sort_values('year')
            state_preds = predictions_df[predictions_df['state'] == pred_state].sort_values('year')

            col1, col2 = st.columns(2)
            with col1:
                # Budget trajectory
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=state_hist['year'], y=state_hist['health_budget_crore'],
                    name='Historical', mode='lines+markers',
                    line=dict(color='#00B4D8', width=3), marker=dict(size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=state_preds['year'], y=state_preds['predicted_budget_crore'],
                    name='Predicted', mode='lines+markers',
                    line=dict(color='#00B894', width=3, dash='dash'), marker=dict(size=10, symbol='diamond')
                ))
                fig.update_layout(
                    title="Budget Trajectory (₹ Crore)", height=350,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#CAF0F8'),
                    xaxis=dict(dtick=1, gridcolor='rgba(144,224,239,0.1)'),
                    yaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
                    legend=dict(x=0, y=1.1, orientation='h'),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                # Doctors trajectory
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=state_hist['year'], y=state_hist['doctors_total'],
                    name='Historical', mode='lines+markers',
                    line=dict(color='#00B4D8', width=3), marker=dict(size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=state_preds['year'], y=state_preds['predicted_doctors'],
                    name='Predicted', mode='lines+markers',
                    line=dict(color='#00B894', width=3, dash='dash'), marker=dict(size=10, symbol='diamond')
                ))
                fig.update_layout(
                    title="Doctor Deployment Trajectory", height=350,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#CAF0F8'),
                    xaxis=dict(dtick=1, gridcolor='rgba(144,224,239,0.1)'),
                    yaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
                    legend=dict(x=0, y=1.1, orientation='h'),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig, width='stretch')

        # All States Overview
        st.markdown('<div class="section-header">All States — Budget Predictions 2028</div>', unsafe_allow_html=True)
        pred_2028 = predictions_df[predictions_df['year'] == 2028].sort_values('predicted_budget_crore', ascending=False)
        fig = px.bar(
            pred_2028, x='state', y='predicted_budget_crore',
            color='predicted_budget_crore', color_continuous_scale='Tealgrn',
            labels={'predicted_budget_crore': 'Predicted Budget (₹ Cr)', 'state': ''}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#CAF0F8'),
            xaxis=dict(tickangle=45, gridcolor='rgba(144,224,239,0.1)'),
            yaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning("Predictions not available. Run `train_predictors.py` first.")


# =============================
# TAB 4: AI Recommendations
# =============================
with tab4:
    st.markdown('<div class="section-header">💡 AI-Powered Healthcare Recommendations</div>', unsafe_allow_html=True)
    
    # Priority Rankings
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown('<div class="section-header">State Priority Ranking</div>', unsafe_allow_html=True)
        rankings = engine.rank_all_states()
        for i, r in enumerate(rankings):
            badge = get_priority_badge(r['category'])
            st.markdown(
                f"**{i+1}.** {r['state']} — {r['score']:.1f} {badge}",
                unsafe_allow_html=True
            )

    with col1:
        rec_state = st.selectbox("Select State for Recommendations", states_list, key="rec_state")
        
        recs = engine.get_state_recommendations(rec_state)
        if recs:
            # Priority header
            badge = get_priority_badge(recs['priority_category'])
            st.markdown(
                f"### {rec_state} — Priority Score: **{recs['priority_score']:.1f}**/100 {badge}",
                unsafe_allow_html=True
            )
            
            # Component scores chart
            comp_scores = recs['component_scores']
            fig = go.Figure(data=[go.Bar(
                x=list(comp_scores.values()),
                y=[s.replace('_', ' ').title() for s in comp_scores.keys()],
                orientation='h',
                marker_color=['#ff7675' if v > 60 else '#fdcb6e' if v > 30 else '#00b894' 
                             for v in comp_scores.values()]
            )])
            fig.update_layout(
                title="Component Priority Scores", height=280,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#CAF0F8'),
                xaxis=dict(title='Score (0-100)', gridcolor='rgba(144,224,239,0.1)', range=[0, 100]),
                yaxis=dict(gridcolor='rgba(144,224,239,0.1)'),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Recommendations
            st.markdown(f'<div class="section-header">Action Items for {rec_state}</div>', unsafe_allow_html=True)
            
            total_cost = 0
            for rec in recs['recommendations']:
                priority = rec['priority']
                icon = "🔴" if priority == "HIGH" else "🟡"
                cost = rec.get('estimated_cost_crore', 0)
                total_cost += cost
                
                with st.expander(f"{rec['category']} — [{priority}] — Est. ₹{cost:,.0f} Cr"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Current:** {rec['current']}")
                        st.markdown(f"**Target:** {rec['target']}")
                    with col_b:
                        st.markdown(f"**Gap:** {rec['gap']}")
                        st.markdown(f"**Est. Investment:** ₹{cost:,.0f} Crore")
                    st.info(f"**Recommended Action:** {rec['action']}")
            
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 1rem;">
                <div class="metric-value">₹{total_cost:,.0f} Crore</div>
                <div class="metric-label">Total Estimated Investment Needed</div>
            </div>
            """, unsafe_allow_html=True)


# =============================
# TAB 5: Policy Chat (SLM + RAG)
# =============================
with tab5:
    st.markdown('<div class="section-header">💬 Healthcare Policy AI Chat</div>', unsafe_allow_html=True)
    st.markdown("*Ask questions about state healthcare, budgets, vaccines, supply chain & more. AI uses RAG + ML predictions.*")
    
    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar-like settings in expander
    with st.expander("⚙️ Query Settings"):
        chat_state = st.selectbox("Focus State (optional)", ["Auto-detect"] + states_list, key="chat_state")
        max_tokens = st.slider("Response Length", 200, 2048, 1024, step=50)

    user_input = st.chat_input("Ask about healthcare policy, budget, vaccines, supply chain...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing policy data & generating recommendations..."):
                # Detect state from query
                detected_state = None
                if chat_state != "Auto-detect":
                    detected_state = chat_state
                else:
                    for s in states_list:
                        if s.lower() in user_input.lower():
                            detected_state = s
                            break

                # RAG retrieval
                retrieved_context = ""
                db = load_vectorstore()
                if db:
                    docs = db.similarity_search(user_input, k=3)
                    # Limit context length to avoid truncating the prompt format
                    retrieved_context = "\n".join([doc.page_content for doc in docs])
                    if len(retrieved_context) > 1500:
                        retrieved_context = retrieved_context[:1500] + "..."

                # ML predictions
                pred_info = ""
                if detected_state:
                    pred_data = engine.get_predictions_for_state(detected_state)
                    if pred_data:
                        pred_info = (
                            f"\nML Predictions for {detected_state} (2028):\n"
                            f"Predicted Budget: ₹{pred_data.get('predicted_budget_crore', 'N/A'):,.0f} crore\n"
                            f"Predicted Hospitals: {pred_data.get('predicted_hospitals', 'N/A'):,.0f}\n"
                            f"Predicted Doctors: {pred_data.get('predicted_doctors', 'N/A'):,.0f}\n"
                            f"Predicted Vaccine Doses: {pred_data.get('predicted_vaccine_doses_cr', 'N/A')} crore\n"
                            f"Predicted ICU Beds: {pred_data.get('predicted_icu_beds', 'N/A'):,.0f}\n"
                        )

                    # Recommendation summary
                    rec_data = engine.get_state_recommendations(detected_state)
                    if rec_data:
                        pred_info += f"\nPriority Score: {rec_data['priority_score']}/100 ({rec_data['priority_category']})\n"
                        for rec in rec_data['recommendations'][:3]:
                            pred_info += f"- {rec['category']}: {rec['gap']}\n"

                # Build Ollama chat prompt
                system_msg = """You are a senior Government Healthcare Strategy Advisor for India with deep expertise in public health policy, resource allocation, and state-level healthcare systems.

Your responses must ALWAYS follow this EXACT structure with numbered headings:
1. Situation Analysis
2. Deployment Strategy
3. Infrastructure Requirements
4. Budget Allocation Plan
5. Risk Mitigation

Be specific, data-driven, and actionable. You MUST extract and prominently display the precise ML Predicted Budget numbers (e.g. ₹XXX crore) and Infrastructure gaps provided below. Use bullet points within each section."""

                user_msg = f"""Using the following official healthcare data, answer the policy question below.

Official Healthcare Data:
{retrieved_context}
{pred_info}

Policy Question: {user_input}"""

                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
                
                # Fetch model
                _, llm_model = load_llm()

            # --- OUTSIDE SPINNER FOR STREAMING ---
            if llm_model:
                try:
                    # Display context info before stream
                    if detected_state:
                        st.info(f"📍 State: **{detected_state}** | 📊 Data Points: {len(retrieved_context.split())} words retrieved")

                    answer = st.write_stream(
                        stream_ollama_chat(llm_model, messages, max_tokens)
                    )
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
            else:
                # Fallback: use recommendation engine
                if detected_state:
                    answer = f"**{detected_state} — Healthcare Recommendations**\n\n"
                    recs = engine.get_state_recommendations(detected_state)
                    if recs:
                        answer += f"**Priority Score:** {recs['priority_score']}/100 ({recs['priority_category']})\n\n"
                        for rec in recs['recommendations']:
                            answer += f"### {rec['category']}\n"
                            answer += f"- **Current:** {rec['current']}\n"
                            answer += f"- **Target:** {rec['target']}\n"
                            answer += f"- **Gap:** {rec['gap']}\n"
                            answer += f"- **Action:** {rec['action']}\n"
                            answer += f"- **Est. Cost:** ₹{rec.get('estimated_cost_crore', 0):,.0f} Crore\n\n"
                        answer += f"\n**Total Investment Needed:** ₹{recs['total_investment_needed_crore']:,.0f} Crore"
                else:
                    answer = (
                        "I can provide recommendations for specific Indian states. "
                        "Please mention a state name (e.g., 'Bihar', 'Maharashtra', 'Uttar Pradesh') "
                        "in your question for targeted analysis.\n\n"
                        "**Available analysis types:**\n"
                        "- Budget allocation and predictions\n"
                        "- Hospital and infrastructure needs\n"
                        "- Doctor deployment strategy\n"
                        "- Vaccine distribution planning\n"
                        "- Supply chain optimization\n"
                        "- 2028-29 resource predictions"
                    )
                
                # Display fallback response
                if detected_state:
                    st.info(f"📍 State: **{detected_state}** | 📊 Data Points: {len(retrieved_context.split())} words retrieved")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
