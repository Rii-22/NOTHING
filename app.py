"""
NOTHING Device Analytics
Pure. Transparent. Data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import re
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="nothing analytics",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# NOTHING DOT MATRIX AESTHETIC
# ============================================================================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    
    /* Grid background like NOTHING's website */
    .stApp {
        background-color: #f5f5f5;
        background-image: 
            linear-gradient(#e0e0e0 1px, transparent 1px),
            linear-gradient(90deg, #e0e0e0 1px, transparent 1px);
        background-size: 20px 20px;
        color: #000000;
    }
    
    /* Sidebar - clean white */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Dot matrix font style */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Mono', monospace !important;
        color: #000000 !important;
        font-weight: 400 !important;
        letter-spacing: 0.15em !important;
        text-transform: lowercase !important;
    }
    
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0.3rem !important;
    }
    
    h2 {
        font-size: 1.2rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1rem !important;
    }
    
    /* Body text - dot matrix style */
    p, label, .stMarkdown {
        font-family: 'Space Mono', monospace !important;
        color: #333333 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        font-weight: 400 !important;
    }
    
    /* Metrics - minimalist numbers */
    [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        color: #000000 !important;
        font-size: 2.5rem !important;
        font-weight: 400 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Space Mono', monospace !important;
        color: #666666 !important;
        font-size: 0.7rem !important;
        text-transform: lowercase !important;
        letter-spacing: 0.1em !important;
        font-weight: 400 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
    }
    
    /* Buttons - minimal outline */
    .stButton>button {
        font-family: 'Space Mono', monospace !important;
        background-color: transparent !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        padding: 8px 24px !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em !important;
        text-transform: lowercase !important;
        transition: all 0.2s !important;
    }
    
    .stButton>button:hover {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Tabs - minimal underline */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: transparent;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace !important;
        background-color: transparent;
        color: #999999;
        border: none;
        padding: 12px 20px;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: lowercase;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: #000000;
        border-bottom: 2px solid #000000;
    }
    
    /* Dataframes */
    .dataframe {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Space Mono', monospace !important;
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #e0e0e0;
        border-radius: 0px;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    
    /* Info boxes */
    .stAlert {
        font-family: 'Space Mono', monospace !important;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0px;
        font-size: 0.75rem;
    }
    
    /* Slider */
    .stSlider {
        font-family: 'Space Mono', monospace !important;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 2rem 0;
    }
    
    /* Remove padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Dots pattern overlay (subtle) */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: radial-gradient(circle, #d0d0d0 1px, transparent 1px);
        background-size: 40px 40px;
        opacity: 0.3;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Ensure content is above pattern */
    [data-testid="stAppViewContainer"] > .main {
        position: relative;
        z-index: 1;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION ENGINE
# ============================================================================

@st.cache_data
def generate_nothing_telemetry(n_sessions=6000, seed=42):
    """
    generate synthetic nothing device telemetry data.
    """
    np.random.seed(seed)
    
    # device models
    devices = np.random.choice(
        ['phone (1)', 'phone (2)', 'ear (1)', 'ear (stick)'],
        n_sessions,
        p=[0.35, 0.40, 0.15, 0.10]
    )
    
    # serial numbers
    def generate_serial(device):
        if 'phone' in device:
            prefix = 'np'
        else:
            prefix = 'ne'
        number = f"{device[-2]}{np.random.randint(100000, 999999)}"
        return f"{prefix}{number}"
    
    serial_numbers = [generate_serial(device) for device in devices]
    
    # build versions
    build_versions = np.random.choice(
        ['2.5.6', '2.5.7', '2.6.0', '2.6.1', '3.0.0'],
        n_sessions,
        p=[0.10, 0.15, 0.30, 0.25, 0.20]
    )
    
    # glyph interface data
    glyph_activations = []
    glyph_brightness_avg = []
    
    for i, device in enumerate(devices):
        if 'phone' in device:
            activations = int(np.random.gamma(shape=3, scale=15))
            glyph_activations.append(activations)
            brightness = np.random.uniform(40, 95)
            glyph_brightness_avg.append(round(brightness, 1))
        else:
            glyph_activations.append(0)
            glyph_brightness_avg.append(0)
    
    # performance metrics
    screen_on_time = []
    for device in devices:
        if 'phone' in device:
            sot = np.random.gamma(shape=4, scale=1.5)
            screen_on_time.append(round(np.clip(sot, 2, 12), 1))
        else:
            screen_on_time.append(0)
    
    # battery data
    battery_cycles = []
    for device in devices:
        if device == 'phone (2)':
            cycles = int(np.random.gamma(shape=2, scale=50))
        elif device == 'phone (1)':
            cycles = int(np.random.gamma(shape=3, scale=80))
        elif device == 'ear (1)':
            cycles = int(np.random.gamma(shape=2, scale=120))
        else:
            cycles = int(np.random.gamma(shape=2, scale=100))
        battery_cycles.append(np.clip(cycles, 5, 800))
    
    battery_health = 100 - (np.array(battery_cycles) / 10)
    battery_health = np.clip(battery_health, 75, 100).round(1)
    
    # transparency score
    transparency_scores = []
    for i, device in enumerate(devices):
        score = 50
        if 'phone' in device:
            score += min(glyph_activations[i] / 2, 30)
        if build_versions[i] in ['2.6.1', '3.0.0']:
            score += 10
        if battery_cycles[i] > 400:
            score -= 10
        transparency_scores.append(round(np.clip(score, 40, 100), 1))
    
    # thermal data
    avg_temp_celsius = []
    for i, device in enumerate(devices):
        if 'phone' in device:
            base_temp = np.random.normal(35, 4)
            usage_impact = screen_on_time[i] * 1.2
            temp = base_temp + usage_impact
            avg_temp_celsius.append(round(np.clip(temp, 28, 48), 1))
        else:
            avg_temp_celsius.append(0)
    
    # software metrics
    app_crashes_per_week = np.random.poisson(lam=0.8, size=n_sessions)
    
    # regions
    regions = np.random.choice(
        ['europe', 'asia', 'north america', 'other'],
        n_sessions,
        p=[0.40, 0.35, 0.15, 0.10]
    )
    
    # purchase dates
    days_since_purchase = np.random.gamma(shape=2, scale=180, size=n_sessions)
    days_since_purchase = np.clip(days_since_purchase, 1, 800).astype(int)
    
    # create dataframe
    df = pd.DataFrame({
        'session_id': [f"ns{str(i).zfill(7)}" for i in range(1, n_sessions + 1)],
        'device_model': devices,
        'serial_number': serial_numbers,
        'nothing_os_build': build_versions,
        'days_owned': days_since_purchase,
        'glyph_activations_daily': glyph_activations,
        'glyph_brightness_avg': glyph_brightness_avg,
        'screen_on_time_hours': screen_on_time,
        'battery_cycles': battery_cycles,
        'battery_health_pct': battery_health,
        'transparency_score': transparency_scores,
        'avg_temp_celsius': avg_temp_celsius,
        'app_crashes_weekly': app_crashes_per_week,
        'region': regions
    })
    
    return df

# ============================================================================
# ANALYTICAL FUNCTIONS
# ============================================================================

@st.cache_data
def glyph_engagement_analysis(df):
    """analyze glyph interface adoption"""
    phone_users = df[df['device_model'].str.contains('phone')].copy()
    
    if len(phone_users) == 0:
        return None
    
    correlation, p_value = stats.pearsonr(
        phone_users['glyph_activations_daily'],
        phone_users['transparency_score']
    )
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

@st.cache_data
def parse_serial_manufacturing(df):
    """extract manufacturing batch from serial numbers"""
    def extract_batch(serial):
        match = re.search(r'n[pe](\d)(\d{5})', serial)
        if match:
            batch = int(match.group(2)[:3])
            return f"batch_{batch // 100}"
        return 'unknown'
    
    df['manufacturing_batch'] = df['serial_number'].apply(extract_batch)
    
    batch_stats = df.groupby('manufacturing_batch').agg({
        'battery_health_pct': 'mean',
        'transparency_score': 'mean',
        'session_id': 'count'
    }).reset_index()
    
    batch_stats.columns = ['batch', 'avg_battery_health', 'avg_transparency', 'device_count']
    batch_stats = batch_stats.sort_values('avg_transparency', ascending=False)
    
    return df, batch_stats

@st.cache_data
def detect_thermal_anomalies(df):
    """identify devices with abnormal thermal behavior"""
    phones = df[df['device_model'].str.contains('phone')].copy()
    
    if len(phones) == 0:
        return phones, pd.DataFrame()
    
    z_scores = np.abs(stats.zscore(phones['avg_temp_celsius']))
    phones['temp_z_score'] = z_scores
    
    thermal_anomalies = phones[z_scores > 2.5].copy()
    
    return phones, thermal_anomalies

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    
    # === HEADER ===
    st.markdown("""
        <div style='text-align: center; margin: 2rem 0 3rem 0;'>
            <h1>nothing analytics</h1>
            <p style='font-size: 0.7rem; color: #999999; letter-spacing: 0.2em; margin-top: 0.5rem;'>
                device telemetry portal
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # === GENERATE DATA ===
    df = generate_nothing_telemetry(n_sessions=6000)
    df, batch_stats = parse_serial_manufacturing(df)
    phones, thermal_anomalies = detect_thermal_anomalies(df)
    glyph_analysis = glyph_engagement_analysis(df)
    
    # === SIDEBAR ===
    st.sidebar.markdown("### filters")
    st.sidebar.markdown("---")
    
    selected_devices = st.sidebar.multiselect(
        "device model",
        options=sorted(df['device_model'].unique()),
        default=sorted(df['device_model'].unique())
    )
    
    selected_regions = st.sidebar.multiselect(
        "region",
        options=sorted(df['region'].unique()),
        default=sorted(df['region'].unique())
    )
    
    min_transparency = st.sidebar.slider(
        "min transparency score",
        min_value=40,
        max_value=100,
        value=50,
        step=5
    )
    
    # apply filters
    filtered_df = df[
        (df['device_model'].isin(selected_devices)) &
        (df['region'].isin(selected_regions)) &
        (df['transparency_score'] >= min_transparency)
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**sessions:** {len(filtered_df):,}")
    
    # === METRICS ===
    st.markdown("### overview")
    st.markdown("")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_transparency = filtered_df['transparency_score'].mean()
        st.metric(
            "transparency",
            f"{avg_transparency:.1f}",
            delta=f"{avg_transparency - 75:.1f}"
        )
    
    with col2:
        avg_battery = filtered_df['battery_health_pct'].mean()
        st.metric(
            "battery health",
            f"{avg_battery:.1f}%",
            delta=f"{avg_battery - 95:.1f}%"
        )
    
    with col3:
        avg_crashes = filtered_df['app_crashes_weekly'].mean()
        st.metric(
            "crashes/week",
            f"{avg_crashes:.2f}",
            delta=f"{0.5 - avg_crashes:.2f}",
            delta_color="inverse"
        )
    
    with col4:
        phones_filtered = filtered_df[filtered_df['device_model'].str.contains('phone')]
        if len(phones_filtered) > 0:
            avg_glyph = phones_filtered['glyph_activations_daily'].mean()
            st.metric(
                "glyph/day",
                f"{avg_glyph:.0f}",
                delta=f"{avg_glyph - 25:.0f}"
            )
        else:
            st.metric("glyph/day", "—")
    
    st.markdown("---")
    
    # === GLYPH ANALYSIS ===
    if glyph_analysis:
        st.markdown("### glyph interface")
        st.markdown("")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            **correlation analysis**
            
            statistical test of glyph engagement vs user satisfaction.
            
            - correlation: {glyph_analysis['correlation']:.4f}
            - p-value: {glyph_analysis['p_value']:.6f}
            - significance: {"confirmed" if glyph_analysis['significant'] else "insufficient data"}
            
            {"users who engage with glyph interface report higher satisfaction scores. the signature feature drives meaningful differentiation." if glyph_analysis['correlation'] > 0.3 and glyph_analysis['significant'] else "moderate correlation detected. glyph shows positive impact." if glyph_analysis['significant'] else "no significant correlation detected."}
            """)
        
        with col2:
            if glyph_analysis['significant']:
                st.success("✓ validated")
            else:
                st.info("— pending")
    
    st.markdown("---")
    
    # === TABS ===
    tab1, tab2, tab3 = st.tabs([
        "battery & thermal",
        "software",
        "manufacturing"
    ])
    
    with tab1:
        st.markdown("#### battery degradation")
        st.markdown("")
        
        ownership_bins = pd.cut(
            filtered_df['days_owned'],
            bins=[0, 90, 180, 365, 545, 730, 1000],
            labels=['<3mo', '3-6mo', '6-12mo', '12-18mo', '18-24mo', '>24mo']
        )
        
        battery_degradation = filtered_df.groupby(ownership_bins)['battery_health_pct'].mean()
        st.line_chart(battery_degradation)
        
        st.markdown("")
        
        if len(thermal_anomalies) > 0:
            st.markdown("#### thermal anomalies")
            st.warning(f"{len(thermal_anomalies)} devices with elevated temps")
            st.dataframe(
                thermal_anomalies[['serial_number', 'device_model', 'avg_temp_celsius']].head(10),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("no thermal anomalies detected")
    
    with tab2:
        st.markdown("#### crash frequency distribution")
        st.markdown("")
        
        crash_dist = filtered_df['app_crashes_weekly'].value_counts().sort_index()
        st.bar_chart(crash_dist)
        
        st.markdown("")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("median crashes", f"{filtered_df['app_crashes_weekly'].median():.1f}")
        with col2:
            zero_crash = len(filtered_df[filtered_df['app_crashes_weekly'] == 0])
            st.metric("zero crash sessions", f"{zero_crash:,}")
    
    with tab3:
        st.markdown("#### batch quality")
        st.markdown("")
        
        st.dataframe(
            batch_stats.head(10),
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("")
        
        best_batch = batch_stats.iloc[0]
        st.info(f"""
        top batch: {best_batch['batch']}  
        transparency: {best_batch['avg_transparency']:.1f}  
        devices: {best_batch['device_count']:,}
        """)
    
    st.markdown("---")
    
    # === FOOTER ===
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; padding: 2rem 0;'>
            <p style='font-size: 0.65rem; color: #999999; letter-spacing: 0.15em;'>
                nothing device analytics portal
            </p>
            <p style='font-size: 0.6rem; color: #cccccc; letter-spacing: 0.1em; margin-top: 0.5rem;'>
                transparent technology
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
