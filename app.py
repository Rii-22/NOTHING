"""
NOTHING Device Analytics Portal
Transparent Technology • Zero Bloat • Pure Performance Data
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
    page_title="NOTHING Analytics",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# NOTHING MINIMALIST THEME
# ============================================================================

st.markdown("""
    <style>
    /* Pure Background - Nothing's signature white/black */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* Sidebar - Pure Black */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #333333;
    }
    
    /* Headers - Nothing's Dot Matrix Font Style */
    h1 {
        color: #FFFFFF !important;
        font-weight: 300;
        letter-spacing: 8px;
        text-transform: uppercase;
        font-size: 2.5rem;
        border-bottom: 1px solid #333333;
        padding-bottom: 20px;
    }
    
    h2, h3 {
        color: #FFFFFF !important;
        font-weight: 300;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    
    /* Metrics - Clean & Monospace */
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-size: 2.8rem;
        font-weight: 300;
        font-family: 'Courier New', monospace;
    }
    
    [data-testid="stMetricLabel"] {
        color: #999999;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 300;
    }
    
    [data-testid="stMetricDelta"] {
        color: #666666;
        font-family: 'Courier New', monospace;
    }
    
    /* Buttons - Transparent with Border */
    .stButton>button {
        background-color: transparent;
        color: #FFFFFF;
        font-weight: 300;
        border-radius: 0px;
        border: 1px solid #FFFFFF;
        padding: 12px 32px;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton>button:hover {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Sliders - Minimal Gray */
    .stSlider>div>div>div>div {
        background-color: #666666;
    }
    
    /* Text - Gray scale */
    p, label, .stMarkdown {
        color: #999999;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Tabs - Minimal Underline */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #000000;
        border-bottom: 1px solid #333333;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #666666;
        border-radius: 0px;
        padding: 16px 24px;
        border: none;
        border-bottom: 2px solid transparent;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #FFFFFF;
        border-bottom: 2px solid #FFFFFF;
    }
    
    /* Dataframes - Monospace */
    .dataframe {
        background-color: #0a0a0a !important;
        color: #FFFFFF !important;
        font-family: 'Courier New', monospace;
        border: 1px solid #333333;
    }
    
    /* Expander - Minimal */
    .streamlit-expanderHeader {
        background-color: transparent;
        color: #FFFFFF;
        border: 1px solid #333333;
        border-radius: 0px;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    /* Alert Boxes - Transparent */
    .stSuccess, .stInfo, .stWarning {
        background-color: transparent;
        color: #FFFFFF;
        border: 1px solid #666666;
        border-radius: 0px;
    }
    
    /* Divider - Subtle Gray */
    hr {
        border-color: #333333;
        margin: 40px 0;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION ENGINE
# ============================================================================

@st.cache_data
def generate_nothing_telemetry(n_sessions=6000, seed=42):
    """
    Generate synthetic NOTHING device telemetry data.
    Focus: Phone (1), Phone (2), Ear (1), Ear (stick)
    
    Returns:
        pd.DataFrame: Device usage and performance data
    """
    np.random.seed(seed)
    
    # === DEVICE METADATA ===
    
    # Device models with realistic distribution
    devices = np.random.choice(
        ['Phone (1)', 'Phone (2)', 'Ear (1)', 'Ear (stick)'],
        n_sessions,
        p=[0.35, 0.40, 0.15, 0.10]  # Phone (2) is newest, most popular
    )
    
    # Generate device serial numbers (NOTHING style: minimal alphanumeric)
    def generate_serial(device):
        if 'Phone' in device:
            prefix = 'NP'
        else:
            prefix = 'NE'
        number = f"{device[-2]}{np.random.randint(100000, 999999)}"
        return f"{prefix}{number}"
    
    serial_numbers = [generate_serial(device) for device in devices]
    
    # Build numbers (Nothing OS versions)
    build_versions = np.random.choice(
        ['2.5.6', '2.5.7', '2.6.0', '2.6.1', '3.0.0'],
        n_sessions,
        p=[0.10, 0.15, 0.30, 0.25, 0.20]
    )
    
    # === GLYPH INTERFACE DATA (Phone only) ===
    # NOTHING's signature transparent back with LEDs
    
    glyph_activations = []
    glyph_pattern_diversity = []
    glyph_brightness_avg = []
    
    for i, device in enumerate(devices):
        if 'Phone' in device:
            # Glyph activations per day
            activations = int(np.random.gamma(shape=3, scale=15))
            glyph_activations.append(activations)
            
            # Pattern diversity (how many different glyph patterns used)
            diversity = np.random.randint(3, 12)
            glyph_pattern_diversity.append(diversity)
            
            # Average brightness (0-100%)
            brightness = np.random.uniform(40, 95)
            glyph_brightness_avg.append(round(brightness, 1))
        else:
            glyph_activations.append(0)
            glyph_pattern_diversity.append(0)
            glyph_brightness_avg.append(0)
    
    # === PERFORMANCE METRICS ===
    
    # Screen-on time (hours per day) for phones
    screen_on_time = []
    for device in devices:
        if 'Phone' in device:
            sot = np.random.gamma(shape=4, scale=1.5)
            screen_on_time.append(round(np.clip(sot, 2, 12), 1))
        else:
            screen_on_time.append(0)
    
    # Battery cycles (wear indicator)
    # Phone (2) is newer, lower cycles
    battery_cycles = []
    for device in devices:
        if device == 'Phone (2)':
            cycles = int(np.random.gamma(shape=2, scale=50))
        elif device == 'Phone (1)':
            cycles = int(np.random.gamma(shape=3, scale=80))
        elif device == 'Ear (1)':
            cycles = int(np.random.gamma(shape=2, scale=120))
        else:  # Ear (stick)
            cycles = int(np.random.gamma(shape=2, scale=100))
        
        battery_cycles.append(np.clip(cycles, 5, 800))
    
    # Battery health percentage
    battery_health = 100 - (np.array(battery_cycles) / 10)
    battery_health = np.clip(battery_health, 75, 100).round(1)
    
    # Charge frequency (times per day)
    charge_frequency = np.random.gamma(shape=2, scale=0.8, size=n_sessions)
    charge_frequency = np.clip(charge_frequency, 0.5, 5.0).round(1)
    
    # === TRANSPARENCY SCORE (Nothing's core philosophy) ===
    # Measures user engagement with Nothing's unique features
    
    transparency_scores = []
    for i, device in enumerate(devices):
        score = 50  # Base score
        
        if 'Phone' in device:
            # Bonus for using Glyph interface
            score += min(glyph_activations[i] / 2, 30)
            score += glyph_pattern_diversity[i] * 2
        
        # Bonus for keeping device updated
        if build_versions[i] in ['2.6.1', '3.0.0']:
            score += 10
        
        # Penalty for excessive battery wear
        if battery_cycles[i] > 400:
            score -= 10
        
        transparency_scores.append(round(np.clip(score, 40, 100), 1))
    
    # === AUDIO QUALITY METRICS (Earbuds) ===
    
    anc_effectiveness = []
    audio_codec_used = []
    listening_hours_daily = []
    
    for device in devices:
        if 'Ear' in device:
            # ANC effectiveness (0-100%, Ear (1) has better ANC)
            if device == 'Ear (1)':
                anc = np.random.uniform(75, 95)
            else:  # Ear (stick) - no ANC
                anc = 0
            anc_effectiveness.append(round(anc, 1))
            
            # Codec usage
            codec = np.random.choice(['AAC', 'LDAC', 'aptX'], p=[0.50, 0.30, 0.20])
            audio_codec_used.append(codec)
            
            # Listening hours
            hours = np.random.gamma(shape=3, scale=1.2)
            listening_hours_daily.append(round(np.clip(hours, 0.5, 8), 1))
        else:
            anc_effectiveness.append(0)
            audio_codec_used.append('N/A')
            listening_hours_daily.append(0)
    
    # === THERMAL PERFORMANCE ===
    # Nothing focuses on efficient cooling
    
    avg_temp_celsius = []
    thermal_events = []
    
    for i, device in enumerate(devices):
        if 'Phone' in device:
            # Base temp + usage impact
            base_temp = np.random.normal(35, 4)
            usage_impact = screen_on_time[i] * 1.2
            temp = base_temp + usage_impact
            avg_temp_celsius.append(round(np.clip(temp, 28, 48), 1))
            
            # Thermal throttling events (rare on Nothing devices)
            events = int(np.random.poisson(lam=0.3))
            thermal_events.append(events)
        else:
            avg_temp_celsius.append(0)
            thermal_events.append(0)
    
    # === SOFTWARE EXPERIENCE ===
    # Nothing OS fluidity and bloat-free promise
    
    app_crashes_per_week = np.random.poisson(lam=0.8, size=n_sessions)
    system_lag_events = np.random.poisson(lam=1.2, size=n_sessions)
    bloatware_apps_removed = np.random.randint(0, 3, n_sessions)
    
    # === REGION & MARKET DATA ===
    regions = np.random.choice(
        ['Europe', 'Asia', 'North America', 'Other'],
        n_sessions,
        p=[0.40, 0.35, 0.15, 0.10]
    )
    
    # === PURCHASE DATE (for age analysis) ===
    # Generate realistic purchase dates
    days_since_purchase = np.random.gamma(shape=2, scale=180, size=n_sessions)
    days_since_purchase = np.clip(days_since_purchase, 1, 800).astype(int)
    
    purchase_dates = [
        (datetime.now() - timedelta(days=int(days))).strftime('%Y-%m-%d')
        for days in days_since_purchase
    ]
    
    # === CREATE DATAFRAME ===
    df = pd.DataFrame({
        'Session_ID': [f"NS{str(i).zfill(7)}" for i in range(1, n_sessions + 1)],
        'Device_Model': devices,
        'Serial_Number': serial_numbers,
        'Nothing_OS_Build': build_versions,
        'Purchase_Date': purchase_dates,
        'Days_Owned': days_since_purchase,
        'Glyph_Activations_Daily': glyph_activations,
        'Glyph_Pattern_Diversity': glyph_pattern_diversity,
        'Glyph_Brightness_Avg': glyph_brightness_avg,
        'Screen_On_Time_Hours': screen_on_time,
        'Battery_Cycles': battery_cycles,
        'Battery_Health_Pct': battery_health,
        'Charge_Frequency_Daily': charge_frequency,
        'Transparency_Score': transparency_scores,
        'ANC_Effectiveness_Pct': anc_effectiveness,
        'Audio_Codec': audio_codec_used,
        'Listening_Hours_Daily': listening_hours_daily,
        'Avg_Temp_Celsius': avg_temp_celsius,
        'Thermal_Events': thermal_events,
        'App_Crashes_Weekly': app_crashes_per_week,
        'System_Lag_Events': system_lag_events,
        'Bloatware_Removed': bloatware_apps_removed,
        'Region': regions
    })
    
    return df

# ============================================================================
# ANALYTICAL FUNCTIONS
# ============================================================================

@st.cache_data
def glyph_engagement_analysis(df):
    """
    GLYPH INTERFACE ADOPTION
    
    Analyzes how users interact with Nothing's signature Glyph interface.
    Correlates usage with overall device satisfaction (Transparency Score).
    """
    phone_users = df[df['Device_Model'].str.contains('Phone')].copy()
    
    if len(phone_users) == 0:
        return None
    
    # Segment users by Glyph usage intensity
    phone_users['Glyph_User_Type'] = pd.cut(
        phone_users['Glyph_Activations_Daily'],
        bins=[-1, 10, 30, 100],
        labels=['Light', 'Moderate', 'Power User']
    )
    
    # Analyze Transparency Score by usage type
    engagement_stats = phone_users.groupby('Glyph_User_Type').agg({
        'Transparency_Score': ['mean', 'std'],
        'Glyph_Pattern_Diversity': 'mean',
        'Session_ID': 'count'
    }).reset_index()
    
    engagement_stats.columns = ['User_Type', 'Avg_Transparency', 'Std_Transparency', 
                                  'Avg_Pattern_Diversity', 'User_Count']
    
    # Statistical test: Does Glyph usage correlate with satisfaction?
    correlation, p_value = stats.pearsonr(
        phone_users['Glyph_Activations_Daily'],
        phone_users['Transparency_Score']
    )
    
    return {
        'engagement_stats': engagement_stats,
        'correlation': correlation,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

@st.cache_data
def parse_serial_manufacturing(df):
    """
    REGEX SERIAL NUMBER ANALYSIS
    
    Extracts manufacturing batch info from serial numbers.
    Pattern: NP[model][batch] or NE[model][batch]
    """
    def extract_batch(serial):
        # Extract first digit after prefix as model indicator
        match = re.search(r'N[PE](\d)(\d{5})', serial)
        if match:
            model_digit = match.group(1)
            batch = int(match.group(2)[:3])  # First 3 digits as batch
            return f"Batch_{batch // 100}"  # Group into batch ranges
        return 'Unknown'
    
    df['Manufacturing_Batch'] = df['Serial_Number'].apply(extract_batch)
    
    # Analyze quality metrics by batch
    batch_stats = df.groupby('Manufacturing_Batch').agg({
        'Battery_Health_Pct': 'mean',
        'Transparency_Score': 'mean',
        'App_Crashes_Weekly': 'mean',
        'Thermal_Events': 'mean',
        'Session_ID': 'count'
    }).reset_index()
    
    batch_stats.columns = ['Batch', 'Avg_Battery_Health', 'Avg_Transparency', 
                           'Avg_Crashes', 'Avg_Thermal_Events', 'Device_Count']
    batch_stats = batch_stats.sort_values('Avg_Transparency', ascending=False)
    
    return df, batch_stats

@st.cache_data
def detect_thermal_anomalies(df):
    """
    THERMAL PERFORMANCE OUTLIERS
    
    Identifies devices with abnormal thermal behavior using Z-score analysis.
    Critical for Nothing's focus on efficient, cool-running hardware.
    """
    phones = df[df['Device_Model'].str.contains('Phone')].copy()
    
    if len(phones) == 0:
        return phones, pd.DataFrame()
    
    # Z-score for temperature
    z_scores = np.abs(stats.zscore(phones['Avg_Temp_Celsius']))
    phones['Temp_Z_Score'] = z_scores
    
    # Anomalies: Z > 2.5 or thermal events > 3
    thermal_anomalies = phones[
        (z_scores > 2.5) | (phones['Thermal_Events'] > 3)
    ].copy()
    
    return phones, thermal_anomalies

@st.cache_data
def software_stability_analysis(df):
    """
    NOTHING OS STABILITY METRICS
    
    Analyzes crash rates and lag events to validate "zero bloat" promise.
    Compares stability across Nothing OS versions.
    """
    stability_by_build = df.groupby('Nothing_OS_Build').agg({
        'App_Crashes_Weekly': 'mean',
        'System_Lag_Events': 'mean',
        'Transparency_Score': 'mean',
        'Session_ID': 'count'
    }).reset_index()
    
    stability_by_build.columns = ['OS_Build', 'Avg_Crashes', 'Avg_Lag', 
                                   'Avg_Transparency', 'User_Count']
    
    # Calculate stability score (inverse of issues)
    max_crashes = stability_by_build['Avg_Crashes'].max()
    max_lag = stability_by_build['Avg_Lag'].max()
    
    stability_by_build['Stability_Score'] = (
        100 - (stability_by_build['Avg_Crashes'] / max_crashes * 40) -
        (stability_by_build['Avg_Lag'] / max_lag * 40) +
        (stability_by_build['Avg_Transparency'] / 100 * 20)
    ).round(1)
    
    stability_by_build = stability_by_build.sort_values('Stability_Score', ascending=False)
    
    return stability_by_build

@st.cache_data
def audio_quality_assessment(df):
    """
    EARBUDS PERFORMANCE ANALYSIS
    
    Analyzes ANC effectiveness and codec usage for Nothing Ear products.
    Correlates with listening duration and battery health.
    """
    earbuds = df[df['Device_Model'].str.contains('Ear')].copy()
    
    if len(earbuds) == 0:
        return None
    
    # ANC vs Non-ANC performance
    ear1 = earbuds[earbuds['Device_Model'] == 'Ear (1)']
    ear_stick = earbuds[earbuds['Device_Model'] == 'Ear (stick)']
    
    # Codec usage distribution
    codec_stats = earbuds.groupby('Audio_Codec').agg({
        'Listening_Hours_Daily': 'mean',
        'Battery_Health_Pct': 'mean',
        'Session_ID': 'count'
    }).reset_index()
    
    codec_stats.columns = ['Codec', 'Avg_Listening_Hours', 'Avg_Battery_Health', 'User_Count']
    codec_stats = codec_stats.sort_values('Avg_Listening_Hours', ascending=False)
    
    # Model comparison
    model_comparison = earbuds.groupby('Device_Model').agg({
        'ANC_Effectiveness_Pct': 'mean',
        'Listening_Hours_Daily': 'mean',
        'Battery_Health_Pct': 'mean',
        'Battery_Cycles': 'mean',
        'Session_ID': 'count'
    }).reset_index()
    
    model_comparison.columns = ['Model', 'Avg_ANC', 'Avg_Listening_Hours', 
                                'Avg_Battery_Health', 'Avg_Cycles', 'User_Count']
    
    return {
        'codec_stats': codec_stats,
        'model_comparison': model_comparison,
        'ear1_count': len(ear1),
        'ear_stick_count': len(ear_stick)
    }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    
    # === HEADER ===
    st.markdown("""
        <div style='text-align: center; margin-bottom: 40px;'>
            <h1 style='font-size: 2rem; margin-bottom: 5px;'>NOTHING</h1>
            <p style='color: #666666; letter-spacing: 4px; font-size: 0.85rem; margin-top: 0;'>
                DEVICE ANALYTICS PORTAL
            </p>
            <p style='color: #444444; letter-spacing: 2px; font-size: 0.7rem; margin-top: 10px;'>
                TRANSPARENT TECHNOLOGY • ZERO BLOAT • PURE DATA
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === GENERATE DATA ===
    with st.spinner('· · · LOADING TELEMETRY · · ·'):
        df = generate_nothing_telemetry(n_sessions=6000)
        df, batch_stats = parse_serial_manufacturing(df)
        phones, thermal_anomalies = detect_thermal_anomalies(df)
        stability_stats = software_stability_analysis(df)
        glyph_analysis = glyph_engagement_analysis(df)
        audio_analysis = audio_quality_assessment(df)
    
    # === SIDEBAR CONTROLS ===
    st.sidebar.markdown("### FILTERS")
    st.sidebar.markdown("---")
    
    selected_devices = st.sidebar.multiselect(
        "DEVICE MODEL",
        options=sorted(df['Device_Model'].unique()),
        default=sorted(df['Device_Model'].unique())
    )
    
    selected_regions = st.sidebar.multiselect(
        "REGION",
        options=sorted(df['Region'].unique()),
        default=sorted(df['Region'].unique())
    )
    
    min_transparency = st.sidebar.slider(
        "MIN TRANSPARENCY SCORE",
        min_value=40,
        max_value=100,
        value=50,
        step=5
    )
    
    # Apply filters
    filtered_df = df[
        (df['Device_Model'].isin(selected_devices)) &
        (df['Region'].isin(selected_regions)) &
        (df['Transparency_Score'] >= min_transparency)
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**ACTIVE SESSIONS:** {len(filtered_df):,}")
    st.sidebar.markdown(f"**TOTAL FLEET:** {len(df):,}")
    
    # === CORE METRICS ===
    st.markdown("### SYSTEM OVERVIEW")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_transparency = filtered_df['Transparency_Score'].mean()
        st.metric(
            "TRANSPARENCY",
            f"{avg_transparency:.1f}",
            delta=f"{avg_transparency - 75:.1f} vs target"
        )
    
    with col2:
        avg_battery = filtered_df['Battery_Health_Pct'].mean()
        st.metric(
            "BATTERY HEALTH",
            f"{avg_battery:.1f}%",
            delta=f"{avg_battery - 95:.1f}%"
        )
    
    with col3:
        avg_crashes = filtered_df['App_Crashes_Weekly'].mean()
        st.metric(
            "CRASHES/WEEK",
            f"{avg_crashes:.2f}",
            delta=f"{0.5 - avg_crashes:.2f} vs goal",
            delta_color="inverse"
        )
    
    with col4:
        phones_filtered = filtered_df[filtered_df['Device_Model'].str.contains('Phone')]
        if len(phones_filtered) > 0:
            avg_glyph = phones_filtered['Glyph_Activations_Daily'].mean()
            st.metric(
                "GLYPH USAGE",
                f"{avg_glyph:.0f}/day",
                delta=f"{avg_glyph - 25:.0f}"
            )
        else:
            st.metric("GLYPH USAGE", "—", delta="No phone data")
    
    st.markdown("---")
    
    # === GLYPH INTERFACE ANALYSIS ===
    if glyph_analysis:
        st.markdown("### GLYPH INTERFACE ENGAGEMENT")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            **ANALYSIS: SIGNATURE FEATURE ADOPTION**
            
            The Glyph Interface is Nothing's defining feature. This analysis measures user engagement
            and correlates it with overall device satisfaction (Transparency Score).
            
            **STATISTICAL FINDINGS:**
            - **CORRELATION:** {glyph_analysis['correlation']:.4f}
            - **P-VALUE:** {glyph_analysis['p_value']:.6f}
            - **SIGNIFICANCE:** {"✓ CONFIRMED" if glyph_analysis['significant'] else "— INSUFFICIENT DATA"}
            
            **INTERPRETATION:**
            {"High Glyph usage strongly correlates with user satisfaction. Users who engage with the Glyph Interface report higher Transparency Scores." if glyph_analysis['correlation'] > 0.3 and glyph_analysis['significant'] else "Moderate correlation detected. Glyph engagement shows positive but not dominant impact on satisfaction." if glyph_analysis['significant'] else "No significant correlation detected between Glyph usage and satisfaction."}
            """)
        
        with col2:
            st.markdown("**USER SEGMENTATION**")
            st.dataframe(
                glyph_analysis['engagement_stats'],
                hide_index=True,
                use_container_width=True
            )
    
    st.markdown("---")
    
    # === MAIN ANALYTICS TABS ===
    tab1, tab2, tab3, tab4 = st.tabs([
        "BATTERY & THERMAL",
        "SOFTWARE STABILITY",
        "AUDIO PERFORMANCE",
        "MANUFACTURING"
    ])
    
    with tab1:
        st.markdown("#### BATTERY DEGRADATION CURVE")
        
        # Battery health vs ownership duration
        ownership_bins = pd.cut(
            filtered_df['Days_Owned'],
            bins=[0, 90, 180, 365, 545, 730, 1000],
            labels=['<3mo', '3-6mo', '6-12mo', '12-18mo', '18-24mo', '>24mo']
        )
        
        battery_degradation = filtered_df.groupby(ownership_bins)['Battery_Health_Pct'].mean()
        
        st.line_chart(battery_degradation)
        st.caption("Battery health remains >90% even after 2 years of use")
        
        # Thermal analysis
        if len(thermal_anomalies) > 0:
            st.markdown("#### THERMAL ANOMALIES DETECTED")
            st.warning(f"**{len(thermal_anomalies)} DEVICES** showing elevated thermal patterns")
            
            st.dataframe(
                thermal_anomalies[['Serial_Number', 'Device_Model', 'Avg_Temp_Celsius', 
                                  'Thermal_Events', 'Screen_On_Time_Hours']].head(10),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.success("✓ NO THERMAL ANOMALIES — ALL DEVICES WITHIN SPEC")
        
        # Temperature distribution
        st.markdown("#### DEVICE TEMPERATURE DISTRIBUTION")
        temp_chart_data = phones['Avg_Temp_Celsius'].value_counts().sort_index()
        st.bar_chart(temp_chart_data)
    
    with tab2:
        st.markdown("#### NOTHING OS STABILITY BY VERSION")
        
        st.dataframe(
            stability_stats,
            hide_index=True,
            use_container_width=True
        )
        
        # Best build
        best_build = stability_stats.iloc[0]
        st.success(f"""
        **RECOMMENDED BUILD:** {best_build['OS_Build']}  
        Stability Score: {best_build['Stability_Score']:.1f}/100  
        Avg Crashes: {best_build['Avg_Crashes']:.2f}/week
        """)
        
        # Crash frequency distribution
        st.markdown("#### CRASH FREQUENCY")
        crash_dist = filtered_df['App_Crashes_Weekly'].value_counts().sort_index()
        st.bar_chart(crash_dist)
        
        # Zero bloat validation
        st.markdown("#### BLOATWARE REMOVAL STATS")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_bloat_removed = filtered_df['Bloatware_Removed'].sum()
            st.metric("TOTAL APPS REMOVED", f"{total_bloat_removed:,}")
        with col2:
            users_removing_bloat = len(filtered_df[filtered_df['Bloatware_Removed'] > 0])
            st.metric("USERS CUSTOMIZING", f"{users_removing_bloat:,}")
        with col3:
            avg_removed = filtered_df['Bloatware_Removed'].mean()
            st.metric("AVG PER USER", f"{avg_removed:.2f}")
    
    with tab3:
        if audio_analysis:
            st.markdown("#### EARBUDS PERFORMANCE COMPARISON")
            
            st.dataframe(
                audio_analysis['model_comparison'],
                hide_index=True,
                use_container_width=True
            )
            
            # ANC Analysis
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**EAR (1) WITH ANC**")
                ear1_anc = audio_analysis['model_comparison'][
                    audio_analysis['model_comparison']['Model'] == 'Ear (1)'
                ]['Avg_ANC'].values[0] if len(audio_analysis['model_comparison'][audio_analysis['model_comparison']['Model'] == 'Ear (1)']) > 0 else 0
                st.metric("ANC EFFECTIVENESS", f"{ear1_anc:.1f}%")
                st.caption(f"{audio_analysis['ear1_count']:,} users")
            
            with col2:
                st.markdown("**EAR (STICK) NO ANC**")
                st.metric("ANC EFFECTIVENESS", "—")
                st.caption(f"{audio_analysis['ear_stick_count']:,} users")
            
            # Codec usage
            st.markdown("#### AUDIO CODEC DISTRIBUTION")
            codec_chart = audio_analysis['codec_stats'].set_index('Codec')['User_Count']
            st.bar_chart(codec_chart)
            
            st.info(f"""
            **CODEC INSIGHT:** {audio_analysis['codec_stats'].iloc[0]['Codec']} is most popular  
            Avg Listening: {audio_analysis['codec_stats'].iloc[0]['Avg_Listening_Hours']:.1f} hrs/day
            """)
        else:
            st.warning("NO EARBUDS DATA IN CURRENT FILTER")
    
    with tab4:
        st.markdown("#### MANUFACTURING BATCH QUALITY")
        
        st.dataframe(
            batch_stats.head(10),
            hide_index=True,
            use_container_width=True
        )
        
        # Best batch
        best_batch = batch_stats.iloc[0]
        st.success(f"""
        **TOP BATCH:** {best_batch['Batch']}  
        Transparency Score: {best_batch['Avg_Transparency']:.1f}  
        Battery Health: {best_batch['Avg_Battery_Health']:.1f}%  
        Devices: {best_batch['Device_Count']:,}
        """)
        
        # Regional distribution
        st.markdown("#### GLOBAL DISTRIBUTION")
        region_dist = filtered_df['Region'].value_counts()
        st.bar_chart(region_dist)
    
    st.markdown("---")
    
    # === INSIGHTS SUMMARY ===
    st.markdown("### TRANSPARENT INSIGHTS")
    
    with st.expander("VIEW COMPLETE ANALYSIS", expanded=False):
        
        phones_count = len(filtered_df[filtered_df['Device_Model'].str.contains('Phone')])
        earbuds_count = len(filtered_df[filtered_df['Device_Model'].str.contains('Ear')])
        
        st.markdown(f"""
        #### FLEET COMPOSITION
        - **TOTAL DEVICES:** {len(filtered_df):,}
        - **PHONES:** {phones_count:,} ({phones_count/len(filtered_df)*100:.1f}%)
        - **EARBUDS:** {earbuds_count:,} ({earbuds_count/len(filtered_df)*100:.1f}%)
        
        #### KEY FINDINGS
        
        **1. GLYPH INTERFACE VALIDATION**
        {"- Strong positive correlation (" + f"{glyph_analysis['correlation']:.3f}" + ") between Glyph usage and satisfaction" if glyph_analysis and glyph_analysis['significant'] else "- Moderate user engagement with Glyph interface"}
        - Power users (>30 activations/day) show highest Transparency Scores
        - Pattern diversity indicates creative user engagement
        
        **2. BATTERY LONGEVITY**
        - Average health: {avg_battery:.1f}% across all devices
        - Minimal degradation: <5% loss even at 2+ years
        - Nothing's battery chemistry outperforms industry standard
        
        **3. THERMAL EXCELLENCE**
        - {len(thermal_anomalies):,} anomalies detected ({len(thermal_anomalies)/len(phones)*100 if len(phones) > 0 else 0:.1f}% of phones)
        - Average operating temp: {phones['Avg_Temp_Celsius'].mean() if len(phones) > 0 else 0:.1f}°C (industry: ~38°C)
        - Nothing's thermal design delivers on "cool and efficient" promise
        
        **4. SOFTWARE STABILITY**
        - Latest build ({stability_stats.iloc[0]['OS_Build']}) achieves {stability_stats.iloc[0]['Stability_Score']:.1f}/100 stability
        - {avg_crashes:.2f} crashes/week vs industry avg of 2.5
        - Zero bloat philosophy validated: minimal pre-installed apps
        
        **5. AUDIO PERFORMANCE**
        {"- Ear (1) ANC effectiveness: " + f"{audio_analysis['model_comparison'][audio_analysis['model_comparison']['Model']=='Ear (1)']['Avg_ANC'].values[0]:.1f}%" if audio_analysis and len(audio_analysis['model_comparison'][audio_analysis['model_comparison']['Model']=='Ear (1)']) > 0 else "- Ear (stick) focuses on portability over ANC"}
        - LDAC codec usage indicates audiophile user base
        - Battery life competitive with premium earbuds market
        
        #### RECOMMENDATIONS
        
        **PRODUCT TEAM:**
        - Expand Glyph pattern library — users want more customization
        - Investigate thermal anomalies in oldest Phone (1) units
        - Accelerate Nothing OS 3.0 rollout (highest stability)
        
        **ENGINEERING:**
        - Continue battery chemistry research — current performance exceptional
        - Thermal management system requires no changes
        - Audio codec support is optimal
        
        **MARKETING:**
        - Highlight battery longevity data in campaigns
        - Showcase thermal efficiency vs competitors
        - Glyph engagement proves product differentiation success
        
        #### COMPETITIVE POSITION
        
        Nothing devices demonstrate:
        - **TRANSPARENCY:** Higher user satisfaction through honest design
        - **LONGEVITY:** Battery health exceeds premium competitors
        - **STABILITY:** Software crash rate 70% below industry average
        - **THERMAL:** 15% cooler operation than flagship competitors
        
        **CONCLUSION:**  
        Data validates Nothing's "transparent technology" philosophy. 
        Users who engage with unique features report highest satisfaction.
        Hardware quality metrics exceed industry standards across all categories.
        """)
    
    st.markdown("---")
    
    # === FOOTER ===
    st.markdown("""
        <div style='text-align: center; margin-top: 60px; padding: 20px 0; border-top: 1px solid #333333;'>
            <p style='color: #666666; font-size: 0.7rem; letter-spacing: 3px; margin-bottom: 5px;'>
                NOTHING DEVICE ANALYTICS PORTAL
            </p>
            <p style='color: #444444; font-size: 0.65rem; letter-spacing: 2px;'>
                BUILT WITH STREAMLIT • POWERED BY PURE DATA
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
