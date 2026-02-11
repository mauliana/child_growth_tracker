import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="Child Growth Tracker", layout="wide")
API_BASE_URL = "http://localhost:5000/api"

# --- API HELPER FUNCTIONS ---
def api_request(method, endpoint, data=None, params=None):
    """Generic API request handler"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def get_children():
    """Get all children from API"""
    result = api_request("GET", "/children")
    if result and result.get('success'):
        df = pd.DataFrame(result['data'])
        if not df.empty:
            df['dob'] = pd.to_datetime(df['dob'])
            df = df.set_index('id')
        return df
    return pd.DataFrame()

def get_child_info(child_id):
    """Get single child info"""
    result = api_request("GET", f"/children/{child_id}")
    if result and result.get('success'):
        data = result['data']
        data['dob'] = pd.to_datetime(data['dob'])
        return pd.Series(data)
    return None

def get_child_measurements(child_id):
    """Get all measurements for a child"""
    result = api_request("GET", f"/measurements/{child_id}")
    if result and result.get('success'):
        df = pd.DataFrame(result['data'])
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['dob'] = pd.to_datetime(df['dob'])
            df = df.set_index('id')
        return df
    return pd.DataFrame()

def save_child(name, gender, dob):
    """Create new child"""
    data = {
        'name': name,
        'gender': gender,
        'dob': dob.strftime('%Y-%m-%d')
    }
    result = api_request("POST", "/children", data=data)
    if result and result.get('success'):
        st.session_state.notification = result.get('message')
        return True
    return False

def save_measurement(child_id, dob, meas_date, weight, height, head_circumference):
    """Create new measurement"""
    data = {
        'child_id': child_id,
        'dob': dob.strftime('%Y-%m-%d'),
        'date': meas_date.strftime('%Y-%m-%d'),
        'weight': weight,
        'height': height,
        'head_circumference': head_circumference
    }
    result = api_request("POST", "/measurements", data=data)
    if result and result.get('success'):
        st.session_state.notification = result.get('message')
        return True
    return False

def delete_child(child_id):
    """Delete child"""
    result = api_request("DELETE", f"/children/{child_id}")
    if result and result.get('success'):
        st.session_state.notification = result.get('message')
        return True
    return False

def bulk_update_measurements(child_id, measurements_df):
    """Bulk update measurements"""
    measurements = []
    for _, row in measurements_df.iterrows():
        measurements.append({
            'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date'],
            'dob': row['dob'].strftime('%Y-%m-%d') if hasattr(row['dob'], 'strftime') else row['dob'],
            'weight': row.get('weight'),
            'height': row.get('height'),
            'head_circumference': row.get('head_circumference')
        })
    
    data = {
        'child_id': child_id,
        'measurements': measurements
    }
    result = api_request("POST", "/measurements/bulk", data=data)
    if result and result.get('success'):
        st.session_state.notification = result.get('message')
        return True
    return False

def get_who_standards(gender, metric_type, age_months):
    """Get WHO growth standards"""
    params = {
        'gender': gender,
        'metric_type': metric_type,
        'age_months': age_months
    }
    result = api_request("GET", "/who-standards", params=params)
    if result and result.get('success'):
        df = pd.DataFrame(result['data'])
        return df, result['axis_type'], result['x_col']
    return pd.DataFrame(), None, None

def get_health_status(measurements_df, gender):
    """Calculate health status"""
    if measurements_df.empty:
        return {}
    
    measurements = []
    for _, row in measurements_df.iterrows():
        measurements.append({
            'age_months': row['age_months'],
            'weight': row['weight'],
            'height': row['height'],
            'bmi': row.get('bmi')
        })
    
    data = {
        'measurements': measurements,
        'gender': gender
    }
    result = api_request("POST", "/health-status", data=data)
    if result and result.get('success'):
        return result.get('statuses', {})
    return {}

def get_trend_analysis(measurements_df, gender):
    """Calculate trend analysis"""
    if measurements_df.empty or len(measurements_df) < 2:
        return {
            'height': 'Not enough data',
            'weight': 'Not enough data'
        }
    
    measurements = []
    for _, row in measurements_df.iterrows():
        measurements.append({
            'age_months': row['age_months'],
            'weight': row['weight'],
            'height': row['height'],
        })
    
    data = {
        'measurements': measurements,
        'gender': gender
    }
    result = api_request("POST", "/trend-analysis", data=data)
    if result and result.get('success'):
        return result.get('trends', {})
    return {
        'height': 'Analysis Error',
        'weight': 'Analysis Error'
    }

def generate_test_data(scenario):
    """Generate test data"""
    data = {'scenario': scenario}
    result = api_request("POST", "/test-data", data=data)
    if result and result.get('success'):
        st.toast(result.get('message'), icon="🧪")
        return True
    return False

def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI locally for display"""
    if pd.isna(weight_kg) or pd.isna(height_cm) or height_cm == 0:
        return None
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 2)

# --- NOTIFICATION HANDLER ---
if "notification" in st.session_state:
    st.toast(st.session_state.notification, icon="✅")
    del st.session_state["notification"]

# --- MAIN LAYOUT ---
st.title("👶 Child Growth Tracker")

tab1, tab2 = st.tabs(["📝 Data Record", "📊 Dashboard"])

# =======================
# TAB 1: REGISTER CHILD & MANAGE DATA
# =======================
with tab1:
    children_df = get_children()

    col1, col2 = st.columns([1, 3])

    # --- LEFT COLUMN: REGISTRATION FORM ---
    with col1:
        st.header("📝 Register New Child")
        with st.form("register_form", clear_on_submit=True):
            name = st.text_input("Child Name")
            gender = st.selectbox("Gender", ["Boys", "Girls"], key="tab1_reg_gender")
            dob = st.date_input("Date of Birth", key="tab1_reg_dob")
            submitted = st.form_submit_button("Save Child Profile")
            
            if submitted and name:
                if save_child(name, gender, dob):
                    st.rerun()

        st.divider()
        
        # ---TEST DATA GENERATOR ---
        st.header("🧪 Test Data Generator")
        st.caption("Click to generate sample data to test Trend Analysis.")
        
        col_gen1, col_gen2 = st.columns(2)
        with col_gen1:
            if st.button("Generate Stable Data", width='stretch'):
                if generate_test_data("stable"):
                    st.rerun()
                
        with col_gen2:
            if st.button("Generate Dropping Data", width='stretch'):
                if generate_test_data("dropping"):
                    st.rerun()
        
        col_gen3, col_gen4 = st.columns(2)
        with col_gen3:
            if st.button("Generate Older Stable", width='stretch'):
                if generate_test_data("older_stable"):
                    st.rerun()
                
        with col_gen4:
            if st.button("Generate Older Dropping", width='stretch'):
                if generate_test_data("older_dropping"):
                    st.rerun()

    # --- RIGHT COLUMN: EDIT & VIEW DATA ---
    with col2:
        st.header("📂 Manage Children Data")
        
        if children_df.empty:
            st.info("No children registered yet.")
        else:
            # Display children profiles
            st.subheader("Children Profiles")
            
            for idx, row in children_df.iterrows():
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.write(f"**{row['name']}** - {row['gender']} - DOB: {row['dob'].strftime('%Y-%m-%d')}")
                with col_b:
                    if st.button("📊 View", key=f"view_{idx}"):
                        st.session_state.selected_child_tab2 = idx
                        st.rerun()
                with col_c:
                    if st.button("🗑️ Delete", key=f"del_{idx}"):
                        if delete_child(idx):
                            st.rerun()

            st.divider()
            
            # Measurement Data Editor
            st.subheader("📊 Measurement Data")
            
            options = [(idx, row['name']) for idx, row in children_df.iterrows()]
            
            selected_child_id = st.selectbox(
                "Select a child to view/edit measurements:", 
                options=options,
                format_func=lambda x: x[1],
                key="tab1_meas_selector"
            )
            
            if selected_child_id:
                child_id = selected_child_id[0]
                child_measures = get_child_measurements(child_id)
                
                if child_measures.empty:
                    st.caption("No measurements recorded for this child.")
                else:
                    edited_measures = st.data_editor(
                        child_measures,
                        num_rows="dynamic",
                        column_config={
                            "child_id": None, 
                            "dob": None,      
                            "date": st.column_config.DateColumn("Date", width="medium"),
                            "age_months": st.column_config.NumberColumn("Age (Mo)", disabled=True, format="%.1f"),
                            "age_weeks": st.column_config.NumberColumn("Age (Wk)", disabled=True, format="%.1f"),
                            "weight": st.column_config.NumberColumn("Weight (kg)", format="%.2f"),
                            "height": st.column_config.NumberColumn("Height (cm)", format="%.1f"),
                            "head_circumference": st.column_config.NumberColumn("Head (cm)", format="%.1f"),
                            "bmi": st.column_config.NumberColumn("BMI", disabled=True, format="%.2f")
                        },
                        width='stretch',
                        hide_index=True,
                        key=f"editor_measures_{child_id}"
                    )
                    
                    if st.button("💾 Save Measurement Changes", key="btn_save_meas_tab1"):
                        if bulk_update_measurements(child_id, edited_measures):
                            st.rerun()

# =======================
# TAB 2: DASHBOARD
# =======================
with tab2:
    children_df = get_children()
    
    if children_df.empty:
        st.warning("No children found. Please go to 'Data Record' tab first.")
    else:
        col1, col2 = st.columns([1, 3])

        # --- LEFT COLUMN: SELECT CHILD & ADD DATA ---
        with col1:
            st.subheader("Select Child")
            
            options = [(idx, row['name']) for idx, row in children_df.iterrows()]
            
            # Use session state if available
            default_idx = 0
            if 'selected_child_tab2' in st.session_state:
                for i, (idx, _) in enumerate(options):
                    if idx == st.session_state.selected_child_tab2:
                        default_idx = i
                        break
            
            selected_id = st.selectbox(
                "Choose a profile:", 
                options=options,
                format_func=lambda x: x[1],
                index=default_idx,
                key="tab2_dash_selector"
            )
            
            child_id = selected_id[0]
            child_info = get_child_info(child_id)
            
            if child_info is not None:
                dob_str = child_info['dob'].strftime('%Y-%m-%d')
                st.info(f"**Name:** {child_info['name']}\n\n**Gender:** {child_info['gender']}\n\n**DOB:** {dob_str}")

                st.divider()
                
                # --- ADD MEASUREMENT SECTION ---
                st.subheader("Add Measurement")
                
                with st.form("add_measurement_form", clear_on_submit=True):
                    meas_date = st.date_input("Measurement Date", value=date.today())
                    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1, format="%.2f")
                    height = st.number_input("Height (cm)", min_value=0.0, step=0.1, format="%.1f")
                    head = st.number_input("Head Circumference (cm)", min_value=0.0, step=0.1, format="%.1f")
                    
                    submitted = st.form_submit_button("Add Record", type="primary")
                    
                    if submitted:
                        if save_measurement(child_id, child_info['dob'], meas_date, weight, height, head):
                            st.rerun()

        # --- RIGHT COLUMN: CHARTS ---
        with col2:
            child_measures = get_child_measurements(child_id)
            
            if not child_measures.empty:
                child_measures = child_measures.sort_values(by='date')
                
                # Get gender from child_info
                gender = child_info['gender']

                # 3. HEALTH & TREND OVERVIEW
                status_meanings = {
                    "Severely Wasted": "Critical malnutrition. Needs immediate doctor.",
                    "Wasted": "Underweight/Malnourished. Needs diet change.",
                    "Normal": "Healthy weight. Keep it up!",
                    "Overweight": "Too heavy. Risk of obesity.",
                    "Stunted": "Growth stunted compared to average.",
                    "Severely Stunted": "Severe growth impairment.",
                    "No Data": "Insufficient data to analyze.",
                    "Thinness": "Below healthy BMI (5-18y)",
                    "Obesity": "Above healthy BMI (5-18y)",
                    "Risk of Overweight": "Monitor nutrition closely",
                    "Obese": "Significantly above healthy weight"
                }

                latest = child_measures.iloc[-1]
                latest_age = latest['age_months']
                latest_weight = latest['weight']
                latest_height = latest['height']
                latest_bmi = latest.get('bmi') or calculate_bmi(latest_weight, latest_height)

                # Get health status
                health_status = get_health_status(child_measures, gender)
                
                st.subheader("📊 Health & Trend Overview")
                
                status_col, trend_col = st.columns([1, 1])

                with status_col:
                    st.markdown("### 🏥 Current Status")
                    
                    g_col, w_col = st.columns(2)
                    
                    with g_col:
                        growth_status = health_status.get('growth', 'No Data')
                        st.metric("Growth", growth_status)
                        st.caption(status_meanings.get(growth_status, ""))
                        
                    with w_col:
                        weight_status = health_status.get('weight', 'No Data')
                        st.metric("Weight", weight_status)
                        st.caption(status_meanings.get(weight_status, ""))
                    
                    if latest_bmi:
                        bmi_status = health_status.get('bmi', 'No Data')
                        st.metric("BMI", f"{bmi_status} ({latest_bmi:.1f})")
                        st.caption(status_meanings.get(bmi_status, ""))
                        
                    st.caption(f"*Based on: {latest['date'].date()}*")

                with trend_col:
                    st.markdown("### 📈 Trend Analysis")
                    
                    trend_meanings = {
                        "⚠️ Dropping (Risk)": "Child's growth percentile is decreasing. Predicts risk of stunting.",
                        "📈 Rising": "Child is improving across percentiles.",
                        "➡️ Stable": "Child is maintaining their growth curve.",
                        "Not enough data": "Need more than 1 record to see trend.",
                    }
                    
                    # Get trend analysis from API
                    trend_analysis = get_trend_analysis(child_measures, gender)
                    
                    hfa_trend = trend_analysis.get('height', 'Not enough data')
                    st.write(f"**Height:** {hfa_trend}")
                    st.caption(trend_meanings.get(hfa_trend, ""))
                    
                    # Only show weight trend for > 24 months
                    if latest_age > 24:
                        wfa_trend = trend_analysis.get('weight', 'Not enough data')
                        if wfa_trend:
                            st.write(f"**Weight:** {wfa_trend}")
                            st.caption(trend_meanings.get(wfa_trend, ""))

                st.divider()

            # 4. CHART TABS
            if child_measures.empty:
                st.info("No measurements recorded for this child yet.")
            else:
                latest_age = child_measures.iloc[-1]['age_months']
                
                chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5 = st.tabs([
                    "Weight-for-Age", 
                    "Length/Height-for-Age", 
                    "Weight-for-Length/Height", 
                    "Head-Circumference-for-Age", 
                    "BMI-for-Age"
                ])

                z_styles = {
                    'SD3neg': {'color': "#060606", 'dash': 'dot', 'name': '-3 SD'},
                    'SD2neg': {'color': '#ff4b4b', 'dash': 'dash', 'name': '-2 SD'},
                    'SD0': {'color': '#00c04b', 'dash': 'solid', 'name': 'Median'},
                    'SD2': {'color': "#ff4b4b", 'dash': 'dash', 'name': '+2 SD'},
                    'SD3': {'color': "#060606", 'dash': 'dot', 'name': '+3 SD'}
                }

                # --- WEIGHT CHART ---
                with chart_tab1:
                    df_who, axis_type, x_col = get_who_standards(gender, 'wfa', latest_age)
                    
                    if not df_who.empty:
                        fig = go.Figure()
                        x_label = "Age (Weeks)" if axis_type == "Week" else "Age (Months)"
                        
                        for col, style in z_styles.items():
                            if col in df_who.columns:
                                fig.add_trace(go.Scatter(
                                    x=df_who[x_col], 
                                    y=df_who[col],
                                    line=dict(color=style['color'], dash=style['dash'], width=1.5),
                                    name=style['name'],
                                    mode='lines'
                                ))
                        
                        user_x = child_measures['age_weeks'] if axis_type == "Week" else child_measures['age_months']
                        
                        fig.add_trace(go.Scatter(
                            x=user_x, 
                            y=child_measures['weight'],
                            mode='markers+lines', name='Child Data',
                            marker=dict(size=8, color='#4b9fff'), line=dict(width=2)
                        ))

                        fig.update_layout(
                            title=f"Weight-for-Age ({axis_type})", 
                            xaxis_title=x_label, 
                            yaxis_title="Weight (kg)"
                        )
                        st.plotly_chart(fig, width='stretch')

                # --- HEIGHT CHART ---
                with chart_tab2:
                    df_who, axis_type, x_col = get_who_standards(gender, 'lhfa', latest_age)
                    
                    if not df_who.empty:
                        fig = go.Figure()
                        x_label = "Age (Weeks)" if axis_type == "Week" else "Age (Months)"
                        
                        for col, style in z_styles.items():
                            if col in df_who.columns:
                                fig.add_trace(go.Scatter(
                                    x=df_who[x_col], 
                                    y=df_who[col],
                                    line=dict(color=style['color'], dash=style['dash'], width=1.5),
                                    name=style['name'],
                                    mode='lines'
                                ))
                        
                        user_x = child_measures['age_weeks'] if axis_type == "Week" else child_measures['age_months']
                        
                        fig.add_trace(go.Scatter(
                            x=user_x, 
                            y=child_measures['height'],
                            mode='markers+lines', name='Child Data',
                            marker=dict(size=8, color='#4b9fff'), line=dict(width=2)
                        ))

                        fig.update_layout(
                            title=f"Length/Height-for-Age ({axis_type})", 
                            xaxis_title=x_label, 
                            yaxis_title="Height (cm)"
                        )
                        st.plotly_chart(fig, width='stretch')
                
                # --- WEIGHT FOR LENGTH/HEIGHT CHART ---
                with chart_tab3:
                    df_who, axis_type, x_col = get_who_standards(gender, 'wfl', latest_age)
                    
                    if not df_who.empty:
                        fig = go.Figure()
                        x_label = axis_type + " (cm)"
                        
                        for col, style in z_styles.items():
                            if col in df_who.columns:
                                fig.add_trace(go.Scatter(
                                    x=df_who[x_col], 
                                    y=df_who[col],
                                    line=dict(color=style['color'], dash=style['dash'], width=1.5),
                                    name=style['name'],
                                    mode='lines'
                                ))
                        
                        fig.add_trace(go.Scatter(
                            x=child_measures['height'], 
                            y=child_measures['weight'],
                            mode='markers+lines', name='Child Data',
                            marker=dict(size=8, color='#4b9fff'), line=dict(width=2)
                        ))

                        fig.update_layout(
                            title=f"Weight-for-{axis_type}", 
                            xaxis_title=x_label, 
                            yaxis_title="Weight (kg)"
                        )
                        st.plotly_chart(fig, width='stretch')
                    
                # HEAD CIRCUMFERENCE FOR AGE
                with chart_tab4:
                    df_who, axis_type, x_col = get_who_standards(gender, 'hcfa', latest_age)

                    if not df_who.empty:
                        fig = go.Figure()
                        x_label = "Age (Weeks)" if axis_type == "Week" else "Age (Months)"
                        
                        for col, style in z_styles.items():
                            if col in df_who.columns:
                                fig.add_trace(go.Scatter(
                                    x=df_who[x_col], 
                                    y=df_who[col],
                                    line=dict(color=style['color'], dash=style['dash'], width=1.5),
                                    name=style['name'],
                                    mode='lines'
                                ))
                        
                        user_x = child_measures['age_weeks'] if axis_type == "Week" else child_measures['age_months']
                        
                        fig.add_trace(go.Scatter(
                            x=user_x, 
                            y=child_measures['head_circumference'],
                            mode='markers+lines', name='Child Data',
                            marker=dict(size=8, color='#4b9fff'), line=dict(width=2)
                        ))

                        fig.update_layout(
                            title=f"Head-Circumference-for-Age ({axis_type})", 
                            xaxis_title=x_label, 
                            yaxis_title="Head Circumference (cm)"
                        )
                        st.plotly_chart(fig, width='stretch')
                
                # BMI FOR AGE
                with chart_tab5:
                    # Calculate BMI if not in dataframe
                    if 'bmi' not in child_measures.columns or child_measures['bmi'].isna().all():
                        child_measures['bmi'] = child_measures.apply(
                            lambda row: calculate_bmi(row['weight'], row['height']), 
                            axis=1
                        )

                    df_who, axis_type, x_col = get_who_standards(gender, 'bmi', latest_age)

                    if not df_who.empty:
                        fig = go.Figure()
                        x_label = "Age (Weeks)" if axis_type == "Week" else "Age (Months)"
                        
                        for col, style in z_styles.items():
                            if col in df_who.columns:
                                fig.add_trace(go.Scatter(
                                    x=df_who[x_col], 
                                    y=df_who[col],
                                    line=dict(color=style['color'], dash=style['dash'], width=1.5),
                                    name=style['name'],
                                    mode='lines'
                                ))
                        
                        user_x = child_measures['age_weeks'] if axis_type == "Week" else child_measures['age_months']
                        
                        fig.add_trace(go.Scatter(
                            x=user_x, 
                            y=child_measures['bmi'],
                            mode='markers+lines', name='Child Data',
                            marker=dict(size=8, color='#4b9fff'), line=dict(width=2)
                        ))

                        fig.update_layout(
                            title=f"BMI-for-Age ({axis_type})", 
                            xaxis_title=x_label, 
                            yaxis_title="BMI (kg/m²)"
                        )
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning("WHO BMI standards not available for this age range.")
