import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime
import sqlite3
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Child Growth Tracker", layout="wide")

# Configuration
BASE_PATH = "./who_metrics"
DB_FILE = "growth_tracker.db"

# ==================== DATABASE FUNCTIONS ====================

def init_db():
    """Initialize database with proper schema"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS children
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  name TEXT NOT NULL, 
                  gender TEXT NOT NULL, 
                  dob TEXT NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS measurements
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  child_id INTEGER NOT NULL, 
                  date TEXT NOT NULL, 
                  dob TEXT NOT NULL, 
                  age_months REAL, 
                  age_weeks REAL, 
                  weight REAL, 
                  height REAL, 
                  head_circumference REAL, 
                  bmi REAL,
                  FOREIGN KEY (child_id) REFERENCES children(id))''')
    
    conn.commit()
    conn.close()

# Initialize database on app start
init_db()

# ==================== HELPER FUNCTIONS ====================

def calculate_age_in_months(birth_date, measurement_date):
    """Calculate age in months and weeks"""
    try:
        b_ts = pd.Timestamp(birth_date)
        m_ts = pd.Timestamp(measurement_date)
        
        if pd.isna(b_ts) or pd.isna(m_ts): 
            return 0.0, 0.0
            
        days_diff = (m_ts - b_ts).days
        months = round(days_diff / 30.4375, 2)
        weeks = round(days_diff / 7.0, 2)
        return months, weeks
    except Exception as e:
        return 0.0, 0.0

def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight (kg) and height (cm)"""
    try:
        if pd.isna(weight_kg) or pd.isna(height_cm) or height_cm == 0:
            return None
        height_m = height_cm / 100.0
        return round(weight_kg / (height_m ** 2), 2)
    except:
        return None

def get_who_standards(gender, metric_type, age_in_months):
    """Load WHO growth standards from Excel files"""
    gender_code = gender.lower()
    
    # Determine file and Axis Type
    if metric_type == 'wfa':
        if age_in_months < 3:
            age_str = "0-to-13-weeks"
            axis_type = "Week"
        else:
            age_str = "0-to-5-years"
            axis_type = "Month"
            
    elif metric_type == 'lhfa':
        if age_in_months < 3:
            age_str = "0-to-13-weeks"
            axis_type = "Week"
        elif age_in_months <= 24:
            age_str = "0-to-2-years"
            axis_type = "Month"
        else:
            age_str = "2-to-5-years"
            axis_type = "Month"
    
    elif metric_type == 'hcfa':
        if age_in_months < 3:
            age_str = "0-13"
            axis_type = "Week"
        else:
            age_str = "0-5"
            axis_type = "Month"
    
    elif metric_type == 'bmi':
        if age_in_months < 3:
            age_str = "0-to-13-weeks"
            axis_type = "Week"
        elif age_in_months <= 24:
            age_str = "0-to-2-years"
            axis_type = "Month"
        else:
            age_str = "2-to-5-years"
            axis_type = "Month"

    elif metric_type == 'wfl':
        if age_in_months <= 24:
            age_str = "0-to-2-years"
            axis_type = "Length"
        else:
            age_str = "2-to-5-years"
            axis_type = "Height"
    else:
        return None, None, None

    # Determine folder and file prefix
    folder_name = 'wfh' if metric_type == 'wfl' else metric_type
    file_prefix = metric_type
    if metric_type == 'wfl' and age_str == "2-to-5-years":
        file_prefix = 'wfh'

    filename = f"{file_prefix}_{gender_code}_{age_str}_zscores.xlsx"
    full_path = os.path.join(BASE_PATH, folder_name, filename)
    
    try:
        df = pd.read_excel(full_path, engine='openpyxl')
        df.columns = [c.strip() for c in df.columns]
        
        # Auto-detect X-axis column
        possible_names = ['Month', 'Months', 'Week', 'Weeks', 'Length', 'Height']
        x_col_name = None
        for name in possible_names:
            if name in df.columns:
                x_col_name = name
                break
        
        if x_col_name is None:
            return None, axis_type, None
            
        return df, axis_type, x_col_name
        
    except Exception as e:
        print(f"Error loading WHO standards: {e}")
        return None, None, None

def get_zscore_from_who(gender, metric_type, age_months, value):
    """
    Calculate Z-score using WHO SD curves
    metric_type: 'lhfa' or 'wfa'
    """
    df, _, x_col = get_who_standards(gender, metric_type, age_months)

    if df is None or df.empty:
        return None

    df['diff'] = (df[x_col] - age_months).abs()
    row = df.loc[df['diff'].idxmin()]

    sd0 = row['SD0']
    sd2 = row.get('SD2')
    sd2neg = row.get('SD2neg')

    if pd.isna(sd0) or pd.isna(sd2) or pd.isna(sd2neg):
        return None

    # WHO SD is asymmetric, approximate from median
    if value >= sd0:
        sd = (sd2 - sd0) / 2
    else:
        sd = (sd0 - sd2neg) / 2

    if sd == 0:
        return None

    return round((value - sd0) / sd, 2)

def analyze_trend_from_measurements(measurements, gender, metric):
    """
    metric: 'height' or 'weight'
    """
    who_metric = 'lhfa' if metric == 'height' else 'wfa'

    ages = []
    z_scores = []

    for _, m in measurements.iterrows():
        age = m.get('age_months')
        value = m.get(metric)

        if age is None or value is None or pd.isna(age) or pd.isna(value):
            continue

        z = get_zscore_from_who(
            gender=gender,
            metric_type=who_metric,
            age_months=age,
            value=value
        )

        if z is not None:
            ages.append(age)
            z_scores.append(z)

    if len(z_scores) < 2:
        return "Not enough data"

    # Linear regression slope
    slope = np.polyfit(ages, z_scores, 1)[0]

    if slope < -0.05:
        return "⚠️ Dropping (Risk)"
    elif slope > 0.05:
        return "📈 Rising"
    else:
        return "➡️ Stable"

# ==================== DATABASE OPERATIONS ====================

@st.cache_data(ttl=1)
def get_children():
    """Get all children from database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql("SELECT * FROM children ORDER BY id", conn)
        conn.close()
        
        if not df.empty:
            df['dob'] = pd.to_datetime(df['dob'])
            df = df.set_index('id')
        return df
    except Exception as e:
        st.error(f"Error loading children: {str(e)}")
        return pd.DataFrame()

def get_child_info(child_id):
    """Get single child info"""
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql("SELECT * FROM children WHERE id = ?", conn, params=(child_id,))
        conn.close()
        
        if df.empty:
            return None
            
        data = df.iloc[0].to_dict()
        data['dob'] = pd.to_datetime(data['dob'])
        return pd.Series(data)
    except Exception as e:
        st.error(f"Error loading child info: {str(e)}")
        return None

@st.cache_data(ttl=1)
def get_child_measurements(child_id):
    """Get all measurements for a child"""
    try:
        conn = sqlite3.connect(DB_FILE)
        query = """
            SELECT id, child_id, date, dob, age_months, age_weeks, 
                   weight, height, head_circumference, bmi 
            FROM measurements 
            WHERE child_id = ? 
            ORDER BY date
        """
        df = pd.read_sql(query, conn, params=(child_id,))
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['dob'] = pd.to_datetime(df['dob'])
            df = df.set_index('id')
        return df
    except Exception as e:
        st.error(f"Error loading measurements: {str(e)}")
        return pd.DataFrame()

def save_child(name, gender, dob):
    """Create new child"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO children (name, gender, dob) VALUES (?, ?, ?)",
                  (name, gender, dob.strftime('%Y-%m-%d')))
        child_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Clear cache
        get_children.clear()
        
        st.session_state.notification = f'{name} registered successfully!'
        return True
    except Exception as e:
        st.error(f"Error saving child: {str(e)}")
        return False

def save_measurement(child_id, dob, meas_date, weight, height, head_circumference):
    """Create new measurement"""
    try:
        # Calculate derived values
        age_m, age_w = calculate_age_in_months(dob, meas_date)
        bmi = calculate_bmi(weight, height)
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            INSERT INTO measurements 
            (child_id, date, dob, age_months, age_weeks, weight, height, head_circumference, bmi) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (child_id, meas_date.strftime('%Y-%m-%d'), dob.strftime('%Y-%m-%d'), 
              age_m, age_w, weight, height, head_circumference, bmi))
        
        conn.commit()
        conn.close()
        
        # Clear cache
        get_child_measurements.clear()
        
        st.session_state.notification = 'Measurement added successfully'
        return True
    except Exception as e:
        st.error(f"Error saving measurement: {str(e)}")
        return False

def delete_child(child_id):
    """Delete child and all measurements"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Delete measurements first
        c.execute("DELETE FROM measurements WHERE child_id = ?", (child_id,))
        # Delete child
        c.execute("DELETE FROM children WHERE id = ?", (child_id,))
        
        conn.commit()
        conn.close()
        
        # Clear cache
        get_children.clear()
        get_child_measurements.clear()
        
        st.session_state.notification = 'Child deleted successfully'
        return True
    except Exception as e:
        st.error(f"Error deleting child: {str(e)}")
        return False

def bulk_update_measurements(child_id, measurements_df):
    """Bulk update measurements"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Delete old measurements
        c.execute("DELETE FROM measurements WHERE child_id = ?", (child_id,))
        
        # Insert new measurements
        for _, row in measurements_df.iterrows():
            age_m, age_w = calculate_age_in_months(row['dob'], row['date'])
            bmi = calculate_bmi(row.get('weight'), row.get('height'))
            
            c.execute("""
                INSERT INTO measurements 
                (child_id, date, dob, age_months, age_weeks, weight, height, head_circumference, bmi) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (child_id, 
                  row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date'],
                  row['dob'].strftime('%Y-%m-%d') if hasattr(row['dob'], 'strftime') else row['dob'],
                  age_m, age_w, 
                  row.get('weight'), row.get('height'), row.get('head_circumference'), bmi))
        
        conn.commit()
        conn.close()
        
        # Clear cache
        get_child_measurements.clear()
        
        st.session_state.notification = 'Measurements updated successfully'
        return True
    except Exception as e:
        st.error(f"Error updating measurements: {str(e)}")
        return False

def calculate_health_status(measurements_df, gender):
    """Calculate health status based on latest measurement"""
    if measurements_df.empty:
        return {}
    
    # Get latest measurement
    latest = measurements_df.iloc[-1]
    latest_age = latest['age_months']
    latest_weight = latest['weight']
    latest_height = latest['height']
    latest_bmi = latest.get('bmi') or calculate_bmi(latest_weight, latest_height)
    
    # Calculate statuses
    statuses = {}
    
    # Height status (stunting)
    lhfa_data, _, lhfa_x = get_who_standards(gender, 'lhfa', latest_age)
    if lhfa_data is not None and not lhfa_data.empty:
        lhfa_data['diff'] = (lhfa_data[lhfa_x] - latest_age).abs()
        closest = lhfa_data.loc[lhfa_data['diff'].idxmin()]
        
        if latest_height < closest['SD3neg']:
            statuses['growth'] = 'Severely Stunted'
        elif latest_height < closest['SD2neg']:
            statuses['growth'] = 'Stunted'
        else:
            statuses['growth'] = 'Normal'
    
    # Weight status (wasting)
    wfl_data, _, wfl_x = get_who_standards(gender, 'wfl', latest_age)
    if wfl_data is not None and not wfl_data.empty:
        wfl_data['diff'] = (wfl_data[wfl_x] - latest_height).abs()
        closest = wfl_data.loc[wfl_data['diff'].idxmin()]
        
        if latest_weight > closest['SD2']:
            statuses['weight'] = 'Overweight'
        elif latest_weight < closest['SD3neg']:
            statuses['weight'] = 'Severely Wasted'
        elif latest_weight < closest['SD2neg']:
            statuses['weight'] = 'Wasted'
        else:
            statuses['weight'] = 'Normal'
    
    # BMI status
    bmi_data, _, bmi_x = get_who_standards(gender, 'bmi', latest_age)
    if bmi_data is not None and not bmi_data.empty and latest_bmi:
        bmi_data['diff'] = (bmi_data[bmi_x] - latest_age).abs()
        closest = bmi_data.loc[bmi_data['diff'].idxmin()]
        
        if latest_age < 60:  # 0-5 years
            if latest_bmi < closest['SD3neg']:
                statuses['bmi'] = 'Severely Wasted'
            elif latest_bmi < closest['SD2neg']:
                statuses['bmi'] = 'Wasted'
            elif latest_bmi > closest['SD3']:
                statuses['bmi'] = 'Obese'
            elif latest_bmi > closest['SD2']:
                statuses['bmi'] = 'Overweight'
            elif closest.get('SD1') is not None and latest_bmi > closest['SD1']:
                statuses['bmi'] = 'Risk of Overweight'
            else:
                statuses['bmi'] = 'Normal'
        else:  # 5-18 years
            if latest_bmi < closest['SD2neg']:
                statuses['bmi'] = 'Thinness'
            elif latest_bmi > closest['SD2']:
                statuses['bmi'] = 'Obesity'
            elif closest.get('SD1') is not None and latest_bmi > closest['SD1']:
                statuses['bmi'] = 'Overweight'
            else:
                statuses['bmi'] = 'Normal'
    
    return statuses

def get_trend_analysis(measurements_df, gender):
    """Calculate trend analysis"""
    if measurements_df.empty or len(measurements_df) < 2:
        return {
            'height': 'Not enough data',
            'weight': 'Not enough data'
        }
    
    height_trend = analyze_trend_from_measurements(measurements_df, gender, metric='height')
    weight_trend = analyze_trend_from_measurements(measurements_df, gender, metric='weight')
    
    return {
        'height': height_trend,
        'weight': weight_trend
    }

def generate_test_data(scenario):
    """Generate test data for scenarios"""
    try:
        # Setup test subject
        display_name = scenario.replace('_', ' ').capitalize()
        name = f"Test Subject - {display_name}"
        gender = "Boys"
        
        # Determine age group
        if "older" in scenario.lower():
            base_year = datetime.now().year - 2
            start_month = 24
        else:
            base_year = datetime.now().year - 1
            start_month = 0
            
        dob = datetime(base_year, 1, 1).strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Get or create child ID
        c.execute("SELECT id FROM children WHERE name = ?", (name,))
        result = c.fetchone()
        
        if result:
            child_id = result[0]
            c.execute("DELETE FROM measurements WHERE child_id = ?", (child_id,))
        else:
            c.execute("INSERT INTO children (name, gender, dob) VALUES (?, ?, ?)", 
                     (name, gender, dob))
            child_id = c.lastrowid
        
        conn.commit()
        
        # Generate 12 measurements
        for month_offset in range(1, 13):
            current_age = start_month + month_offset
            meas_date_dt = datetime.strptime(dob, '%Y-%m-%d') + pd.DateOffset(months=current_age)
            meas_date = meas_date_dt.strftime('%Y-%m-%d')
            
            # Get WHO medians
            hfa_data, _, hfa_x = get_who_standards(gender, 'lhfa', current_age)
            wfa_data, _, wfa_x = get_who_standards(gender, 'wfa', current_age)
            hcfa_data, _, hcfa_x = get_who_standards(gender, 'hcfa', current_age)
            
            if hfa_data is None or hfa_data.empty or wfa_data is None or wfa_data.empty:
                continue
            
            hfa_data['diff'] = (hfa_data[hfa_x] - current_age).abs()
            wfa_data['diff'] = (wfa_data[wfa_x] - current_age).abs()
            
            median_height = hfa_data.loc[hfa_data['diff'].idxmin()]['SD0']
            median_weight = wfa_data.loc[wfa_data['diff'].idxmin()]['SD0']

            if hcfa_data is not None and not hcfa_data.empty:
                hcfa_data['diff'] = (hcfa_data[hcfa_x] - current_age).abs()
                median_head = hcfa_data.loc[hcfa_data['diff'].idxmin()]['SD0']
            else:
                median_head = 45.0  # Default fallback value
            
            # Apply scenario
            is_dropping = "dropping" in scenario.lower()
            
            if is_dropping and month_offset > 6:
                drop_pct = 0.08 * ((month_offset - 6) / 6)
                final_h = median_height * (1 - drop_pct)
                final_w = median_weight * (1 - drop_pct)
                final_head = median_head * (1 - drop_pct) 
            else:
                final_h = median_height
                final_w = median_weight
                final_head = median_head
            
            # Insert measurement
            age_m, age_w = calculate_age_in_months(dob, meas_date)
            bmi = calculate_bmi(final_w, final_h)
            
            c.execute("""
                INSERT INTO measurements 
                (child_id, date, dob, age_months, age_weeks, weight, height, head_circumference, bmi) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (child_id, meas_date, dob, age_m, age_w, final_w, final_h, final_head, bmi))
        
        conn.commit()
        conn.close()
        
        # Clear cache
        get_children.clear()
        get_child_measurements.clear()
        
        st.toast(f'Test data generated for {name}', icon="🧪")
        return True
        
    except Exception as e:
        st.error(f"Error generating test data: {str(e)}")
        return False

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

                # HEALTH & TREND OVERVIEW
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
                health_status = calculate_health_status(child_measures, gender)
                
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
                    
                    # Get trend analysis
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

            # CHART TABS
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
                    
                    if df_who is not None and not df_who.empty:
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
                    
                    if df_who is not None and not df_who.empty:
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
                    
                    if df_who is not None and not df_who.empty:
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

                    if df_who is not None and not df_who.empty:
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

                    if df_who is not None and not df_who.empty:
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
