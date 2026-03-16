import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import sqlite3
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Child Growth Tracker", layout="wide")

# Configuration
BASE_PATH = "./who_expanded_metrics"
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

# def calculate_age_in_months(birth_date, measurement_date):
#     """Calculate age in months and weeks"""
#     try:
#         b_ts = pd.Timestamp(birth_date)
#         m_ts = pd.Timestamp(measurement_date)
        
#         if pd.isna(b_ts) or pd.isna(m_ts): 
#             return 0.0, 0.0
            
#         days_diff = (m_ts - b_ts).days
#         months = round(days_diff / 30.4375, 2)
#         weeks = round(days_diff / 7.0, 2)
#         return months, weeks
#     except Exception as e:
#         return 0.0, 0.0

def calculate_age_exact(birth_date, measurement_date):
    """
    Calculate age in completed months and completed weeks
    using calendar logic (WHO-style).
    """
    try:
        b_ts = pd.to_datetime(birth_date, errors="coerce")
        r_ts = pd.to_datetime(measurement_date, errors="coerce") if measurement_date else pd.Timestamp.today()

        if pd.isna(b_ts) or pd.isna(r_ts):
            return None, None

        if r_ts < b_ts:
            return None, None

        delta = relativedelta(r_ts, b_ts)

        # Completed calendar months
        completed_months = delta.years * 12 + delta.months

        # Completed weeks (floor)
        total_days = (r_ts - b_ts).days
        completed_weeks = total_days // 7

        return completed_months, completed_weeks

    except Exception as e:
        print(f"Age calculation error: {e}")
        return None, None

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
    """
    Load WHO growth standards.
    Robust version: Handles 'table' vs 'tables' and various column headers.
    """
    gender_code = "boys" if gender.lower() == "boys" else "girls"
    
    # Map app metric types to the new filename prefixes
    prefix_map = {
        'bmi': 'bfa',
        'lhfa': 'lhfa',
        'wfa': 'wfa',
        'hcfa': 'hcfa',
    }

    if metric_type == 'wfl':
        if age_in_months > 24:
            prefix = 'wfh'
        else:
            prefix = 'wfl'
    else:
        if metric_type not in prefix_map:
            return None, None, None
        prefix = prefix_map[metric_type]

    # Construct filename base
    # We will try multiple variations to handle the user's file naming
    base_filename = f"{prefix}-{gender_code}-zscore-expanded-table"
    
    # Attempt to find a valid file
    possible_extensions = ['.xlsx', '.xlxs'] # handle common typos
    possible_suffixes = ['s', ''] # 'tables' or 'table'
    
    df = None
    full_path = ""
    
    for ext in possible_extensions:
        for suffix in possible_suffixes:
            test_path = os.path.join(BASE_PATH, f"{base_filename}{suffix}{ext}")
            if os.path.exists(test_path):
                full_path = test_path
                break
        if full_path:
            break
            
    # Fallback: Try strict naming if loop failed (e.g. strict .xlsx with tables)
    if not full_path:
        full_path = os.path.join(BASE_PATH, f"{base_filename}s.xlsx")

    try:
        df = pd.read_excel(full_path, engine='openpyxl')
        df.columns = [c.strip() for c in df.columns]

        # Expanded list of possible column names
        possible_names = [
            'Day', 'Month', 'Week', 
            'Length', 'Height', 
            'Length cm', 'Height cm', 'Length (cm)', 'Height (cm)'
        ]
        
        x_col_name = next((n for n in possible_names if n in df.columns), None)

        if x_col_name is None:
            # DEBUG: Print columns to console so we can see what's in the file
            print(f"⚠️ X-axis column not found in {os.path.basename(full_path)}")
            print(f"   Available columns: {list(df.columns)}")
            return None, None, None

        axis_type = x_col_name 
        return df, axis_type, x_col_name

    except FileNotFoundError:
        print(f"⚠️ File not found: {full_path}")
        return None, None, None
    except Exception as e:
        print(f"⚠️ Error loading WHO standards: {e} | path={full_path}")
        return None, None, None

def _who_x_value(axis_type, age_months, age_weeks, dob=None, meas_date=None):
    """
    Return the correct x-axis value based on column type.
    Added support for 'Day' column.
    """
    if axis_type == "Day":
        # Need total days difference
        if dob and meas_date:
            b_ts = pd.to_datetime(dob)
            r_ts = pd.to_datetime(meas_date)
            return (r_ts - b_ts).days
        else:
            # Fallback approximation if dates aren't passed
            return age_months * 30.4375
            
    elif axis_type == "Week":
        return age_weeks if age_weeks is not None and not pd.isna(age_weeks) else age_months * 4.34524
        
    # Month, Length, Height
    return age_months

def get_zscore_from_who(gender, metric_type, age_months, value, age_weeks=None, dob=None, meas_date=None):
    """
    Calculate Z-score using WHO SD curves.
    Added dob and meas_date arguments for Day-based calculations.
    """
    df, axis_type, x_col = get_who_standards(gender, metric_type, age_months)
    if df is None or df.empty or x_col is None:
        return None

    # Pass dob and meas_date here
    x_val = _who_x_value(axis_type, age_months, age_weeks, dob, meas_date)

    df = df.copy()
    df['diff'] = (df[x_col] - x_val).abs()
    row = df.loc[df['diff'].idxmin()]

    m = row.get('SD0')
    sd2 = row.get('SD2')
    sd2neg = row.get('SD2neg')

    if pd.isna(m) or pd.isna(sd2) or pd.isna(sd2neg):
        return None

    if value >= m:
        sd = (sd2 - m) / 2.0
    else:
        sd = (m - sd2neg) / 2.0

    if sd == 0 or pd.isna(sd):
        return None

    return round((value - m) / sd, 2)

def analyze_trend_from_measurements(measurements, gender):
    """
    Analyze growth trend based on specific clinical events:
    1. Flat Growth (No height gain)
    2. Rapid Weight Gain (Z-score increase >= 2)
    3. Growth Faltering (Z-score drop >= 1)
    4. Normal Growth
    """
    if measurements is None or len(measurements) < 2:
        return "Not enough data"

    # Ensure data is sorted by date
    measurements = measurements.sort_values('date')

    # -------------------------------------------------------------------
    # 1. CHECK FLAT GROWTH CURVE (Raw Height Check)
    # Criteria: No height gain within an age-specific window (e.g., 3-6 months)
    # Implementation: Check last 4 months of data. If max height < min height + 0.5cm
    # -------------------------------------------------------------------
    latest_date = measurements.iloc[-1]['date']
    four_months_ago = latest_date - relativedelta(months=4)
    
    recent_heights = measurements[measurements['date'] >= four_months_ago]['height']
    
    if len(recent_heights) >= 2:
        # Allow 0.5cm variance for measurement error
        if recent_heights.max() - recent_heights.min() < 0.5:
            return "Flat Growth Curve"

    # -------------------------------------------------------------------
    # 2. CALCULATE Z-SCORE HISTORY (Needed for Delta calculations)
    # -------------------------------------------------------------------
    z_weights = []
    z_heights = []

    for _, row in measurements.iterrows():
        age_m = row['age_months']
        age_w = row.get('age_weeks')
        
        # Calculate WFA Z-score
        w_z = get_zscore_from_who(
            gender=gender, metric_type='wfa', 
            age_months=age_m, value=row['weight'], age_weeks=age_w,
            dob=row.get('dob'),       
            meas_date=row.get('date') 
        )
        z_weights.append(w_z)

        # Calculate LHFA Z-score
        h_z = get_zscore_from_who(
            gender=gender, metric_type='lhfa', 
            age_months=age_m, value=row['height'], age_weeks=age_w,
            dob=row.get('dob'),       
            meas_date=row.get('date') 
        )
        z_heights.append(h_z)

    # Filter out None values for accurate delta calc
    z_weights = [z for z in z_weights if z is not None]
    z_heights = [z for z in z_heights if z is not None]

    if not z_weights:
        return "Not enough data"

    first_z_w = z_weights[0]
    last_z_w = z_weights[-1]
    
    # -------------------------------------------------------------------
    # 3. CHECK RAPID WEIGHT GAIN
    # Criteria: Crossing >= 2 centile bands upward (Z-score increase >= 2)
    # -------------------------------------------------------------------
    if (last_z_w - first_z_w) >= 2.0:
        return "Rapid Weight Gain"

    # -------------------------------------------------------------------
    # 4. CHECK GROWTH FALTERING
    # Criteria: Drop of >= 1 Z-score in weight OR height over time
    # -------------------------------------------------------------------
    faltering = False
    
    # Check Weight Faltering
    if (last_z_w - first_z_w) <= -1.0:
        faltering = True
        
    # Check Height Faltering
    if z_heights:
        first_z_h = z_heights[0]
        last_z_h = z_heights[-1]
        if (last_z_h - first_z_h) <= -1.0:
            faltering = True

    if faltering:
        return "Growth Faltering"

    # -------------------------------------------------------------------
    # 5. NORMAL GROWTH
    # Criteria: Stable Z-scores, no faltering, no rapid gain
    # -------------------------------------------------------------------
    return "Normal Growth"

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
        
        get_children.clear()
        
        st.session_state.notification = f'{name} registered successfully!'
        return True
    except Exception as e:
        st.error(f"Error saving child: {str(e)}")
        return False

def save_measurement(child_id, dob, meas_date, weight, height, head_circumference):
    """Create new measurement"""
    try:
        age_m, age_w = calculate_age_exact(dob, meas_date)
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
        
        get_child_measurements.clear()
        
        st.session_state.notification = 'Measurement added successfully'
        return True
    except Exception as e:
        st.error(f"Error saving measurement: {str(e)}")
        return False

def update_child_profile(child_id, new_data, old_data):
    """
    Updates the child's profile in the DB and syncs measurements 
    if DOB or Gender changed.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        
        new_dob = pd.to_datetime(new_data['dob']).date()
        old_dob = pd.to_datetime(old_data['dob']).date()
        
        if new_dob != old_dob:
            measurements = pd.read_sql(
                "SELECT id, date FROM measurements WHERE child_id = ?", 
                conn, params=(child_id,)
            )
            for _, m_row in measurements.iterrows():
                age_m, age_w = calculate_age_exact(new_dob, m_row['date'])
                conn.execute("""
                    UPDATE measurements 
                    SET dob = ?, age_months = ?, age_weeks = ? 
                    WHERE id = ?
                """, (new_dob.strftime('%Y-%m-%d'), age_m, age_w, m_row['id']))

        conn.execute("""
            UPDATE children 
            SET name = ?, gender = ?, dob = ? 
            WHERE id = ?
        """, (new_data['name'], new_data['gender'], new_dob.strftime('%Y-%m-%d'), child_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating child {child_id}: {e}")
        return False

def delete_child(child_id):
    """Delete child and all measurements"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute("DELETE FROM measurements WHERE child_id = ?", (child_id,))
        c.execute("DELETE FROM children WHERE id = ?", (child_id,))
        
        conn.commit()
        conn.close()
        
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
        
        c.execute("DELETE FROM measurements WHERE child_id = ?", (child_id,))
        
        for _, row in measurements_df.iterrows():
            age_m, age_w = calculate_age_exact(row['dob'], row['date'])
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
        
        get_child_measurements.clear()
        
        st.session_state.notification = 'Measurements updated successfully'
        return True
    except Exception as e:
        st.error(f"Error updating measurements: {str(e)}")
        return False

def _get_sd_value(row, col):
    """Safely retrieve an SD column value from a WHO reference row."""
    val = row.get(col)
    return val if val is not None and not pd.isna(val) else None


def calculate_health_status(measurements_df, gender):
    """
    Classify nutritional status using Permenkes + WHO z-score thresholds
    for children 0–60 months.

    Uses get_zscore_from_who() for WFA, WFL/WFH, and BMI-for-age so every
    classification is based on the same z-score arithmetic as the trend engine.
    LHFA still compares raw height against SD boundaries (WHO standard practice).

    Categories per index
    --------------------
    WFA (Weight-for-Age):
        < −3 SD  → Severely Underweight
        −3 to < −2 SD → Underweight
        −2 to +1 SD   → Normal Weight
        > +1 SD        → Risk of Overweight  (Permenkes)
        > +2 SD        → Overweight          (WHO)
        > +3 SD        → Obese

    LFA/HFA (Length/Height-for-Age):
        < −3 SD  → Severely Stunted
        −3 to < −2 SD → Stunted
        −2 to +3 SD   → Normal Stature
        > +2 SD        → Tall (WHO)
        > +3 SD        → Very Tall (WHO) / Tall (Permenkes)

    WFL/WFH (Weight-for-Length/Height):
        < −3 SD         → Severely Wasted
        −3 to < −2 SD   → Wasted
        −2 to +1 SD     → Normal
        > +1 to +2 SD   → Possible Risk of Overweight
        > +2 to +3 SD   → Overweight
        > +3 SD         → Obese

    BMI-for-Age:
        < −3 SD         → Severely Wasted
        −3 to < −2 SD   → Wasted
        −2 to +1 SD     → Normal
        > +1 to +2 SD   → Possible Risk of Overweight
        > +2 to +3 SD   → Overweight
        > +3 SD         → Obese
    """
    if measurements_df.empty:
        return {}

    # Always sort before picking latest row
    measurements_df = measurements_df.sort_values('date')
    latest = measurements_df.iloc[-1]

    latest_age_m = latest.get('age_months')
    latest_age_w = latest.get('age_weeks')
    latest_weight = latest.get('weight')
    latest_height = latest.get('height')
    latest_bmi = latest.get('bmi') or calculate_bmi(latest_weight, latest_height)

    if latest_age_m is None or pd.isna(latest_age_m):
        return {}

    statuses = {}

    # -----------------------------------------------------------------------
    # 1. LENGTH/HEIGHT-FOR-AGE  (LFA/HFA) — compare raw height vs SD lines
    #    Permenkes + WHO: 0–60 months
    # -----------------------------------------------------------------------
    lhfa_data, lhfa_axis, lhfa_x = get_who_standards(gender, 'lhfa', latest_age_m)
    if (lhfa_data is not None and not lhfa_data.empty and lhfa_x is not None
            and latest_height is not None and not pd.isna(latest_height)):

        x_val = _who_x_value(lhfa_axis, latest_age_m, latest_age_w)
        lhfa_data = lhfa_data.copy()
        lhfa_data['diff'] = (lhfa_data[lhfa_x] - x_val).abs()
        row = lhfa_data.loc[lhfa_data['diff'].idxmin()]

        sd3neg = _get_sd_value(row, 'SD3neg')
        sd2neg = _get_sd_value(row, 'SD2neg')
        sd2    = _get_sd_value(row, 'SD2')
        sd3    = _get_sd_value(row, 'SD3')

        if sd3neg is not None and latest_height < sd3neg:
            statuses['growth'] = 'Severely Stunted'
        elif sd2neg is not None and latest_height < sd2neg:
            statuses['growth'] = 'Stunted'
        elif sd3 is not None and latest_height > sd3:
            statuses['growth'] = 'Very Tall'        # > +3 SD (WHO/Permenkes)
        elif sd2 is not None and latest_height > sd2:
            statuses['growth'] = 'Tall'             # > +2 SD (WHO)
        else:
            statuses['growth'] = 'Normal Stature'   # −2 to +2 SD

    # -----------------------------------------------------------------------
    # 2. WEIGHT-FOR-AGE  (WFA) — classify via z-score
    #    Permenkes + WHO: 0–60 months
    #    Normal band: −2 SD to +1 SD
    # -----------------------------------------------------------------------
    if latest_weight is not None and not pd.isna(latest_weight):
        wfa_z = get_zscore_from_who(
            gender=gender,
            metric_type='wfa',
            age_months=latest_age_m,
            value=latest_weight,
            age_weeks=latest_age_w
        )
        if wfa_z is not None:
            if wfa_z < -3:
                statuses['wfa'] = 'Severely Underweight'
            elif wfa_z < -2:
                statuses['wfa'] = 'Underweight'
            elif wfa_z <= 1:
                statuses['wfa'] = 'Normal Weight'
            elif wfa_z <= 2:
                statuses['wfa'] = 'Risk of Overweight'  # > +1 SD (Permenkes)
            elif wfa_z <= 3:
                statuses['wfa'] = 'Overweight'           # > +2 SD (WHO)
            else:
                statuses['wfa'] = 'Obese'                # > +3 SD

    # -----------------------------------------------------------------------
    # 3. WEIGHT-FOR-LENGTH/HEIGHT  (WFL/WFH) — classify via z-score
    #    Permenkes + WHO: 0–60 months
    #    x-axis = child height, NOT age
    # -----------------------------------------------------------------------
    if (latest_weight is not None and not pd.isna(latest_weight)
            and latest_height is not None and not pd.isna(latest_height)):

        # For WFL/WFH the "age_months" argument only controls which file is chosen
        # (wfl 0-2y vs wfh 2-5y). The actual z-score lookup uses height as x-axis.
        wfl_data, wfl_axis, wfl_x = get_who_standards(gender, 'wfl', latest_age_m)
        if wfl_data is not None and not wfl_data.empty and wfl_x is not None:
            wfl_data = wfl_data.copy()
            wfl_data['diff'] = (wfl_data[wfl_x] - latest_height).abs()
            row = wfl_data.loc[wfl_data['diff'].idxmin()]

            m      = _get_sd_value(row, 'SD0')
            sd1    = _get_sd_value(row, 'SD1')
            sd2    = _get_sd_value(row, 'SD2')
            sd3    = _get_sd_value(row, 'SD3')
            sd2neg = _get_sd_value(row, 'SD2neg')
            sd3neg = _get_sd_value(row, 'SD3neg')

            if m is not None and sd2 is not None and sd2neg is not None:
                # Compute z-score using asymmetric SD approach
                if latest_weight >= m:
                    sd_unit = (sd2 - m) / 2.0 if sd2 != m else None
                else:
                    sd_unit = (m - sd2neg) / 2.0 if sd2neg != m else None

                if sd_unit and sd_unit != 0:
                    wfl_z = (latest_weight - m) / sd_unit

                    if wfl_z < -3:
                        statuses['wfl'] = 'Severely Wasted'
                    elif wfl_z < -2:
                        statuses['wfl'] = 'Wasted'
                    elif wfl_z <= 1:
                        statuses['wfl'] = 'Normal'
                    elif wfl_z <= 2:
                        statuses['wfl'] = 'Possible Risk of Overweight'
                    elif wfl_z <= 3:
                        statuses['wfl'] = 'Overweight'
                    else:
                        statuses['wfl'] = 'Obese'

    # -----------------------------------------------------------------------
    # 4. BMI-FOR-AGE — classify via z-score
    #    Permenkes + WHO: 0–60 months
    # -----------------------------------------------------------------------
    if latest_bmi is not None and not pd.isna(latest_bmi):
        bmi_z = get_zscore_from_who(
            gender=gender,
            metric_type='bmi',
            age_months=latest_age_m,
            value=latest_bmi,
            age_weeks=latest_age_w
        )
        
        if bmi_z is not None:
            # --- DIFFERENTIATE AGE GROUPS ---
            if latest_age_m <= 60:
                # --- 0-60 Months (Child) - Permenkes + WHO ---
                if bmi_z < -3:
                    statuses['bmi'] = 'Severely Wasted'
                elif bmi_z < -2:
                    statuses['bmi'] = 'Wasted'
                elif bmi_z <= 1:
                    statuses['bmi'] = 'Normal'
                elif bmi_z <= 2:
                    statuses['bmi'] = 'Possible Risk of Overweight'
                elif bmi_z <= 3:
                    statuses['bmi'] = 'Overweight'
                else:
                    statuses['bmi'] = 'Obese'
            
            elif latest_age_m > 60:
                # --- 5-18 Years (Adolescent) - WHO 2007 / IMT/U ---
                # Note: 5-18 years standards typically go up to +2 or +3 SD.
                if bmi_z < -3:
                    statuses['bmi'] = 'Severely Thin'
                elif bmi_z < -2:
                    statuses['bmi'] = 'Thinness'
                elif bmi_z <= 1:
                    statuses['bmi'] = 'Normal'
                elif bmi_z <= 2:
                    statuses['bmi'] = 'Overweight'
                else:
                    statuses['bmi'] = 'Obesity'
    
    # -----------------------------------------------------------------------
    # 5. HEAD CIRCUMFERENCE-FOR-AGE (HCFA)
    # -----------------------------------------------------------------------
    hcfa_value = latest.get('head_circumference')
    if hcfa_value is not None and not pd.isna(hcfa_value):
        hcfa_z = get_zscore_from_who(
            gender=gender,
            metric_type='hcfa',
            age_months=latest_age_m,
            value=hcfa_value,
            age_weeks=latest_age_w,
            dob=latest.get('dob'),
            meas_date=latest.get('date')
        )
        
        if hcfa_z is not None:
            # Logic based on provided guidelines (with gap filling for robustness)
            if hcfa_z < -3:
                statuses['hcfa'] = 'Small Head Circumference'
            elif hcfa_z <= -2:
                # Corrected range to -3 to -2
                statuses['hcfa'] = 'Mildly Small Head Circumference'
            elif hcfa_z <= 2:
                # Covers -2 to +2 (incorporating the specific -1 to +1 guideline)
                statuses['hcfa'] = 'Normal Head Circumference'
            elif hcfa_z <= 3:
                statuses['hcfa'] = 'Larger Head Circumference'
            else:
                statuses['hcfa'] = 'Severely Large Head Circumference'

    return statuses

def get_trend_analysis(measurements_df, gender):
    """
    Calculate trend analysis.
    Since the new logic considers both height and weight to determine a clinical status,
    we run the analysis once and apply the result to both metrics.
    """
    if measurements_df.empty or len(measurements_df) < 2:
        return {
            'height': 'Not enough data',
            'weight': 'Not enough data'
        }
    
    # Call the unified analysis function
    # Note: We no longer pass 'metric' because the function checks both internally
    overall_trend = analyze_trend_from_measurements(measurements_df, gender)
    
    # Return the same status for both, as the new logic is holistic.
    # e.g., if Height drops, it flags "Growth Faltering" for the whole child.
    return {
        'height': overall_trend,
        'weight': overall_trend
    }

def generate_test_data(scenario):
    """Generate test data for scenarios"""
    try:
        display_name = scenario.replace('_', ' ').capitalize()
        name = f"Test Subject - {display_name}"
        gender = "Boys"
        
        if "older" in scenario.lower():
            base_year = datetime.now().year - 2
            start_month = 24
        else:
            base_year = datetime.now().year - 1
            start_month = 0
            
        dob = datetime(base_year, 1, 1).strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
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
        
        for month_offset in range(1, 13):
            current_age = start_month + month_offset
            meas_date_dt = datetime.strptime(dob, '%Y-%m-%d') + pd.DateOffset(months=current_age)
            meas_date = meas_date_dt.strftime('%Y-%m-%d')

            age_m, age_w = calculate_age_exact(dob, meas_date)
            
            hfa_data, hfa_axis, hfa_x = get_who_standards(gender, 'lhfa', age_m)
            wfa_data, wfa_axis, wfa_x = get_who_standards(gender, 'wfa', age_m)
            hcfa_data, hcfa_axis, hcfa_x = get_who_standards(gender, 'hcfa', age_m)
            
            if hfa_data is None or hfa_data.empty or wfa_data is None or wfa_data.empty:
                continue

            hfa_xval = _who_x_value(hfa_axis, age_m, age_w)
            wfa_xval = _who_x_value(wfa_axis, age_m, age_w)

            hfa_data = hfa_data.copy()
            wfa_data = wfa_data.copy()

            hfa_data['diff'] = (hfa_data[hfa_x] - hfa_xval).abs()
            wfa_data['diff'] = (wfa_data[wfa_x] - wfa_xval).abs()
            
            median_height = hfa_data.loc[hfa_data['diff'].idxmin()]['SD0']
            median_weight = wfa_data.loc[wfa_data['diff'].idxmin()]['SD0']

            if hcfa_data is not None and not hcfa_data.empty and hcfa_x is not None:
                hcfa_xval = _who_x_value(hcfa_axis, age_m, age_w)
                hcfa_data = hcfa_data.copy()
                hcfa_data['diff'] = (hcfa_data[hcfa_x] - hcfa_xval).abs()
                median_head = hcfa_data.loc[hcfa_data['diff'].idxmin()]['SD0']
            else:
                median_head = 45.0
            
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
        
            bmi = calculate_bmi(final_w, final_h)
            
            c.execute("""
                INSERT INTO measurements 
                (child_id, date, dob, age_months, age_weeks, weight, height, head_circumference, bmi) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (child_id, meas_date, dob, age_m, age_w, final_w, final_h, final_head, bmi))
        
        conn.commit()
        conn.close()
        
        get_children.clear()
        get_child_measurements.clear()
        
        st.toast(f'Test data generated for {name}', icon="🧪")
        return True
        
    except Exception as e:
        st.error(f"Error generating test data: {str(e)}")
        return False

# ==================== CHART HELPER FUNCTIONS ====================

def _build_who_reference_for_chart(gender, metric_type, max_age_m):
    """
    Build WHO reference DataFrame for charting.
    Simplified: Uses the 'expanded' files which contain all necessary data ranges.
    """
    
    # Determine which file to load (handles WFL vs WFH switch)
    # Pass max_age_m to help decide wfl vs wfh
    df_who, axis_type, x_col = get_who_standards(gender, metric_type, max_age_m)
    
    if df_who is None or df_who.empty:
        return None, None, None, None, None

    # Determine label for user
    if axis_type == "Day":
        x_label = "Age (Days)"
    elif axis_type == "Week":
        x_label = "Age (Weeks)"
    elif axis_type in ["Length", "Height"]:
        x_label = f"{axis_type} (cm)"
    else:
        x_label = "Age (Months)"

    # Determine what key from user data maps to the X-axis
    if axis_type in ["Length", "Height"]:
        user_x_key = None # Special handling: use child's height
    elif axis_type == "Day":
        user_x_key = "date" # We will calculate diff from date
    elif axis_type == "Week":
        user_x_key = "age_weeks"
    else:
        user_x_key = "age_months"

    return df_who, x_col, x_label, axis_type, user_x_key


def _add_who_reference_traces(fig, df_who, x_col, z_styles):
    """Add WHO reference SD lines to a plotly figure."""
    for col, style in z_styles.items():
        if col in df_who.columns:
            fig.add_trace(go.Scatter(
                x=df_who[x_col],
                y=df_who[col],
                line=dict(color=style['color'], dash=style['dash'], width=1.5),
                name=style['name'],
                mode='lines'
            ))


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
        
        # --- TEST DATA GENERATOR ---
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
            st.subheader("Children Profiles")
            st.caption("Double-click cells to edit. Click the trash icon on the right to delete.")

            edited_children = st.data_editor(
                children_df,
                column_config={
                    "name": st.column_config.TextColumn("Name", required=True),
                    "gender": st.column_config.SelectboxColumn(
                        "Gender", 
                        options=["Boys", "Girls"], 
                        required=True
                    ),
                    "dob": st.column_config.DateColumn("Date of Birth", required=True),
                },
                num_rows="dynamic",
                width='stretch',
                key="children_profile_editor"
            )

            if st.button("💾 Save Profile Changes", type="primary"):
                changes_made = False

                # --- Detect deletions: rows present in original but missing in edited ---
                deleted_ids = set(children_df.index) - set(edited_children.index)
                for child_id in deleted_ids:
                    if delete_child(child_id):
                        changes_made = True
                
                for child_id, new_row in edited_children.iterrows():
                    if child_id in children_df.index:
                        success = update_child_profile(
                            child_id, 
                            new_row, 
                            children_df.loc[child_id]
                        )
                        if success:
                            changes_made = True

                if changes_made:
                    get_children.clear()
                    get_child_measurements.clear()
                    st.toast("Profiles updated and measurements synced!", icon="🔄")
                    st.rerun()

            st.divider()
            
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
                # FIX #1: Sort once here; use max_age consistently throughout
                child_measures = child_measures.sort_values(by='date')
                gender = child_info['gender']

                # FIX #1: Use max_age (not latest_age) as the single source of truth
                max_age = float(child_measures['age_months'].max())

                # ---- HEALTH & TREND OVERVIEW ----
                # Permenkes + WHO category descriptions (0–60 months)
                status_meanings = {
                    # LFA/HFA
                    "Severely Stunted":           "< −3 SD. Severe growth impairment. Needs immediate assessment.",
                    "Stunted":                    "−3 to < −2 SD. Growth stunted vs age. Review nutrition & health.",
                    "Normal Stature":             "−2 to +2 SD. Height within healthy range.",
                    "Tall":                       "> +2 SD. Above-average height (WHO). Usually not a concern.",
                    "Very Tall":                  "> +3 SD. Significantly above average. Check if endocrine cause.",
                    # WFA
                    "Severely Underweight":       "< −3 SD. Critical. Immediate medical attention needed.",
                    "Underweight":                "−3 to < −2 SD. Below healthy weight for age.",
                    "Normal Weight":              "−2 to +1 SD. Healthy weight for age.",
                    "Risk of Overweight":         "> +1 SD. Monitor diet and activity (Permenkes).",
                    "Overweight":                 "> +2 SD. Above healthy weight. Review diet.",
                    "Obese":                      "> +3 SD. Significantly above healthy weight.",
                    # WFL/WFH 
                    "Severely Wasted":            "< −3 SD. Critical malnutrition. Needs immediate doctor.",
                    "Wasted":                     "−3 to < −2 SD. Malnourished. Needs diet improvement.",
                    "Normal":                     "−2 to +1 SD. Healthy nutritional status.",
                    "Possible Risk of Overweight":"≥ +1 to +2 SD. Monitor closely (Permenkes + WHO).",
                    # BMI
                    # 0-60 Months
                    "Severely Wasted":            "< −3 SD. Critical malnutrition. Needs immediate doctor.",
                    "Wasted":                     "−3 to < −2 SD. Malnourished. Needs diet improvement.",
                    "Normal":                     "−2 to +1 SD. Healthy nutritional status.",
                    "Possible Risk of Overweight":"≥ +1 to +2 SD. Monitor closely (Permenkes + WHO).",
                    
                    # 5-18 Years (New)
                    "Severely Thin":              "< −3 SD. Severely thin. Requires medical assessment.",
                    "Thinness":                   "−3 to < −2 SD. Thin. Review nutritional intake.",
                    "Overweight":                 "> +1 to +2 SD. Overweight. Increase physical activity.",
                    "Obesity":                    "> +2 SD. Obesity. High risk of metabolic diseases.",
                    
                    # HCFA
                    "Small Head Circumference":     "< -3 SD. Severely small head size. Consult pediatrician.",
                    "Mildly Small Head Circumference": "-3 to -2 SD. Below average size. Monitor growth.",
                    "Normal Head Circumference":    "-2 to +2 SD. Head size is within healthy range.",
                    "Larger Head Circumference":    "+2 to +3 SD. Above average size. Usually benign, but monitor.",
                    "Severely Large Head Circumference": "> +3 SD. Significantly large. Medical assessment recommended.",

                    # Shared
                    "Overweight":                 "Above healthy weight. Review diet.",
                    "Obese":                      "Significantly above healthy weight.",
                    "No Data":                    "Insufficient data to analyze.",
                }

                latest = child_measures.iloc[-1]
                latest_weight = latest['weight']
                latest_height = latest['height']
                latest_bmi = latest.get('bmi') or calculate_bmi(latest_weight, latest_height)

                # Calculate status if not already done above
                if 'health_status' not in locals():
                    health_status = calculate_health_status(child_measures, gender)

                st.subheader("📊 Health & Trend Overview")

                status_col, trend_col = st.columns([2, 1])

                with status_col:
                    st.markdown("### 🏥 Current Status")

                    # Row 1: Growth, Weight, Head Circumference (3 Cols)
                    g_col, w_col, bmi_col = st.columns(3)
                    
                    with g_col:
                        growth_status = health_status.get('growth', 'No Data')
                        st.metric("Length/Height", growth_status)
                        st.caption(status_meanings.get(growth_status, ""))
                        
                    with w_col:
                        wfa_status = health_status.get('wfa', 'No Data')
                        st.metric("Weight", wfa_status)
                        st.caption(status_meanings.get(wfa_status, ""))

                    with bmi_col:
                        if latest_bmi:
                            bmi_status = health_status.get('bmi', 'No Data')
                            st.metric("BMI", f"{bmi_status}")
                            st.caption(status_meanings.get(bmi_status, ""))   
                    

                    # Row 2: WFL/WFH  |  BMI-for-Age (2 Cols)
                    wfl_col, h_col = st.columns([1, 2])
                    
                    with wfl_col:
                        wfl_status = health_status.get('wfl', 'No Data')
                        st.metric("Weight-for-Length/Height", wfl_status)
                        st.caption(status_meanings.get(wfl_status, ""))
                    
                    with h_col:
                        hcfa_status = health_status.get('hcfa', 'No Data')
                        st.metric("Head Circumference", hcfa_status)
                        st.caption(status_meanings.get(hcfa_status, ""))
                        
                    

                    st.caption(f"*Based on: {latest['date'].date()}*")

                with trend_col:
                    st.markdown("### 📈 Trend Analysis")
                    
                    trend_meanings = {
                        "Flat Growth Curve": "No height gain detected in the last 4 months. Could indicate nutritional deficiency or chronic health issues.",
                        "Rapid Weight Gain": "Weight has crossed upward by ≥2 centile bands. Monitor for overfeeding or sedentary lifestyle risks.",
                        "Growth Faltering": "Z-score has dropped by ≥1. This indicates growth slowing down, possibly due to malnutrition or illness.",
                        "Normal Growth": "Stable Z-scores with a consistent growth curve. Child is growing as expected.",
                        "Not enough data": "Need at least 2 records to analyze trend.",
                    }
                    
                    trend_analysis = get_trend_analysis(child_measures, gender)
                    
                    hfa_trend = trend_analysis.get('height', 'Not enough data')
                    st.write(f"**Height:** {hfa_trend}")
                    st.caption(trend_meanings.get(hfa_trend, ""))

                    # FIX #3: Show weight trend for ALL ages, not just > 24 months
                    # WHO WFA is valid 0–5 years. No clinical reason to hide it under 24 months.
                    wfa_trend = trend_analysis.get('weight', 'Not enough data')
                    st.write(f"**Weight:** {wfa_trend}")
                    st.caption(trend_meanings.get(wfa_trend, ""))

                st.divider()

            # ---- CHART TABS ----
            if child_measures.empty:
                st.info("No measurements recorded for this child yet.")
            else:
                # FIX #7: Add SD1 to z_styles so it's drawn on BMI chart
                z_styles = {
                    'SD3neg': {'color': "#060606", 'dash': 'dot',   'name': '-3 SD'},
                    'SD2neg': {'color': '#ff4b4b', 'dash': 'dash',  'name': '-2 SD'},
                    'SD0':    {'color': '#00c04b', 'dash': 'solid', 'name': 'Median'},
                    'SD1':    {'color': '#ffa500', 'dash': 'dot',   'name': '+1 SD'},
                    'SD2':    {'color': '#ff4b4b', 'dash': 'dash',  'name': '+2 SD'},
                    'SD3':    {'color': "#060606", 'dash': 'dot',   'name': '+3 SD'},
                }

                # SD1 should only appear on the BMI chart; use this subset for all others
                z_styles_no_sd1 = {k: v for k, v in z_styles.items() if k != 'SD1'}

                chart_tab1, chart_tab2, chart_tab3, chart_tab4, chart_tab5 = st.tabs([
                    "Weight-for-Age", 
                    "Length/Height-for-Age", 
                    "Weight-for-Length/Height", 
                    "Head-Circumference-for-Age", 
                    "BMI-for-Age"
                ])

                # --- WEIGHT CHART ---
                with chart_tab1:
                    # FIX #1: Use max_age consistently
                    df_who, x_col, x_label, axis_type, user_x_key = _build_who_reference_for_chart(
                        gender, 'wfa', max_age
                    )
                    
                    if df_who is not None and not df_who.empty:
                        fig = go.Figure()
                        _add_who_reference_traces(fig, df_who, x_col, z_styles_no_sd1)

                        if user_x_key == "date":
                            # Calculate days since birth for the X-axis
                            if 'dob' in child_measures.columns:
                                dob = pd.to_datetime(child_measures['dob'].iloc[0])
                                meas_dates = pd.to_datetime(child_measures['date'])
                                user_x = (meas_dates - dob).dt.days
                            else:
                                user_x = child_measures['age_months'] * 30.44
                        elif user_x_key is None:
                            # For WFL/WFH, X is height
                            user_x = child_measures['height']
                        else:
                            # Standard age_weeks or age_months
                            user_x = child_measures[user_x_key]

                        fig.add_trace(go.Scatter(
                            x=user_x, y=child_measures['weight'],
                            mode='markers+lines', name='Child Data',
                            marker=dict(size=8, color='#4b9fff'), line=dict(width=2)
                        ))
                        fig.update_layout(
                            title=f"Weight-for-Age ({axis_type})",
                            xaxis_title=x_label,
                            yaxis_title="Weight (kg)"
                        )
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning("WHO WFA standards could not be loaded for this age range.")

                # --- HEIGHT CHART ---
                with chart_tab2:
                    # FIX #1 & #6: Use shared helper (same stitching logic, consistent max_age)
                    df_who, x_col, x_label, axis_type, user_x_key = _build_who_reference_for_chart(
                        gender, 'lhfa', max_age
                    )

                    if df_who is not None and not df_who.empty:
                        fig = go.Figure()
                        _add_who_reference_traces(fig, df_who, x_col, z_styles_no_sd1)

                        if user_x_key == "date":
                            # Calculate days since birth for the X-axis
                            if 'dob' in child_measures.columns:
                                dob = pd.to_datetime(child_measures['dob'].iloc[0])
                                meas_dates = pd.to_datetime(child_measures['date'])
                                user_x = (meas_dates - dob).dt.days
                            else:
                                user_x = child_measures['age_months'] * 30.44
                        elif user_x_key is None:
                            # For WFL/WFH, X is height
                            user_x = child_measures['height']
                        else:
                            # Standard age_weeks or age_months
                            user_x = child_measures[user_x_key]
                        fig.add_trace(go.Scatter(
                            x=user_x, y=child_measures['height'],
                            mode='markers+lines', name='Child Data',
                            marker=dict(size=8, color='#4b9fff'), line=dict(width=2)
                        ))
                        fig.update_layout(
                            title=f"Length/Height-for-Age ({axis_type})",
                            xaxis_title=x_label,
                            yaxis_title="Height (cm)"
                        )
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning("WHO LHFA standards could not be loaded for this child/age range.")

                # --- WEIGHT FOR LENGTH/HEIGHT CHART ---
                with chart_tab3:
                    # FIX #2: Use shared helper which stitches WFL+WFH for children > 24 months
                    df_who, x_col, x_label, axis_type, user_x_key = _build_who_reference_for_chart(
                        gender, 'wfl', max_age
                    )

                    if df_who is not None and not df_who.empty:
                        fig = go.Figure()
                        _add_who_reference_traces(fig, df_who, x_col, z_styles_no_sd1)

                        if user_x_key == "date":
                            # Calculate days since birth for the X-axis
                            if 'dob' in child_measures.columns:
                                dob = pd.to_datetime(child_measures['dob'].iloc[0])
                                meas_dates = pd.to_datetime(child_measures['date'])
                                user_x = (meas_dates - dob).dt.days
                            else:
                                user_x = child_measures['age_months'] * 30.44
                        elif user_x_key is None:
                            # For WFL/WFH, X is height
                            user_x = child_measures['height']
                        else:
                            # Standard age_weeks or age_months
                            user_x = child_measures[user_x_key]

                        # x-axis is the child's measured height (cm), not age
                        fig.add_trace(go.Scatter(
                            x=child_measures['height'], y=child_measures['weight'],
                            mode='markers+lines', name='Child Data',
                            marker=dict(size=8, color='#4b9fff'), line=dict(width=2)
                        ))
                        fig.update_layout(
                            title=f"Weight-for-{axis_type}",
                            xaxis_title=x_label,
                            yaxis_title="Weight (kg)"
                        )
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning("WHO WFL/WFH standards could not be loaded for this age range.")

                # --- HEAD CIRCUMFERENCE FOR AGE ---
                with chart_tab4:
                    # FIX #1: Use max_age consistently
                    df_who, x_col, x_label, axis_type, user_x_key = _build_who_reference_for_chart(
                        gender, 'hcfa', max_age
                    )

                    if df_who is not None and not df_who.empty:
                        fig = go.Figure()
                        _add_who_reference_traces(fig, df_who, x_col, z_styles_no_sd1)

                        if user_x_key == "date":
                            # Calculate days since birth for the X-axis
                            if 'dob' in child_measures.columns:
                                dob = pd.to_datetime(child_measures['dob'].iloc[0])
                                meas_dates = pd.to_datetime(child_measures['date'])
                                user_x = (meas_dates - dob).dt.days
                            else:
                                user_x = child_measures['age_months'] * 30.44
                        elif user_x_key is None:
                            # For WFL/WFH, X is height
                            user_x = child_measures['height']
                        else:
                            # Standard age_weeks or age_months
                            user_x = child_measures[user_x_key]
                        fig.add_trace(go.Scatter(
                            x=user_x, y=child_measures['head_circumference'],
                            mode='markers+lines', name='Child Data',
                            marker=dict(size=8, color='#4b9fff'), line=dict(width=2)
                        ))
                        fig.update_layout(
                            title=f"Head-Circumference-for-Age ({axis_type})",
                            xaxis_title=x_label,
                            yaxis_title="Head Circumference (cm)"
                        )
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning("WHO HCFA standards could not be loaded for this age range.")

                # --- BMI FOR AGE ---
                with chart_tab5:
                    # Ensure BMI is calculated
                    if 'bmi' not in child_measures.columns or child_measures['bmi'].isna().all():
                        child_measures['bmi'] = child_measures.apply(
                            lambda row: calculate_bmi(row['weight'], row['height']), axis=1
                        )

                    # FIX #1 & #6: Use shared helper (stitches 0-2y + 2-5y); FIX #7: include SD1
                    df_who, x_col, x_label, axis_type, user_x_key = _build_who_reference_for_chart(
                        gender, 'bmi', max_age
                    )

                    if df_who is not None and not df_who.empty:
                        fig = go.Figure()
                        # FIX #7: Use full z_styles (including SD1) for BMI chart only
                        _add_who_reference_traces(fig, df_who, x_col, z_styles)

                        if user_x_key == "date":
                            # Calculate days since birth for the X-axis
                            if 'dob' in child_measures.columns:
                                dob = pd.to_datetime(child_measures['dob'].iloc[0])
                                meas_dates = pd.to_datetime(child_measures['date'])
                                user_x = (meas_dates - dob).dt.days
                            else:
                                user_x = child_measures['age_months'] * 30.44
                        elif user_x_key is None:
                            # For WFL/WFH, X is height
                            user_x = child_measures['height']
                        else:
                            # Standard age_weeks or age_months
                            user_x = child_measures[user_x_key]
                        fig.add_trace(go.Scatter(
                            x=user_x, y=child_measures['bmi'],
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