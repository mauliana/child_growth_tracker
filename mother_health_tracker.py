import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import random
import sqlite3

# --- CONFIGURATION ---
st.set_page_config(page_title="Maternity app", layout="wide")

# --- DATABASE SETUP ---
DB_FILE = "pregnancy_data.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON")   # 🔥 Enable FK enforcement
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS mother
                (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                name TEXT, 
                dob TEXT, 
                height REAL, 
                pre_weight REAL, 
                last_period_date TEXT,
                arm_circumference REAL,
                twin TEXT,
                pre_bmi REAL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS measurements
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
            mother_id INTEGER, 
            date TEXT,
            gestational_weeks INTEGER,
            weight REAL,
            FOREIGN KEY (mother_id) REFERENCES mother(id) ON DELETE CASCADE)''')

    conn.commit()
    conn.close()

# Run this once at the start of the script
init_db()

# --- HELPER FUNCTIONS ---
def get_mother():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM mother", conn, index_col="id")
    conn.close()
    
    # FIX: Use ISO8601 format for consistency
    df['dob'] = pd.to_datetime(df['dob'], format='ISO8601')
    
    return df

def get_mother_info(mother_id):
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT * FROM mother WHERE id = ?"
    df = pd.read_sql(query, conn, params=(mother_id,))
    conn.close()
    
    if not df.empty:
        # FIX: Use ISO8601 format
        df['dob'] = pd.to_datetime(df['dob'], format='ISO8601')
        return df.iloc[0]
    return None

def get_mother_measurements(mother_id):
    """Get all pregnancy measurements for a mother"""
    try:
        conn = sqlite3.connect(DB_FILE)
        query = """
            SELECT id, mother_id, date, 
                   gestational_weeks, weight 
            FROM measurements 
            WHERE mother_id = ? 
            ORDER BY date
        """
        df = pd.read_sql(query, conn, params=(mother_id,))
        conn.close()
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('id')

        return df

    except Exception as e:
        st.error(f"Error loading measurements: {str(e)}")
        return pd.DataFrame()

def save_mother(name, dob, height, pre_weight, last_period, arm, twin):
    bmi = calculate_bmi(pre_weight, height)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Insert into the database (ID is auto-generated)
    dob_str = dob.strftime('%Y-%m-%d')
    c.execute("INSERT INTO mother (name, dob, height, pre_weight, last_period_date, arm_circumference, twin, pre_bmi) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (name, dob_str, height, pre_weight, last_period, arm, twin, bmi))
    
    conn.commit()
    conn.close()
    
    st.session_state.notification = f"{name} registered successfully!"

def save_measurement(mother_id, lmp_date, meas_date, weight):
    try:
        gest_weeks = calculate_gestational_weeks(lmp_date, meas_date)

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        c.execute("""
            INSERT INTO measurements 
            (mother_id, date, gestational_weeks, weight) 
            VALUES (?, ?, ?, ?)
        """, (
            mother_id,
            meas_date.strftime('%Y-%m-%d'),
            gest_weeks,
            weight
        ))

        conn.commit()
        conn.close()

        st.session_state.notification = "Measurement added successfully"
        return True
    except Exception as e:
        st.error(f"Error saving measurement: {str(e)}")
        return False

def sync_mother_table(original_df, edited_df):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("PRAGMA foreign_keys = ON")
        c = conn.cursor()

        # Detect deleted rows
        deleted_ids = set(original_df.index) - set(edited_df.index)
        for did in deleted_ids:
            c.execute("DELETE FROM mother WHERE id = ?", (int(did),))

        # Update existing rows
        for idx, row in edited_df.iterrows():
            c.execute("""
                UPDATE mother SET
                    name = ?, 
                    dob = ?, 
                    height = ?, 
                    pre_weight = ?, 
                    last_period_date = ?, 
                    arm_circumference = ?, 
                    twin = ?, 
                    pre_bmi = ?
                WHERE id = ?
            """, (
                row["name"],
                pd.to_datetime(row["dob"]).strftime('%Y-%m-%d'),
                row["height"],
                row["pre_weight"],
                row["last_period_date"],
                row["arm_circumference"],
                row["twin"],
                row["pre_bmi"],
                int(idx)
            ))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        st.error(f"Profile update error: {str(e)}")
        return False

def sync_measurement_table(original_df, edited_df, mother_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        # Detect deleted rows
        deleted_ids = set(original_df.index) - set(edited_df.index)
        for did in deleted_ids:
            c.execute("DELETE FROM measurements WHERE id = ?", (int(did),))

        # Update existing rows
        for idx, row in edited_df.iterrows():
            c.execute("""
                UPDATE measurements SET
                    date = ?,
                    gestational_weeks = ?,
                    weight = ?
                WHERE id = ?
            """, (
                pd.to_datetime(row["date"]).strftime('%Y-%m-%d'),
                row["gestational_weeks"],
                row["weight"],
                int(idx)
            ))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        st.error(f"Measurement update error: {str(e)}")
        return False

def delete_mother(mother_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM mother WHERE id = ?", (mother_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting mother: {str(e)}")
        return False


def delete_measurement(measurement_id):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("DELETE FROM measurements WHERE id = ?", (measurement_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting measurement: {str(e)}")
        return False

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

def calculate_gestational_weeks(lmp_date, measurement_date):
    """Calculate gestational age in weeks"""
    try:
        lmp = pd.to_datetime(lmp_date)
        meas = pd.to_datetime(measurement_date)

        if meas < lmp:
            return None

        days = (meas - lmp).days
        weeks = days // 7
        return weeks
    except:
        return None
    
def get_weight_gain_baseline(pre_bmi, pre_weight, twin_status):
    """
    Returns dataframe of week 1–40 expected min/max weight
    Handles singleton and twin pregnancies
    """

    weeks = list(range(1, 41))
    min_weights = []
    max_weights = []

    # ===============================
    # SINGLETON BASELINE
    # ===============================
    if twin_status == "No":

        categories = {
            "under": {"bmi_max": 18.5, "weekly_min": 1*0.4536, "weekly_max": 1.3*0.4536},
            "normal": {"bmi_max": 25, "weekly_min": 0.8*0.4536, "weekly_max": 1*0.4536},
            "over": {"bmi_max": 30, "weekly_min": 0.5*0.4536, "weekly_max": 0.7*0.4536},
            "obese": {"bmi_max": 100, "weekly_min": 0.4*0.4536, "weekly_max": 0.6*0.4536},
        }

        for cat in categories.values():
            if pre_bmi < cat["bmi_max"]:
                selected = cat
                break

        for w in weeks:
            if w <= 13:
                min_gain = (0.5 / 13) * w
                max_gain = (2 / 13) * w
            else:
                min_gain = 0.5 + selected["weekly_min"] * (w - 13)
                max_gain = 2 + selected["weekly_max"] * (w - 13)

            min_weights.append(pre_weight + min_gain)
            max_weights.append(pre_weight + max_gain)

    # ===============================
    # TWIN BASELINE
    # ===============================
    else:

        # Interquartile cumulative gain (kg)
        twin_categories = {
            "normal": {"bmi_max": 25, "total_min": 16.8, "total_max": 24.5},
            "over": {"bmi_max": 30, "total_min": 14.1, "total_max": 22.7},
            "obese": {"bmi_max": 100, "total_min": 11.4, "total_max": 19.1},
        }

        # Underweight fallback → treat as normal
        if pre_bmi < 18.5:
            selected = twin_categories["normal"]
        else:
            for cat in twin_categories.values():
                if pre_bmi < cat["bmi_max"]:
                    selected = cat
                    break

        total_min = selected["total_min"]
        total_max = selected["total_max"]

        # Distribute cumulative gain
        for w in weeks:

            if w <= 13:
                # 20% of total gain occurs in 1st trimester
                min_gain = (0.2 * total_min / 13) * w
                max_gain = (0.2 * total_max / 13) * w
            else:
                # Remaining 80% distributed to week 38
                remaining_weeks = 38 - 13
                min_gain = (0.2 * total_min) + ((0.8 * total_min) / remaining_weeks) * (w - 13)
                max_gain = (0.2 * total_max) + ((0.8 * total_max) / remaining_weeks) * (w - 13)

            min_weights.append(pre_weight + min_gain)
            max_weights.append(pre_weight + max_gain)

    return pd.DataFrame({
        "week": weeks,
        "min_weight": min_weights,
        "max_weight": max_weights
    })

def get_bmi_category(bmi):
    if bmi is None:
        return "Unknown"
    elif bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"
    
def get_trimester(gest_weeks):
    if gest_weeks is None:
        return "Unknown"
    elif gest_weeks <= 13:
        return "First Trimester"
    elif gest_weeks <= 27:
        return "Second Trimester"
    else:
        return "Third Trimester"
    
def get_ced_status(muac):
    if muac is None:
        return "Unknown"
    return "Potentially CED" if muac < 23.5 else "No CED"

def get_lbw_risk(muac):
    if muac is None:
        return "Unknown"
    if muac < 22.0:
        return "High LBW Risk"
    elif muac < 23.0:
        return "Increased LBW Risk"
    elif muac < 24.0:
        return "Possible LBW Risk"
    else:
        return "Lower LBW Risk"
    
def evaluate_health_trend(measurements, baseline_df):
    """
    Evaluate mother's pregnancy weight trend.
    Returns: Improving / Stable / Dropping / No Data
    """

    if measurements.empty or len(measurements) < 2:
        return "Insufficient Data"

    # Get last two measurements
    last = measurements.iloc[-1]
    prev = measurements.iloc[-2]

    weight_diff = last["weight"] - prev["weight"]

    # Get recommended range for current week
    current_week = int(last["gestational_weeks"])

    baseline_row = baseline_df[baseline_df["week"] == current_week]

    if baseline_row.empty:
        return "Unknown"

    min_w = baseline_row["min_weight"].values[0]
    max_w = baseline_row["max_weight"].values[0]

    current_weight = last["weight"]

    # ===============================
    # Decision Logic
    # ===============================

    # Weight dropping
    if weight_diff < -0.3:
        return "Dropping"

    # Below recommended range
    if current_weight < min_w:
        return "Needs Improvement"

    # Above recommended range
    if current_weight > max_w:
        return "Excess Gain"

    # Inside range
    if abs(weight_diff) <= 0.3:
        return "Stable"
    elif weight_diff > 0:
        return "Improving"
    else:
        return "Stable"

def generate_full_simulation(case_type, is_twin=False):
    """
    Create a mother profile + generate simulated data.
    case_type: stable / improving / dropping
    is_twin: True / False
    """

    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("PRAGMA foreign_keys = ON")
        c = conn.cursor()

        # --------------------------
        # Create dummy mother
        # --------------------------
        name = f"Simulation_{case_type.capitalize()}_{'Twin' if is_twin else 'Single'}"
        dob = "1995-01-01"
        height = 160
        pre_weight = 55
        arm = 24.0
        twin = "Yes" if is_twin else "No"

        bmi = calculate_bmi(pre_weight, height)

        # LMP 20 weeks ago
        lmp_date = (pd.Timestamp.today() - pd.Timedelta(weeks=20)).strftime("%Y-%m-%d")

        c.execute("""
            INSERT INTO mother 
            (name, dob, height, pre_weight, last_period_date, arm_circumference, twin, pre_bmi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            dob,
            height,
            pre_weight,
            lmp_date,
            arm,
            twin,
            bmi
        ))

        mother_id = c.lastrowid

        # --------------------------
        # Generate measurements
        # --------------------------
        baseline_df = get_weight_gain_baseline(bmi, pre_weight, twin)

        for week in range(8, 37, 2):

            baseline_row = baseline_df[baseline_df["week"] == week]
            if baseline_row.empty:
                continue

            min_w = baseline_row["min_weight"].values[0]
            max_w = baseline_row["max_weight"].values[0]
            mid_w = (min_w + max_w) / 2

            # Twins fluctuate slightly more
            noise_range = 0.4 if is_twin else 0.25
            noise = random.uniform(-noise_range, noise_range)

            # Slight twin acceleration in 2nd trimester
            twin_boost = (week * 0.05) if is_twin else 0

            if case_type == "stable":
                base_weight = mid_w + twin_boost * 0.2

            elif case_type == "improving":
                if week < 18:
                    base_weight = min_w - (1.5 if is_twin else 1.2)
                elif week < 26:
                    base_weight = min_w - (0.6 if is_twin else 0.4)
                else:
                    base_weight = mid_w + twin_boost * 0.2

            elif case_type == "dropping":
                if week < 20:
                    base_weight = mid_w
                elif week < 28:
                    base_weight = min_w - (1.0 if is_twin else 0.8)
                else:
                    base_weight = min_w - (2.0 if is_twin else 1.6)

            else:
                base_weight = mid_w

            weight = base_weight + noise

            meas_date = pd.to_datetime(lmp_date) + pd.Timedelta(weeks=week)

            c.execute("""
                INSERT INTO measurements (mother_id, date, gestational_weeks, weight)
                VALUES (?, ?, ?, ?)
            """, (
                mother_id,
                meas_date.strftime("%Y-%m-%d"),
                week,
                round(weight, 2)
            ))

        conn.commit()
        conn.close()

        return True

    except Exception as e:
        st.error(f"Simulation error: {str(e)}")
        return False

# --- NOTIFICATION HANDLER ---
if "notification" in st.session_state:
    st.toast(st.session_state.notification, icon="✅")
    del st.session_state["notification"]

# --- MAIN LAYOUT ---
st.title("Mother Health Tracker")

tab1, tab2 = st.tabs(["📝 Data Record", "📊 Dashboard"])

# =======================
# TAB 1: REGISTER & MANAGE DATA
# =======================
with tab1:
    # Use the new DB function
    mother_df = get_mother()

    col1, col2 = st.columns([1, 3])
    
    # --- LEFT COLUMN: REGISTRATION FORM ---
    with col1:
        st.header("📝 Register New Mother")
        with st.form("register_form", clear_on_submit=True):
            name = st.text_input("Mother Name")
            dob = st.date_input("Date of Birth", min_value=datetime(1900, 1, 1), max_value=datetime.today())
            height = st.number_input("Height (cm)", min_value=0.0, step=0.1)            
            pre_weight = st.number_input("Pre-pregnancy Weight (kg)", min_value=0.0, step=0.1)
            last_period = st.date_input("First Day of Last Period Date", min_value=datetime(1900, 1, 1), max_value=datetime.today())
            arm = st.number_input("Arm Circumference (cm)", min_value=0.0, step=0.1)
            twins = st.selectbox("Twins", ["No", "Yes"])

            submitted = st.form_submit_button("Save Mother Profile")
            
            if submitted and name:
                save_mother(name, dob, height, pre_weight, last_period, arm, twins)
                st.rerun()

        st.divider()
        st.subheader("🧪 Quick Demo Simulation")

        col_sim1, col_sim2, col_sim3 = st.columns(3)

        with col_sim1:
            if st.button("Stable (Single)"):
                if generate_full_simulation("stable", False):
                    st.success("Stable singleton simulation created.")
                    st.rerun()

            if st.button("Stable (Twin)"):
                if generate_full_simulation("stable", True):
                    st.success("Stable twin simulation created.")
                    st.rerun()

        with col_sim2:
            if st.button("Improving (Single)"):
                if generate_full_simulation("improving", False):
                    st.success("Improving singleton simulation created.")
                    st.rerun()

            if st.button("Improving (Twin)"):
                if generate_full_simulation("improving", True):
                    st.success("Improving twin simulation created.")
                    st.rerun()

        with col_sim3:
            if st.button("Dropping (Single)"):
                if generate_full_simulation("dropping", False):
                    st.success("Dropping singleton simulation created.")
                    st.rerun()

            if st.button("Dropping (Twin)"):
                if generate_full_simulation("dropping", True):
                    st.success("Dropping twin simulation created.")
                    st.rerun()
        
        
    # --- RIGHT COLUMN: EDIT & VIEW DATA ---
    with col2:
        st.header("📂 Manage Data")
        
        if mother_df.empty:
            st.info("No data registered yet.")
        else:
            # =========================
            # MOTHER TABLE
            # =========================
            st.subheader("Mother Profiles")

            original_profiles = mother_df.copy()

            edited_profiles = st.data_editor(
                original_profiles,
                num_rows="dynamic",
                width='stretch',
                hide_index=False,
                key="editor_profiles"
            )

            if st.button("💾 Update Profiles"):
                if sync_mother_table(original_profiles, edited_profiles):
                    st.success("Profiles updated successfully.")
                    st.rerun()

            st.divider()

            # =========================
            # MEASUREMENT TABLE
            # =========================
            st.subheader("Measurements Data")

            selected_profile_id = st.selectbox(
                "Select Profile",
                options=mother_df.index
            )

            measurement_df = get_mother_measurements(selected_profile_id)

            if not measurement_df.empty:

                original_measurements = measurement_df.copy()

                edited_measurements = st.data_editor(
                    original_measurements,
                    num_rows="dynamic",
                    width='stretch',
                    hide_index=False,
                    key="editor_measurements"
                )

                if st.button("💾 Update Measurements"):
                    if sync_measurement_table(original_measurements, edited_measurements, selected_profile_id):
                        st.success("Measurements updated successfully.")
                        st.rerun()
            else:
                st.info("No measurement data available.")

# =======================
# TAB 2: DASHBOARD
# =======================
with tab2:
    # Load children from DB dynamically
    mother_df = get_mother()

    if mother_df.empty:
        st.warning("No data found. Please go to 'Data Record' tab first and do registration.")
    else:
        # DASHBOARD LAYOUT
        col1, col2 = st.columns([1, 3])

        # --- LEFT COLUMN: SELECT CHILD & ADD DATA ---
        with col1:
            st.subheader("Select Mother")
            
            # Create a list of tuples: (ID, Name)
            options = [(idx, row['name']) for idx, row in mother_df.iterrows()]
            
            selected_id = st.selectbox(
                "Choose a profile:", 
                options=options, 
                format_func=lambda x: x[1],
                key="tab2_dash_selector"
            )

            if isinstance(selected_id, tuple):
                selected_id = selected_id[0]

            mother_info = get_mother_info(selected_id)
            # --- Calculate Age ---
            today = date.today()
            dob = mother_info['dob']
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

            # --- Format LMP (HPHT) ---
            lmp_date = mother_info.get("last_period_date")

            if lmp_date:
                if isinstance(lmp_date, str):
                    lmp_date = datetime.strptime(lmp_date, "%Y-%m-%d").date()
                lmp_str = lmp_date.strftime("%Y-%m-%d")
            else:
                lmp_str = "Not recorded"

            # --- Display Info ---
            st.info(
                f"**Name:** {mother_info['name']}\n\n"
                f"**Age:** {age} years\n\n"
                f"**First Day of Last Period (LMP):** {lmp_str}")

            st.divider()

            # --- ADD MEASUREMENT SECTION ---
            st.subheader("Add Measurement")
            
            # Initialize state for inputs
            if 'm_date' not in st.session_state:
                st.session_state.m_date = datetime.today()
            if 'm_weight' not in st.session_state:
                st.session_state.m_weight = 0.0

            meas_date = st.date_input("Measurement Date", value=st.session_state.m_date, key="in_date")
            weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1, value=st.session_state.m_weight, format="%.2f", key="in_weight")
            height = mother_info['height']

            add_meas = st.button("Add Record", type="primary", key="btn_add_meas_tab2")
            
            if add_meas:
                save_measurement(
                                    selected_id,
                                    mother_info['last_period_date'],
                                    meas_date,
                                    weight
                                )
                st.session_state.m_date = datetime.today()
                st.session_state.m_weight = 0.0
                st.session_state.m_arm = 0.0
                st.rerun()

        # --- RIGHT COLUMN: CHARTS ---
        with col2:
            measurements = get_mother_measurements(selected_id)

            baseline_df = get_weight_gain_baseline(
                mother_info['pre_bmi'],
                mother_info['pre_weight'],
                mother_info['twin']
            )

            # =========================================
            # SUMMARY INFORMATION PANEL
            # =========================================

            bmi_category = get_bmi_category(mother_info["pre_bmi"])
            muac = mother_info["arm_circumference"]
            ced_status = get_ced_status(muac)
            lbw_risk = get_lbw_risk(muac)

            if not measurements.empty:
                latest_row = measurements.iloc[-1]
                current_weight = latest_row["weight"]
                current_weeks = latest_row["gestational_weeks"]
                trimester = get_trimester(current_weeks)
            else:
                current_weight = mother_info["pre_weight"]
                current_weeks = 0
                trimester = "Not yet recorded"

            st.markdown("### 📋 Pregnancy Summary")

            c1, c2, c3, c4, c5 = st.columns(5)

            bmi_value = mother_info["pre_bmi"]

            health_status = evaluate_health_trend(measurements, baseline_df)

            # ---- Column 1: BMI ----
            if bmi_value is not None:
                c1.metric("Pre-Pregnancy BMI", f"{bmi_category}")
                c1.caption(f"BMI: {bmi_value:.2f}")
            else:
                c1.metric("Pre-Pregnancy BMI", "Unknown")

            # ---- Column 2: Current Weight ----
            c2.metric("Current Weight (kg)", f"{current_weight:.1f}")

            # ---- Column 3: Trimester ----
            c3.metric("Current Trimester", trimester)

            # ---- Column 4: MUAC + Nutritional Risk ----
            c4.metric("MUAC (cm)", f"{muac:.1f}")
            c4.caption(f"CED: {ced_status} | LBW: {lbw_risk}")

            # ---- Column 5: Health Status ----
            c5.metric("Current Health Status", health_status)

            st.divider()

            if not measurements.empty:
                measurements = measurements[measurements["gestational_weeks"] <= 40]

                fig = go.Figure()

                # Shaded baseline
                fig.add_trace(go.Scatter(
                    x=baseline_df["week"],
                    y=baseline_df["min_weight"],
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=baseline_df["week"],
                    y=baseline_df["max_weight"],
                    fill='tonexty',
                    name="Recommended Range",
                    line=dict(width=0),
                    fillcolor="rgba(0,150,255,0.2)" if mother_info['twin']=="Yes" else "rgba(0,200,0,0.2)"
                ))

                # Actual measurements
                if not measurements.empty:
                    measurements = measurements[measurements["gestational_weeks"] <= 40]

                    fig.add_trace(go.Scatter(
                        x=measurements["gestational_weeks"],
                        y=measurements["weight"],
                        mode="lines+markers",
                        name="Actual Weight",
                        line=dict(color="black", width=2),
                        marker=dict(color="black", size=5)
                    ))

                fig.update_layout(
                    title=f"Pregnancy Weight Tracking (Twin: {mother_info['twin']})",
                    xaxis_title="Gestational Weeks",
                    yaxis_title="Weight (kg)",
                    xaxis=dict(range=[1, 40]),
                    height=550,
                    template="plotly_white"
                )

                st.plotly_chart(fig, width='stretch')