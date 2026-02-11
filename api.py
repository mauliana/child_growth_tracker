from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

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
            
        return df.to_dict('records'), axis_type, x_col_name
        
    except Exception as e:
        print(f"Error loading WHO standards: {e}")
        return None, None, None

def get_zscore_from_who(gender, metric_type, age_months, value):
    """
    Calculate Z-score using WHO SD curves
    metric_type: 'lhfa' or 'wfa'
    """
    data, _, x_col = get_who_standards(gender, metric_type, age_months)

    if not data:
        return None

    df = pd.DataFrame(data)
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

    for m in measurements:
        age = m.get('age_months')
        value = m.get(metric)

        if age is None or value is None:
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

# ==================== API ENDPOINTS ====================

@app.route('/api/children', methods=['GET'])
def get_children():
    """Get all children"""
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql("SELECT * FROM children ORDER BY id", conn)
        conn.close()
        
        return jsonify({
            'success': True,
            'data': df.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/children/<int:child_id>', methods=['GET'])
def get_child(child_id):
    """Get single child info"""
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql("SELECT * FROM children WHERE id = ?", conn, params=(child_id,))
        conn.close()
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'Child not found'
            }), 404
            
        return jsonify({
            'success': True,
            'data': df.iloc[0].to_dict()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/children', methods=['POST'])
def create_child():
    """Create new child"""
    try:
        data = request.json
        name = data.get('name')
        gender = data.get('gender')
        dob = data.get('dob')
        
        if not all([name, gender, dob]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO children (name, gender, dob) VALUES (?, ?, ?)",
                  (name, gender, dob))
        child_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'{name} registered successfully!',
            'child_id': child_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/children/<int:child_id>', methods=['PUT'])
def update_child(child_id):
    """Update child info"""
    try:
        data = request.json
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("UPDATE children SET name = ?, gender = ?, dob = ? WHERE id = ?",
                  (data['name'], data['gender'], data['dob'], child_id))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Child updated successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/children/<int:child_id>', methods=['DELETE'])
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
        
        return jsonify({
            'success': True,
            'message': 'Child deleted successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/measurements/<int:child_id>', methods=['GET'])
def get_measurements(child_id):
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
        
        return jsonify({
            'success': True,
            'data': df.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/measurements', methods=['POST'])
def create_measurement():
    """Create new measurement"""
    try:
        data = request.json
        child_id = data.get('child_id')
        dob = data.get('dob')
        meas_date = data.get('date')
        weight = data.get('weight')
        height = data.get('height')
        head_circumference = data.get('head_circumference')
        
        # Calculate derived values
        age_m, age_w = calculate_age_in_months(dob, meas_date)
        bmi = calculate_bmi(weight, height)
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            INSERT INTO measurements 
            (child_id, date, dob, age_months, age_weeks, weight, height, head_circumference, bmi) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (child_id, meas_date, dob, age_m, age_w, weight, height, head_circumference, bmi))
        
        measurement_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Measurement added successfully',
            'measurement_id': measurement_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/measurements/<int:measurement_id>', methods=['DELETE'])
def delete_measurement(measurement_id):
    """Delete a measurement"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM measurements WHERE id = ?", (measurement_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Measurement deleted successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/measurements/bulk', methods=['POST'])
def bulk_update_measurements():
    """Bulk update measurements for a child"""
    try:
        data = request.json
        child_id = data.get('child_id')
        measurements = data.get('measurements', [])
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Delete old measurements
        c.execute("DELETE FROM measurements WHERE child_id = ?", (child_id,))
        
        # Insert new measurements
        for m in measurements:
            age_m, age_w = calculate_age_in_months(m['dob'], m['date'])
            bmi = calculate_bmi(m.get('weight'), m.get('height'))
            
            c.execute("""
                INSERT INTO measurements 
                (child_id, date, dob, age_months, age_weeks, weight, height, head_circumference, bmi) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (child_id, m['date'], m['dob'], age_m, age_w, 
                  m.get('weight'), m.get('height'), m.get('head_circumference'), bmi))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Measurements updated successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/who-standards', methods=['GET'])
def get_who_data():
    """Get WHO growth standards"""
    try:
        gender = request.args.get('gender')
        metric_type = request.args.get('metric_type')
        age_months = float(request.args.get('age_months', 0))
        
        data, axis_type, x_col = get_who_standards(gender, metric_type, age_months)
        
        if data is None:
            return jsonify({
                'success': False,
                'error': 'WHO standards not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': data,
            'axis_type': axis_type,
            'x_col': x_col
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health-status', methods=['POST'])
def calculate_health_status():
    """Calculate health status and trends"""
    try:
        data = request.json
        measurements = data.get('measurements', [])
        gender = data.get('gender')
        
        if not measurements:
            return jsonify({
                'success': False,
                'error': 'No measurements provided'
            }), 400
        
        # Get latest measurement
        latest = measurements[-1]
        latest_age = latest['age_months']
        latest_weight = latest['weight']
        latest_height = latest['height']
        latest_bmi = latest.get('bmi') or calculate_bmi(latest_weight, latest_height)
        
        # Calculate statuses
        statuses = {}
        
        # Height status (stunting)
        lhfa_data, _, lhfa_x = get_who_standards(gender, 'lhfa', latest_age)
        if lhfa_data:
            df_lhfa = pd.DataFrame(lhfa_data)
            df_lhfa['diff'] = (df_lhfa[lhfa_x] - latest_age).abs()
            closest = df_lhfa.loc[df_lhfa['diff'].idxmin()]
            
            if latest_height < closest['SD3neg']:
                statuses['growth'] = 'Severely Stunted'
            elif latest_height < closest['SD2neg']:
                statuses['growth'] = 'Stunted'
            else:
                statuses['growth'] = 'Normal'
        
        # Weight status (wasting)
        wfl_data, _, wfl_x = get_who_standards(gender, 'wfl', latest_age)
        if wfl_data:
            df_wfl = pd.DataFrame(wfl_data)
            df_wfl['diff'] = (df_wfl[wfl_x] - latest_height).abs()
            closest = df_wfl.loc[df_wfl['diff'].idxmin()]
            
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
        if bmi_data and latest_bmi:
            df_bmi = pd.DataFrame(bmi_data)
            df_bmi['diff'] = (df_bmi[bmi_x] - latest_age).abs()
            closest = df_bmi.loc[df_bmi['diff'].idxmin()]
            
            if latest_age < 60:  # 0-5 years
                if latest_bmi < closest['SD3neg']:
                    statuses['bmi'] = 'Severely Wasted'
                elif latest_bmi < closest['SD2neg']:
                    statuses['bmi'] = 'Wasted'
                elif latest_bmi > closest['SD3']:
                    statuses['bmi'] = 'Obese'
                elif latest_bmi > closest['SD2']:
                    statuses['bmi'] = 'Overweight'
                elif latest_bmi > closest['SD1']:
                    statuses['bmi'] = 'Risk of Overweight'
                else:
                    statuses['bmi'] = 'Normal'
            else:  # 5-18 years
                if latest_bmi < closest['SD2neg']:
                    statuses['bmi'] = 'Thinness'
                elif latest_bmi > closest['SD2']:
                    statuses['bmi'] = 'Obesity'
                elif latest_bmi > closest['SD1']:
                    statuses['bmi'] = 'Overweight'
                else:
                    statuses['bmi'] = 'Normal'
        
        return jsonify({
            'success': True,
            'statuses': statuses,
            'latest_values': {
                'age': latest_age,
                'weight': latest_weight,
                'height': latest_height,
                'bmi': latest_bmi
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trend-analysis', methods=['POST'])
def trend_analysis():
    """
    Calculate growth trend based on WHO Z-score slope
    """
    try:
        data = request.json
        measurements = data.get('measurements', [])
        gender = data.get('gender')

        if not measurements or len(measurements) < 2:
            return jsonify({
                'success': True,
                'trends': {
                    'height': 'Not enough data',
                    'weight': 'Not enough data'
                }
            })

        height_trend = analyze_trend_from_measurements(
            measurements, gender, metric='height'
        )

        weight_trend = analyze_trend_from_measurements(
            measurements, gender, metric='weight'
        )

        return jsonify({
            'success': True,
            'trends': {
                'height': height_trend,
                'weight': weight_trend
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test-data', methods=['POST'])
def generate_test_data():
    """Generate test data for scenarios"""
    try:
        data = request.json
        scenario = data.get('scenario', 'stable')
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
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
            
            if not hfa_data or not wfa_data:
                continue
            
            df_hfa = pd.DataFrame(hfa_data)
            df_wfa = pd.DataFrame(wfa_data)
            
            df_hfa['diff'] = (df_hfa[hfa_x] - current_age).abs()
            df_wfa['diff'] = (df_wfa[wfa_x] - current_age).abs()
            
            median_height = df_hfa.loc[df_hfa['diff'].idxmin()]['SD0']
            median_weight = df_wfa.loc[df_wfa['diff'].idxmin()]['SD0']

            if hcfa_data:
                df_hcfa = pd.DataFrame(hcfa_data)
                df_hcfa['diff'] = (df_hcfa[hcfa_x] - current_age).abs()
                median_head = df_hcfa.loc[df_hcfa['diff'].idxmin()]['SD0']
            
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
        
        return jsonify({
            'success': True,
            'message': f'Test data generated for {name}',
            'child_id': child_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Initialize database on startup
init_db()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
