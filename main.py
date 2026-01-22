import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime
import re
import json
import pickle

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

st.set_page_config(page_title="EnergyGuard Complete", layout="wide")

if 'geo_data' not in st.session_state:
    st.session_state.geo_data = {}
if 'pdf_extracted' not in st.session_state:
    st.session_state.pdf_extracted = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'history' not in st.session_state:
    st.session_state.history = {
        'records': [],
        'usage_log': [],
        'recovered_log': [],
        'remaining_log': []
    }

class EnergyRecord:
    def __init__(self, usage, expected, sector, time_of_day, sunlight, temperature):
        self.usage = usage
        self.expected = expected
        self.sector = sector
        self.time_of_day = time_of_day
        self.sunlight = sunlight
        self.temperature = temperature

class EnergyAnalytics:
    @staticmethod
    def usage_ratio(record):
        return record.usage / record.expected if record.expected > 0 else 0

    @staticmethod
    def detect_anomaly(record, history):
        if not history['usage_log']:
            return False
        last = history['usage_log'][-1]
        return record.usage > last * 1.25

    @staticmethod
    def alert_level(ratio, anomaly):
        if ratio >= 1.35 or anomaly:
            return "CRITICAL"
        elif ratio >= 1.15:
            return "WARNING"
        return "NORMAL"

    @staticmethod
    def efficiency_score(ratio):
        score = 100 - abs(ratio - 1) * 75
        return round(max(0, min(100, score)), 1)

    @staticmethod
    def waste_recovery(record):
        wasted = 0.30 * record.usage
        recovered = 0.80 * wasted
        remaining = wasted - recovered
        return round(recovered, 2), round(remaining, 2)

class KeenAI:
    def analyze(self, record, ratio, anomaly, alert, recovered):
        reasons = []
        actions = []
        confidence = 30

        reasons.append(f"Energy usage is {ratio:.2f}x expected")

        if anomaly:
            reasons.append("Sudden abnormal spike detected")
            confidence += 15

        if record.temperature > 30:
            reasons.append("High temperature increased cooling demand")
            confidence += 10

        if record.sunlight and record.time_of_day.lower() == "day":
            reasons.append("Sunlight available but underutilized")
            confidence += 15

        if record.sector.lower() in ["factory", "power plant"]:
            reasons.append("High recoverable industrial losses")
            confidence += 15

        actions.append(("HIGH", f"Recover wasted electricity continuously (~{recovered[0]} kWh)"))
        actions.append(("HIGH", f"Reserved for system stability (~{recovered[1]} kWh)"))

        if anomaly:
            actions.append(("IMMEDIATE", "Activate Null Line to capture leakage"))

        if alert == "CRITICAL":
            actions.append(("IMMEDIATE", "Reduce non-essential loads"))
            actions.append(("HIGH", "Shift base load to geothermal / renewable"))
            if record.sunlight:
                actions.append(("IMMEDIATE", "Activate Smart Daylight-Mirroring System"))
        elif alert == "WARNING":
            actions.append(("MEDIUM", "Optimize operating schedule"))
        else:
            actions.append(("LOW", "System operating optimally"))

        return reasons, actions, min(100, confidence)

tab1, tab2 = st.tabs(["Household Energy Status", "Data and Calculation"])

with tab1:
    st.title("EnergyGuard AI - Household Monitor")
    st.write("Monitor and optimize energy usage with AI-driven insights.")

    analytics = EnergyAnalytics()
    ai = KeenAI()

    with st.form("energy_input"):
        st.subheader("Enter Energy Data")
        usage = st.number_input("Energy usage (kWh):", min_value=0.0, step=0.1)
        expected = st.number_input("Expected usage (kWh):", min_value=0.0, step=0.1)
        sector = st.selectbox("Sector:", ["Home", "Factory", "Power Plant"])
        time_of_day = st.selectbox("Time of Day:", ["Day", "Night"])
        sunlight = st.checkbox("Sunlight available?")
        temperature = st.number_input("Temperature (C):", step=0.1)
        submitted = st.form_submit_button("Analyze Energy")

    if submitted:
        record = EnergyRecord(usage, expected, sector, time_of_day, sunlight, temperature)

        ratio = analytics.usage_ratio(record)
        anomaly = analytics.detect_anomaly(record, st.session_state.history)
        alert = analytics.alert_level(ratio, anomaly)
        score = analytics.efficiency_score(ratio)
        recovered = analytics.waste_recovery(record)

        st.session_state.history['records'].append(record)
        st.session_state.history['usage_log'].append(record.usage)
        st.session_state.history['recovered_log'].append(recovered[0])
        st.session_state.history['remaining_log'].append(recovered[1])

        st.subheader("Energy Status")
        if alert == "CRITICAL":
            st.error("CRITICAL - Immediate optimization required")
        elif alert == "WARNING":
            st.warning("WARNING - Efficiency dropping")
        else:
            st.success("NORMAL - System balanced")

        st.write(f"Efficiency Score: {score}/100")

        reasons, actions, confidence = ai.analyze(record, ratio, anomaly, alert, recovered)

        st.subheader("AI Diagnosis")
        for r in reasons:
            st.write("-", r)

        st.subheader("AI Action Plan")
        for level, act in actions:
            st.write(f"[{level}] {act}")

        st.write(f"AI Confidence Level: {confidence}%")

        if len(st.session_state.history['usage_log']) >= 2:
            st.subheader("Continuous Waste Recovery Performance")
            fig, ax = plt.subplots()
            ax.plot(st.session_state.history['usage_log'], label="Total Usage (kWh)")
            ax.plot(st.session_state.history['recovered_log'], label="Recovered Energy (kWh)")
            ax.plot(st.session_state.history['remaining_log'], label="Unrecovered Waste (kWh)")
            ax.set_xlabel("Monitoring Step")
            ax.set_ylabel("Energy (kWh)")
            ax.set_title("Continuous Waste Recovery Performance")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
with tab2:
    subtab1, subtab2, subtab3 = st.tabs(["PDF Analyzer", "Geographic Calculator", "Time-Series Predictor"])

    with subtab1:
        st.title("Document Analyzer")
        st.markdown("Automatic extraction of energy data from technical documents")

        st.markdown("""
        ### Upload Technical Documents
        This tool automatically extracts:
        - Geographic coordinates (latitude, longitude)
        - Waterfall specifications (height, flow rate)
        - Geothermal data (temperature, drilling depth)
        - Material specifications
        - Energy calculations
        """)

        if not PYPDF2_AVAILABLE:
            st.error("PyPDF2 not installed. Run: pip install PyPDF2")
        else:
            uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])

            if uploaded_file is not None:
                try:
                    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                    
                    full_text = ""
                    for page in pdf_reader.pages:
                        full_text += page.extract_text() + "\n"
                    
                    st.success(f"PDF loaded successfully! {len(pdf_reader.pages)} pages extracted.")
                    
                    with st.expander("View Extracted Text"):
                        st.text_area("Raw Text", full_text, height=300)
                    
                    st.markdown("---")
                    st.subheader("Extracted Data")
                    
                    extracted_data = {}
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Waterfall Data")
                        
                        flow_patterns = [
                            r'[Ff]low\s*[Rr]ate[:\s]*(\d+\.?\d*)\s*m[³3]/s',
                            r'[Ww]ater\s*[Ff]low\s*[Rr]ate[:\s]*(\d+\.?\d*)\s*m[³3]/s',
                            r'Q\s*=\s*(\d+\.?\d*)\s*m[³3]/s',
                            r'[Ff]low[:\s]*(\d+\.?\d*)\s*m[³3]/s',
                            r'(\d+\.?\d*)\s*m[³3]/s',
                        ]
                        
                        waterfall_flow = 0
                        for pattern in flow_patterns:
                            flow_matches = re.findall(pattern, full_text)
                            if flow_matches:
                                waterfall_flow = float(flow_matches[0])
                                extracted_data['waterfall_flow'] = waterfall_flow
                                st.metric("Flow Rate", f"{waterfall_flow} m³/s")
                                break
                        
                        if waterfall_flow == 0:
                            st.info("No flow rate found")
                            extracted_data['waterfall_flow'] = 0
                        
                        height_patterns = [
                            r'[Ww]aterfall\s*[Hh]eight[:\s]*(\d+\.?\d*)\s*m',
                            r'[Hh]eight[:\s]*(\d+\.?\d*)\s*m',
                            r'H\s*=\s*(\d+\.?\d*)\s*m',
                            r'[Hh]ead[:\s]*(\d+\.?\d*)\s*m',
                        ]
                        
                        waterfall_height = 0
                        for pattern in height_patterns:
                            height_matches = re.findall(pattern, full_text)
                            if height_matches:
                                waterfall_height = float(height_matches[0])
                                extracted_data['waterfall_height'] = waterfall_height
                                st.metric("Height", f"{waterfall_height} m")
                                break
                        
                        if waterfall_height == 0:
                            st.info("No height found")
                            extracted_data['waterfall_height'] = 0
                        
                        efficiency_matches = re.findall(r'η\s*=\s*(\d+\.?\d*)', full_text)
                        if efficiency_matches:
                            efficiency = float(efficiency_matches[0])
                            st.metric("Turbine Efficiency", f"{efficiency}")
                        
                        power_matches = re.findall(r'(\d+\.?\d*)\s*MW', full_text)
                        if power_matches:
                            st.info(f"Document mentions: {power_matches[0]} MW power output")
                    
                    with col2:
                        st.markdown("### Geothermal Data")
                        
                        temp_range_matches = re.findall(r'(\d+)\s*[-–]\s*(\d+)\s*°C', full_text)
                        temp_single_matches = re.findall(r'[Tt]emperature[:\s]*(\d+)\s*°C', full_text)
                        
                        if temp_range_matches:
                            temp_range = temp_range_matches[0]
                            avg_temp = (int(temp_range[0]) + int(temp_range[1])) / 2
                            extracted_data['geo_temp'] = avg_temp
                            st.metric("Temperature Range", f"{temp_range[0]}-{temp_range[1]}°C")
                            st.info(f"Using average: {avg_temp}°C")
                        elif temp_single_matches:
                            avg_temp = float(temp_single_matches[0])
                            extracted_data['geo_temp'] = avg_temp
                            st.metric("Temperature", f"{avg_temp}°C")
                        else:
                            st.info("No temperature found")
                            extracted_data['geo_temp'] = 0
                        
                        depth_range_matches = re.findall(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*km', full_text)
                        depth_single_matches = re.findall(r'[Dd]epth[:\s]*(\d+\.?\d*)\s*km', full_text)
                        depth_drilling_matches = re.findall(r'[Dd]rilling\s*[Dd]epth[:\s]*(\d+\.?\d*)\s*km', full_text)
                        
                        if depth_drilling_matches:
                            avg_depth = float(depth_drilling_matches[0])
                            extracted_data['depth'] = avg_depth
                            st.metric("Drilling Depth", f"{avg_depth} km")
                        elif depth_range_matches:
                            depth_range = depth_range_matches[0]
                            avg_depth = (float(depth_range[0]) + float(depth_range[1])) / 2
                            extracted_data['depth'] = avg_depth
                            st.metric("Drilling Depth Range", f"{depth_range[0]}-{depth_range[1]} km")
                            st.info(f"Using average: {avg_depth} km")
                        elif depth_single_matches:
                            avg_depth = float(depth_single_matches[0])
                            extracted_data['depth'] = avg_depth
                            st.metric("Depth", f"{avg_depth} km")
                        else:
                            st.info("No drilling depth found")
                            extracted_data['depth'] = 3.0
                    
                    st.markdown("---")
                    st.subheader("Materials Identified")
                    
                    materials_found = []
                    material_patterns = [
                        r'Stainless Steel',
                        r'Inconel',
                        r'Ceramic composites',
                        r'SiC',
                        r'[Tt]itanium alloys',
                        r'Incoloy'
                    ]
                    
                    for pattern in material_patterns:
                        if re.search(pattern, full_text, re.IGNORECASE):
                            materials_found.append(pattern)
                    
                    if materials_found:
                        st.success(f"Materials mentioned: {', '.join(materials_found)}")
                    
                    st.markdown("---")
                    st.subheader("Location Data")
                    
                    lat_patterns = [
                        r'[Ll]atitude[:\s]*(\d+\.?\d*)',
                        r'[Ll]at[:\s]*(\d+\.?\d*)',
                    ]
                    
                    lon_patterns = [
                        r'[Ll]ongitude[:\s]*(\d+\.?\d*)',
                        r'[Ll]on[:\s]*(\d+\.?\d*)',
                    ]
                    
                    default_lat = 23.8103
                    default_lon = 90.4125
                    coords_found = False
                    
                    for pattern in lat_patterns:
                        lat_match = re.search(pattern, full_text)
                        if lat_match:
                            default_lat = float(lat_match.group(1))
                            coords_found = True
                            break
                    
                    for pattern in lon_patterns:
                        lon_match = re.search(pattern, full_text)
                        if lon_match:
                            default_lon = float(lon_match.group(1))
                            coords_found = True
                            break
                    
                    if coords_found:
                        st.success(f"Coordinates detected: {default_lat}, {default_lon}")
                    else:
                        st.info("Coordinates not auto-detected. Using default Bangladesh location.")
                    
                    if coords_found:
                        if default_lat < -90 or default_lat > 90:
                            st.error(f"ERROR: Invalid latitude {default_lat}. Must be between -90 and 90.")
                            default_lat = 23.8103
                        if default_lon < -180 or default_lon > 180:
                            st.error(f"ERROR: Invalid longitude {default_lon}. Must be between -180 and 180.")
                            default_lon = 90.4125
                    
                    location_name = st.text_input("Location Name", value="Extracted Location")
                    
                    if not location_name or location_name.strip() == "":
                        st.caption("Warning: Location name is required")
                    
                    extracted_data['location_name'] = location_name
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        lat_input = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=default_lat, step=0.0001, format="%.4f", help="Valid range: -90 to +90")
                        extracted_data['latitude'] = lat_input
                    with col2:
                        lng_input = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=default_lon, step=0.0001, format="%.4f", help="Valid range: -180 to +180")
                        extracted_data['longitude'] = lng_input
                    
                    st.markdown("---")
                    st.subheader("Data Validation Summary")
                    
                    validation_messages = []
                    
                    if waterfall_flow > 0 and waterfall_height > 0:
                        validation_messages.append(("VALID", "Waterfall data complete"))
                    elif waterfall_flow > 0 or waterfall_height > 0:
                        validation_messages.append(("WARNING", "Incomplete waterfall data - both height and flow required"))
                    
                    if waterfall_height > 500:
                        validation_messages.append(("WARNING", f"Waterfall height ({waterfall_height}m) is extremely high - verify accuracy"))
                    
                    if waterfall_flow > 1000:
                        validation_messages.append(("WARNING", f"Flow rate ({waterfall_flow} m³/s) is extremely high - verify accuracy"))
                    
                    if extracted_data.get('geo_temp', 0) >= 50:
                        validation_messages.append(("VALID", "Geothermal temperature viable for generation"))
                    elif extracted_data.get('geo_temp', 0) > 0:
                        validation_messages.append(("WARNING", f"Geothermal temperature ({extracted_data.get('geo_temp')}°C) below 50°C - too low for efficient generation"))
                    
                    if extracted_data.get('geo_temp', 0) > 600:
                        validation_messages.append(("WARNING", f"Temperature ({extracted_data.get('geo_temp')}°C) exceeds 600°C - requires specialized equipment"))
                    
                    if extracted_data.get('depth', 0) > 7:
                        validation_messages.append(("WARNING", f"Drilling depth ({extracted_data.get('depth')} km) is very deep - expect high costs"))
                    
                    has_waterfall = waterfall_flow > 0 and waterfall_height > 0
                    has_geothermal = extracted_data.get('geo_temp', 0) >= 50
                    
                    if not has_waterfall and not has_geothermal:
                        validation_messages.append(("ERROR", "No viable energy source detected in document"))
                    
                    if validation_messages:
                        for msg_type, msg in validation_messages:
                            if msg_type == "VALID":
                                st.success(msg)
                            elif msg_type == "WARNING":
                                st.warning(msg)
                            elif msg_type == "ERROR":
                                st.error(msg)
                    else:
                        st.info("No data extracted from document")
                    
                    st.markdown("---")
                    
                    if st.button("Send Data to Geographic Calculator", type="primary"):
                        validation_errors = []
                        
                        if not extracted_data.get('location_name') or extracted_data.get('location_name', '').strip() == "":
                            validation_errors.append("ERROR: Location name is required")
                        
                        has_waterfall = extracted_data.get('waterfall_flow', 0) > 0 and extracted_data.get('waterfall_height', 0) > 0
                        has_geothermal = extracted_data.get('geo_temp', 0) >= 50
                        
                        if not has_waterfall and not has_geothermal:
                            validation_errors.append("ERROR: No viable energy source. Document must contain either waterfall data or geothermal data with temperature >= 50°C")
                        
                        if (extracted_data.get('waterfall_flow', 0) > 0 and extracted_data.get('waterfall_height', 0) == 0) or \
                           (extracted_data.get('waterfall_flow', 0) == 0 and extracted_data.get('waterfall_height', 0) > 0):
                            validation_errors.append("WARNING: Incomplete waterfall data - both height and flow are required")
                        
                        if validation_errors:
                            st.error("### Cannot Send Data - Validation Failed:")
                            for error in validation_errors:
                                if error.startswith("ERROR"):
                                    st.error(error)
                                else:
                                    st.warning(error)
                            
                            critical_errors = [e for e in validation_errors if e.startswith("ERROR")]
                            if critical_errors:
                                st.error("Please correct the errors above before sending data to the calculator.")
                                st.stop()
                        
                        st.session_state.pdf_extracted = extracted_data
                        st.session_state.geo_data['pdf_source'] = uploaded_file.name
                        
                        st.success("Data extracted and sent to Geographic Calculator!")
                        
                        st.markdown("### Extracted Summary")
                        summary_df = pd.DataFrame([{
                            'Parameter': 'Location',
                            'Value': extracted_data.get('location_name', 'N/A'),
                            'Unit': ''
                        }, {
                            'Parameter': 'Latitude',
                            'Value': extracted_data.get('latitude', 0),
                            'Unit': '°'
                        }, {
                            'Parameter': 'Longitude',
                            'Value': extracted_data.get('longitude', 0),
                            'Unit': '°'
                        }, {
                            'Parameter': 'Waterfall Flow',
                            'Value': extracted_data.get('waterfall_flow', 0),
                            'Unit': 'm³/s'
                        }, {
                            'Parameter': 'Waterfall Height',
                            'Value': extracted_data.get('waterfall_height', 0),
                            'Unit': 'm'
                        }, {
                            'Parameter': 'Geothermal Temp',
                            'Value': extracted_data.get('geo_temp', 0),
                            'Unit': '°C'
                        }, {
                            'Parameter': 'Drilling Depth',
                            'Value': extracted_data.get('depth', 0),
                            'Unit': 'km'
                        }])
                        
                        st.dataframe(summary_df, use_container_width=True)
                        
                        st.info("Go to the Geographic Calculator and select 'Use PDF Data' to calculate energy potential!")
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.info("Make sure PyPDF2 is installed: pip install PyPDF2")

            else:
                st.info("Upload a PDF document to begin extraction")
                
                st.markdown("---")
                st.markdown("### Sample Data Format")
                st.markdown("""
                The analyzer looks for patterns like:
                - Flow rates: `Flow Rate: 12.5 m³/s` or `Q = 10 m³/s`
                - Heights: `Waterfall Height: 65 m` or `H = 50 m`
                - Temperatures: `180-200°C` or `Temperature: 190°C`
                - Depths: `Drilling Depth: 2.8 km` or `2-3.5 km`
                - Coordinates: `Latitude: 22.3569` and `Longitude: 91.7832`
                - Power outputs: `4.5 MW`
                """)
with subtab2:
        st.title("Geographic Energy Calculator")
        st.markdown("Map-based renewable energy potential analysis")

        st.sidebar.header("Location Input")
        input_method = st.sidebar.radio(
            "Choose input method:",
            ["Manual Entry", "Click on Map", "Use PDF Data", "Batch Analysis (CSV)"]
        )

        calc_tab1, calc_tab2, calc_tab3, calc_tab4 = st.tabs(["Calculate", "Map View", "Analysis", "Export"])

        with calc_tab2:
            st.header("Interactive Location Map")
            
            if not FOLIUM_AVAILABLE:
                st.error("folium and streamlit-folium not installed. Run: pip install folium streamlit-folium")
            else:
                if st.session_state.geo_data:
                    map_lat = st.session_state.geo_data.get('latitude', 23.8103)
                    map_lng = st.session_state.geo_data.get('longitude', 90.4125)
                else:
                    map_lat = 23.8103
                    map_lng = 90.4125
                
                m = folium.Map(
                    location=[map_lat, map_lng],
                    zoom_start=8,
                    tiles='OpenStreetMap'
                )
                
                if st.session_state.geo_data and 'location_name' in st.session_state.geo_data:
                    location_name = st.session_state.geo_data.get('location_name', 'Selected Location')
                    total_mw = st.session_state.geo_data.get('P_total_MW', 0)
                    households = st.session_state.geo_data.get('households_total', 0)
                    
                    popup_html = f"""
                    <div style="font-family: Arial; width: 200px;">
                        <h4>{location_name}</h4>
                        <b>Total Power:</b> {total_mw:.2f} MW<br>
                        <b>Households:</b> {households:,}<br>
                        <b>Coordinates:</b><br>
                        {map_lat:.4f}, {map_lng:.4f}
                    </div>
                    """
                    
                    if total_mw > 5:
                        color = 'green'
                    elif total_mw > 2:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    folium.Marker(
                        [map_lat, map_lng],
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"{location_name}: {total_mw:.2f} MW",
                        icon=folium.Icon(color=color, icon='info-sign')
                    ).add_to(m)
                
                st.markdown("Click on the map to select a new location")
                map_data = st_folium(m, width=700, height=500, key="main_map")
                
                if map_data and map_data.get('last_clicked'):
                    clicked_lat = map_data['last_clicked']['lat']
                    clicked_lng = map_data['last_clicked']['lng']
                    
                    st.session_state.geo_data['clicked_lat'] = clicked_lat
                    st.session_state.geo_data['clicked_lng'] = clicked_lng
                    
                    st.success(f"New location selected: {clicked_lat:.4f}, {clicked_lng:.4f}")
                    st.info("Go to the 'Calculate' tab and select 'Click on Map' input method to use these coordinates.")

        with calc_tab1:
            st.header("Energy Potential Calculator")
            
            if 'clicked_lat' in st.session_state.geo_data and 'clicked_lng' in st.session_state.geo_data:
                if input_method != "Click on Map":
                    st.warning(f"Map coordinates available: {st.session_state.geo_data['clicked_lat']:.4f}, {st.session_state.geo_data['clicked_lng']:.4f}. Change input method to 'Click on Map' in the sidebar to use them.")
            
            if input_method == "Use PDF Data" and st.session_state.pdf_extracted:
                st.success("Using data from PDF Analyzer")
                initial_latitude = float(st.session_state.pdf_extracted.get('latitude', 23.8103))
                initial_longitude = float(st.session_state.pdf_extracted.get('longitude', 90.4125))
                initial_waterfall_height = float(st.session_state.pdf_extracted.get('waterfall_height', 0.0))
                initial_waterfall_flow = float(st.session_state.pdf_extracted.get('waterfall_flow', 0.0))
                initial_geo_temp = float(st.session_state.pdf_extracted.get('geo_temp', 0.0))
                initial_depth = float(st.session_state.pdf_extracted.get('depth', 3.0))
                initial_location_name = st.session_state.pdf_extracted.get('location_name', 'PDF Location')
            elif input_method == "Use PDF Data":
                st.warning("No PDF data available. Please use the PDF analyzer to upload and extract data first!")
                initial_latitude = 23.8103
                initial_longitude = 90.4125
                initial_waterfall_height = 0.0
                initial_waterfall_flow = 0.0
                initial_geo_temp = 0.0
                initial_depth = 3.0
                initial_location_name = "Default Location"
            elif input_method == "Click on Map":
                if 'clicked_lat' in st.session_state.geo_data and 'clicked_lng' in st.session_state.geo_data:
                    st.success(f"Using map location: {st.session_state.geo_data['clicked_lat']:.4f}, {st.session_state.geo_data['clicked_lng']:.4f}")
                    initial_latitude = float(st.session_state.geo_data['clicked_lat'])
                    initial_longitude = float(st.session_state.geo_data['clicked_lng'])
                else:
                    st.info("Go to the 'Map View' tab to click on a location first.")
                    initial_latitude = 23.8103
                    initial_longitude = 90.4125
                initial_waterfall_height = 50.0
                initial_waterfall_flow = 10.0
                initial_geo_temp = 200.0
                initial_depth = 3.0
                initial_location_name = "Map Location"
            elif input_method == "Batch Analysis (CSV)":
                st.info("Upload CSV in the 'Export' tab for batch processing!")
                initial_latitude = 23.8103
                initial_longitude = 90.4125
                initial_waterfall_height = 50.0
                initial_waterfall_flow = 10.0
                initial_geo_temp = 200.0
                initial_depth = 3.0
                initial_location_name = "Batch Location"
            else:
                initial_latitude = 23.8103
                initial_longitude = 90.4125
                initial_waterfall_height = 50.0
                initial_waterfall_flow = 10.0
                initial_geo_temp = 200.0
                initial_depth = 3.0
                initial_location_name = "My Location"
            
            location_name = st.text_input("Location Name", value=initial_location_name, help="Enter a descriptive name for this location")
            
            if not location_name or location_name.strip() == "":
                st.caption("Warning: Location name is required")
            
            col1, col2 = st.columns(2)
            
            with col1:
                latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=initial_latitude, step=0.0001, format="%.4f", help="Valid range: -90 to +90 (South to North)")
                waterfall_height = st.number_input("Waterfall Height (m)", min_value=0.0, value=initial_waterfall_height, step=5.0, help="Height of waterfall in meters. Leave at 0 if no waterfall.")
                if waterfall_height > 500:
                    st.caption("Warning: Very high waterfall - please verify")
                geo_temp = st.number_input("Geothermal Temperature (C)", min_value=0.0, max_value=900.0, value=initial_geo_temp, step=10.0, help="Underground temperature in Celsius. Minimum 50C for viable generation.")
                if geo_temp > 0 and geo_temp < 50:
                    st.caption("Warning: Temperature too low for efficient energy generation")
                elif geo_temp > 600:
                    st.caption("Warning: Extremely high temperature - specialized equipment required")
            
            with col2:
                longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=initial_longitude, step=0.0001, format="%.4f", help="Valid range: -180 to +180 (West to East)")
                waterfall_flow = st.number_input("Water Flow Rate (m³/s)", min_value=0.0, value=initial_waterfall_flow, step=0.5, help="Water flow in cubic meters per second. Leave at 0 if no waterfall.")
                if waterfall_flow > 1000:
                    st.caption("Warning: Very high flow rate - please verify")
                depth = st.number_input("Drilling Depth (km)", min_value=0.5, max_value=10.0, value=initial_depth, step=0.5, help="Geothermal drilling depth in kilometers. Typical range: 1-5 km")
                if depth > 7:
                    st.caption("Warning: Very deep drilling - higher costs expected")
            
            st.markdown("---")
            
            validation_summary = []
            if waterfall_height > 0 or waterfall_flow > 0:
                if waterfall_height > 0 and waterfall_flow > 0:
                    validation_summary.append("VALID: Waterfall data complete")
                else:
                    validation_summary.append("WARNING: Waterfall data incomplete (need both height and flow)")
            
            if geo_temp >= 50:
                validation_summary.append("VALID: Geothermal data viable")
            elif geo_temp > 0:
                validation_summary.append("WARNING: Geothermal temperature too low")
            
            if not validation_summary:
                st.info("No energy source data entered yet")
            else:
                for summary in validation_summary:
                    if summary.startswith("VALID"):
                        st.success(summary)
                    else:
                        st.warning(summary)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                turbine_efficiency = st.slider("Turbine Efficiency", 0.5, 0.95, 0.9, 0.01)
            with col2:
                geo_efficiency = st.slider("Geothermal Conversion Efficiency", 0.10, 0.25, 0.15, 0.01)
            with col3:
                capacity_factor = st.slider("Capacity Factor", 0.5, 0.95, 0.85, 0.01)
            
            if st.button("Calculate Energy Potential", type="primary"):
                validation_errors = []
                
                if not location_name or location_name.strip() == "":
                    validation_errors.append("ERROR: Location name cannot be empty")
                
                if latitude == 0 and longitude == 0:
                    validation_errors.append("WARNING: Coordinates are at (0, 0). Please verify this is correct.")
                
                if waterfall_height > 0 and waterfall_flow == 0:
                    validation_errors.append("ERROR: Waterfall height is set but flow rate is 0. Both must be greater than 0 for waterfall calculations.")
                
                if waterfall_flow > 0 and waterfall_height == 0:
                    validation_errors.append("ERROR: Flow rate is set but waterfall height is 0. Both must be greater than 0 for waterfall calculations.")
                
                if waterfall_height > 500:
                    validation_errors.append("WARNING: Waterfall height exceeds 500m. This is extremely high - please verify.")
                
                if waterfall_flow > 1000:
                    validation_errors.append("WARNING: Flow rate exceeds 1000 m³/s. This is extremely high - please verify.")
                
                if geo_temp > 0 and geo_temp < 50:
                    validation_errors.append("WARNING: Geothermal temperature below 50C is too low for efficient energy generation.")
                
                if geo_temp > 600:
                    validation_errors.append("WARNING: Temperature exceeds 600C. This requires specialized equipment and safety measures.")
                
                if depth > 10:
                    validation_errors.append("WARNING: Drilling depth exceeds 10 km. This is beyond typical geothermal project limits.")
                
                if depth < 1 and geo_temp > 100:
                    validation_errors.append("WARNING: High temperature at shallow depth is unusual. Please verify data.")
                
                if waterfall_height == 0 and waterfall_flow == 0 and (geo_temp == 0 or geo_temp < 50):
                    validation_errors.append("ERROR: No viable energy source detected. Please enter either waterfall data or geothermal data (temp > 50C).")
                
                if validation_errors:
                    st.error("### Validation Issues Found:")
                    for error in validation_errors:
                        if error.startswith("ERROR"):
                            st.error(error)
                        else:
                            st.warning(error)
                    
                    critical_errors = [e for e in validation_errors if e.startswith("ERROR")]
                    if critical_errors:
                        st.error("Cannot proceed with calculation. Please fix the errors above.")
                        st.stop()
                    else:
                        if not st.checkbox("I acknowledge the warnings above and want to proceed with calculation"):
                            st.stop()
                
                with st.spinner("Calculating..."):
                    rho = 1000
                    g = 9.81
                    specific_heat = 4.18
                    surface_temp = 25
                    
                    if waterfall_height > 0 and waterfall_flow > 0:
                        P_waterfall_watts = rho * g * waterfall_flow * waterfall_height * turbine_efficiency
                        P_waterfall_MW = P_waterfall_watts / 1_000_000
                        E_waterfall_year_MWh = P_waterfall_MW * 24 * 365
                        households_waterfall = int(E_waterfall_year_MWh * 1000 / 7.2)
                        has_waterfall = True
                    else:
                        P_waterfall_MW = 0
                        E_waterfall_year_MWh = 0
                        households_waterfall = 0
                        has_waterfall = False
                    
                    if geo_temp > 50 and depth > 0:
                        flow_rate_geo = 50.0
                        temp_diff = geo_temp - surface_temp
                        
                        thermal_power_kW = flow_rate_geo * specific_heat * temp_diff
                        P_geo_MW = (thermal_power_kW * geo_efficiency) / 1000
                        E_geo_year_MWh = P_geo_MW * 24 * 365 * capacity_factor
                        households_geo = int(E_geo_year_MWh * 1000 / 7.2)
                        
                        if geo_temp < 300:
                            pipe_material = "Stainless Steel / Incoloy"
                            relative_cost = 1.0
                        elif geo_temp < 600:
                            pipe_material = "Inconel alloys / Nickel-chromium"
                            relative_cost = 2.5
                        else:
                            pipe_material = "Ceramic composites / SiC / Titanium alloys"
                            relative_cost = 5.0
                        
                        has_geothermal = True
                    else:
                        P_geo_MW = 0
                        E_geo_year_MWh = 0
                        households_geo = 0
                        pipe_material = "N/A"
                        relative_cost = 0
                        has_geothermal = False
                    
                    base_waste_sources = 0
                    
                    if has_waterfall:
                        waterfall_waste = E_waterfall_year_MWh * 1000 * 0.30
                        base_waste_sources += waterfall_waste
                    
                    if has_geothermal:
                        geothermal_waste = E_geo_year_MWh * 1000 * 0.30
                        base_waste_sources += geothermal_waste
                    
                    waste_recovered_kWh = base_waste_sources * 0.80
                    waste_remaining_kWh = base_waste_sources * 0.20
                    
                    E_waste_recovered_MWh = waste_recovered_kWh / 1000
                    E_waste_remaining_MWh = waste_remaining_kWh / 1000
                    
                    households_waste = int(E_waste_recovered_MWh * 1000 / 7.2)
                    
                    P_total_MW = P_waterfall_MW + P_geo_MW
                    E_total_year_MWh = E_waterfall_year_MWh + E_geo_year_MWh + E_waste_recovered_MWh
                    households_total = int(E_total_year_MWh * 1000 / 7.2)
                    
                    st.session_state.geo_data = {
                        'location_name': location_name,
                        'latitude': latitude,
                        'longitude': longitude,
                        'waterfall_height': waterfall_height,
                        'waterfall_flow': waterfall_flow,
                        'geo_temp': geo_temp,
                        'depth': depth,
                        'P_waterfall_MW': P_waterfall_MW,
                        'P_geo_MW': P_geo_MW,
                        'P_total_MW': P_total_MW,
                        'E_waterfall_year_MWh': E_waterfall_year_MWh,
                        'E_geo_year_MWh': E_geo_year_MWh,
                        'E_waste_recovered_MWh': E_waste_recovered_MWh,
                        'E_waste_remaining_MWh': E_waste_remaining_MWh,
                        'base_waste_sources': base_waste_sources / 1000,
                        'E_total_year_MWh': E_total_year_MWh,
                        'households_total': households_total,
                        'pipe_material': pipe_material,
                        'has_waterfall': has_waterfall,
                        'has_geothermal': has_geothermal
                    }
                    
                    st.session_state.predictions['waterfall_mw'] = P_waterfall_MW
                    st.session_state.predictions['geo_mw'] = P_geo_MW
                    st.session_state.predictions['total_annual_mwh'] = E_total_year_MWh
                    st.session_state.predictions['location'] = location_name
                
                st.success("Calculation Complete!")
                
                st.markdown("### Total System Output")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Power", f"{P_total_MW:.2f} MW")
                with col2:
                    st.metric("Annual Energy", f"{E_total_year_MWh:,.0f} MWh")
                with col3:
                    st.metric("Households Powered", f"{households_total:,}")
                with col4:
                    revenue_estimate = E_total_year_MWh * 80
                    st.metric("Est. Annual Revenue", f"${revenue_estimate:,.0f}")
                
                st.markdown("---")
                
                st.markdown("### Energy Source Breakdown")
                
                st.info("Waste Recovery System: Operates continuously and independently from main generation systems, capturing 80% of thermal and mechanical losses.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if has_waterfall or has_geothermal:
                        labels = []
                        values = []
                        colors = []
                        
                        if has_waterfall:
                            labels.append('Waterfall')
                            values.append(E_waterfall_year_MWh)
                            colors.append('#1f77b4')
                        
                        if has_geothermal:
                            labels.append('Geothermal')
                            values.append(E_geo_year_MWh)
                            colors.append('#ff7f0e')
                        
                        if E_waste_recovered_MWh > 0:
                            labels.append('Waste Recovery')
                            values.append(E_waste_recovered_MWh)
                            colors.append('#2ca02c')
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            marker=dict(colors=colors),
                            hole=0.3
                        )])
                        fig_pie.update_layout(title="Energy Generation Mix")
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    sources = []
                    power_values = []
                    
                    if has_waterfall:
                        sources.append('Waterfall')
                        power_values.append(P_waterfall_MW)
                    
                    if has_geothermal:
                        sources.append('Geothermal')
                        power_values.append(P_geo_MW)
                    
                    if E_waste_recovered_MWh > 0:
                        sources.append('Waste Recovery')
                        power_values.append(E_waste_recovered_MWh / (24 * 365))
                    
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=sources,
                        y=power_values,
                        name='Power (MW)',
                        marker_color='lightblue'
                    ))
                    fig_bar.update_layout(
                        title="Power Output by Source",
                        yaxis_title="Power (MW)",
                        xaxis_title="Energy Source"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with st.expander("Detailed Calculations"):
                    if has_waterfall:
                        st.markdown("#### Waterfall Turbine System")
                        st.write(f"- Flow Rate: {waterfall_flow} m³/s")
                        st.write(f"- Height: {waterfall_height} m")
                        st.write(f"- Efficiency: {turbine_efficiency*100}%")
                        st.write(f"- Power Output: {P_waterfall_MW:.2f} MW")
                        st.write(f"- Annual Energy: {E_waterfall_year_MWh:,.0f} MWh")
                        st.write(f"- Households: {households_waterfall:,}")
                        st.markdown("---")
                    
                    if has_geothermal:
                        st.markdown("#### Geothermal System")
                        st.write(f"- Depth: {depth} km")
                        st.write(f"- Underground Temperature: {geo_temp}C")
                        st.write(f"- Temperature Differential: {geo_temp - surface_temp}C")
                        st.write(f"- Conversion Efficiency: {geo_efficiency*100}%")
                        st.write(f"- Capacity Factor: {capacity_factor*100}%")
                        st.write(f"- Power Output: {P_geo_MW:.2f} MW")
                        st.write(f"- Annual Energy: {E_geo_year_MWh:,.0f} MWh")
                        st.write(f"- Households: {households_geo:,}")
                        st.write(f"- Recommended Pipe Material: {pipe_material}")
                        st.write(f"- Relative Cost Factor: {relative_cost}x")
                        st.markdown("---")
                    
                    if E_waste_recovered_MWh > 0:
                        st.markdown("#### Continuous Waste Energy Recovery (Second Line)")
                        st.write("System Type: Always-ON independent recovery")
                        st.write(f"- Total Waste Available: {base_waste_sources / 1000:,.2f} MWh/year (30% of main output)")
                        st.write(f"- Recovery Efficiency: 80% continuous capture")
                        st.write(f"- Recovered Energy: {E_waste_recovered_MWh:,.2f} MWh/year")
                        st.write(f"- System Reserve: {E_waste_remaining_MWh:,.2f} MWh/year (20% for stability)")
                        st.write(f"- Additional Households Powered: {households_waste:,}")
                        st.info("Note: This recovery system operates continuously and independently, capturing waste heat, friction losses, and mechanical inefficiencies from the primary generation systems.")
                
                st.success("Data saved and sent to Time-Series Predictor")

        with calc_tab3:
            st.header("Geographic Energy Analysis")
            
            if st.session_state.geo_data and 'P_total_MW' in st.session_state.geo_data:
                st.subheader("Optimal Placement Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.geo_data.get('has_waterfall'):
                        st.success("Waterfall Turbine System")
                        st.write("Recommended Installation:")
                        st.write("- Install turbines at the base of waterfall")
                        st.write("- Use adjustable blade systems for flow variation")
                        st.write("- Implement AI-controlled flow monitoring")
                        st.write("- Add modular blade replacement capability")
                        
                        height = st.session_state.geo_data.get('waterfall_height', 0)
                        if height > 100:
                            st.warning("Very high waterfall - consider multiple turbine stages")
                        elif height < 20:
                            st.info("Low head turbine recommended (Kaplan or Francis type)")
                    else:
                        st.info("No waterfall data - not applicable for this location")
                
                with col2:
                    if st.session_state.geo_data.get('has_geothermal'):
                        st.success("Geothermal System")
                        st.write("Recommended Installation:")
                        st.write(f"- Drill to {st.session_state.geo_data.get('depth', 0)} km depth")
                        st.write(f"- Use {st.session_state.geo_data.get('pipe_material', 'N/A')}")
                        st.write("- Implement closed-loop heat exchanger")
                        st.write("- Add AI monitoring for pipe stress & temperature")
                        
                        temp = st.session_state.geo_data.get('geo_temp', 0)
                        if temp > 300:
                            st.warning("High temperature - enhanced safety protocols required")
                        if temp < 150:
                            st.info("Consider binary cycle system for low-temp geothermal")
                    else:
                        st.info("No geothermal data - not applicable for this location")
                
                st.markdown("---")
                st.subheader("Sensitivity Analysis")
                
                sensitivity_param = st.selectbox(
                    "Select parameter to analyze:",
                    ["Waterfall Flow Rate", "Waterfall Height", "Geothermal Temperature", "Drilling Depth"]
                )
                
                if sensitivity_param == "Waterfall Flow Rate":
                    base_flow = st.session_state.geo_data.get('waterfall_flow', 10)
                    base_height = st.session_state.geo_data.get('waterfall_height', 50)
                    
                    flow_range = np.linspace(base_flow * 0.5, base_flow * 1.5, 50)
                    power_range = [1000 * 9.81 * f * base_height * 0.9 / 1_000_000 for f in flow_range]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=flow_range,
                        y=power_range,
                        mode='lines',
                        name='Power Output',
                        line=dict(color='blue', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[base_flow],
                        y=[st.session_state.geo_data.get('P_waterfall_MW', 0)],
                        mode='markers',
                        name='Current Setup',
                        marker=dict(size=15, color='red')
                    ))
                    fig.update_layout(
                        title="Power Output vs Flow Rate",
                        xaxis_title="Flow Rate (m³/s)",
                        yaxis_title="Power (MW)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif sensitivity_param == "Waterfall Height":
                    base_flow = st.session_state.geo_data.get('waterfall_flow', 10)
                    base_height = st.session_state.geo_data.get('waterfall_height', 50)
                    
                    height_range = np.linspace(base_height * 0.5, base_height * 1.5, 50)
                    power_range = [1000 * 9.81 * base_flow * h * 0.9 / 1_000_000 for h in height_range]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=height_range,
                        y=power_range,
                        mode='lines',
                        name='Power Output',
                        line=dict(color='green', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[base_height],
                        y=[st.session_state.geo_data.get('P_waterfall_MW', 0)],
                        mode='markers',
                        name='Current Setup',
                        marker=dict(size=15, color='red')
                    ))
                    fig.update_layout(
                        title="Power Output vs Waterfall Height",
                        xaxis_title="Height (m)",
                        yaxis_title="Power (MW)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif sensitivity_param == "Geothermal Temperature":
                    base_temp = st.session_state.geo_data.get('geo_temp', 200)
                    
                    temp_range = np.linspace(150, 400, 50)
                    power_range = [(50 * 4.18 * (t - 25) * 0.15) / 1000 for t in temp_range]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=temp_range,
                        y=power_range,
                        mode='lines',
                        name='Power Output',
                        line=dict(color='orange', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[base_temp],
                        y=[st.session_state.geo_data.get('P_geo_MW', 0)],
                        mode='markers',
                        name='Current Setup',
                        marker=dict(size=15, color='red')
                    ))
                    fig.update_layout(
                        title="Power Output vs Underground Temperature",
                        xaxis_title="Temperature (C)",
                        yaxis_title="Power (MW)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    base_depth = st.session_state.geo_data.get('depth', 3.0)
                    base_temp = st.session_state.geo_data.get('geo_temp', 200)
                    
                    surface_temp = 25
                    gradient = (base_temp - surface_temp) / base_depth
                    
                    depth_range = np.linspace(1, 10, 50)
                    temp_at_depth = [surface_temp + gradient * d for d in depth_range]
                    power_range = [(50 * 4.18 * (t - surface_temp) * 0.15) / 1000 for t in temp_at_depth]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=depth_range,
                        y=power_range,
                        mode='lines',
                        name='Power Output',
                        line=dict(color='purple', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[base_depth],
                        y=[st.session_state.geo_data.get('P_geo_MW', 0)],
                        mode='markers',
                        name='Current Setup',
                        marker=dict(size=15, color='red')
                    ))
                    fig.update_layout(
                        title="Power Output vs Drilling Depth",
                        xaxis_title="Depth (km)",
                        yaxis_title="Power (MW)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Comparison with Other Renewable Sources")
                
                total_mwh = st.session_state.geo_data.get('E_total_year_MWh', 0)
                
                solar_equiv_capacity = total_mwh / (8760 * 0.20)
                wind_equiv_capacity = total_mwh / (8760 * 0.35)
                
                comparison_df = pd.DataFrame({
                    'Source': ['Your System', 'Equivalent Solar', 'Equivalent Wind'],
                    'Capacity (MW)': [
                        st.session_state.geo_data.get('P_total_MW', 0),
                        solar_equiv_capacity,
                        wind_equiv_capacity
                    ],
                    'Capacity Factor': ['85%', '20%', '35%'],
                    'Annual MWh': [total_mwh, total_mwh, total_mwh]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                st.info("""
                Why your system is better:
                - Higher capacity factor (85% vs 20-35%)
                - More predictable output (not weather dependent)
                - Smaller land footprint
                - Lower intermittency
                """)
            else:
                st.warning("No calculation data available. Please go to the 'Calculate' tab first!")

        with calc_tab4:
            st.header("Export & Batch Analysis")
            
            if st.session_state.geo_data and 'location_name' in st.session_state.geo_data:
                st.subheader("Download Current Calculation")
                
                export_data = {
                    'Location Name': [st.session_state.geo_data.get('location_name', 'N/A')],
                    'Latitude': [st.session_state.geo_data.get('latitude', 0)],
                    'Longitude': [st.session_state.geo_data.get('longitude', 0)],
                    'Waterfall Power (MW)': [st.session_state.geo_data.get('P_waterfall_MW', 0)],
                    'Geothermal Power (MW)': [st.session_state.geo_data.get('P_geo_MW', 0)],
                    'Total Power (MW)': [st.session_state.geo_data.get('P_total_MW', 0)],
                    'Annual Energy (MWh)': [st.session_state.geo_data.get('E_total_year_MWh', 0)],
                    'Households Powered': [st.session_state.geo_data.get('households_total', 0)],
                    'Pipe Material': [st.session_state.geo_data.get('pipe_material', 'N/A')]
                }
                
                export_df = pd.DataFrame(export_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"energy_calc_{st.session_state.geo_data.get('location_name', 'location').replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_str = export_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download as JSON",
                        data=json_str,
                        file_name=f"energy_calc_{st.session_state.geo_data.get('location_name', 'location').replace(' ', '_')}.json",
                        mime="application/json"
                    )
            
            st.markdown("---")
            st.subheader("Batch Analysis from CSV")
            
            st.markdown("""
            Upload a CSV file with multiple locations to analyze them all at once.
            
            Required CSV format:
            ```
            location_name,latitude,longitude,waterfall_height_m,waterfall_flow_m3s,geo_temp_c,depth_km
            Location 1,23.8103,90.4125,50,10,200,3.0
            Location 2,24.8949,91.8687,30,12,150,2.5
            ```
            """)
            
            sample_data = {
                'location_name': ['Chittagong Hills', 'Sylhet Valley', 'Khulna Region'],
                'latitude': [22.3569, 24.8949, 22.8456],
                'longitude': [91.7832, 91.8687, 89.5403],
                'waterfall_height_m': [45, 30, 0],
                'waterfall_flow_m3s': [8.5, 12.0, 0],
                'geo_temp_c': [180, 150, 200],
                'depth_km': [2.8, 2.5, 3.5]
            }
            sample_df = pd.DataFrame(sample_data)
            
            st.download_button(
                label="Download Sample CSV Template",
                data=sample_df.to_csv(index=False),
                file_name="sample_locations_template.csv",
                mime="text/csv"
            )
            
            uploaded_csv = st.file_uploader("Choose a CSV file", type=['csv'], key='batch_csv')
            
            if uploaded_csv is not None:
                try:
                    batch_df = pd.read_csv(uploaded_csv)
                    st.write("### Uploaded Data Preview")
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    if st.button("Process All Locations", type="primary"):
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in batch_df.iterrows():
                            status_text.text(f"Processing {row.get('location_name', f'Location {idx+1}')}...")
                            
                            lat = row.get('latitude', 0)
                            lng = row.get('longitude', 0)
                            h = row.get('waterfall_height_m', 0)
                            q = row.get('waterfall_flow_m3s', 0)
                            temp = row.get('geo_temp_c', 0)
                            depth = row.get('depth_km', 3.0)
                            
                            if h > 0 and q > 0:
                                p_waterfall = (1000 * 9.81 * q * h * 0.9) / 1_000_000
                                e_waterfall = p_waterfall * 24 * 365
                                waterfall_waste = e_waterfall * 1000 * 0.30
                            else:
                                p_waterfall = 0
                                e_waterfall = 0
                                waterfall_waste = 0
                            
                            if temp > 50:
                                p_geo = (50 * 4.18 * (temp - 25) * 0.15) / 1000
                                e_geo = p_geo * 24 * 365 * 0.85
                                geothermal_waste = e_geo * 1000 * 0.30
                            else:
                                p_geo = 0
                                e_geo = 0
                                geothermal_waste = 0
                            
                            total_waste = waterfall_waste + geothermal_waste
                            e_waste_recovered = (total_waste * 0.80) / 1000
                            
                            total_annual = e_waterfall + e_geo + e_waste_recovered
                            households = int(total_annual * 1000 / 7.2)
                            
                            if temp < 300:
                                material = "Stainless Steel"
                            elif temp < 600:
                                material = "Inconel alloys"
                            else:
                                material = "Ceramic composites"
                            
                            results.append({
                                'Location': row.get('location_name', f'Location {idx+1}'),
                                'Latitude': lat,
                                'Longitude': lng,
                                'Waterfall_MW': round(p_waterfall, 2),
                                'Geothermal_MW': round(p_geo, 2),
                                'Total_Annual_MWh': round(total_annual, 0),
                                'Households': households,
                                'Pipe_Material': material if temp > 50 else 'N/A'
                            })
                            
                            progress_bar.progress((idx + 1) / len(batch_df))
                        
                        status_text.text("Processing complete!")
                        
                        results_df = pd.DataFrame(results)
                        st.write("### Batch Analysis Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Locations", len(results_df))
                        with col2:
                            total_power = results_df['Waterfall_MW'].sum() + results_df['Geothermal_MW'].sum()
                            st.metric("Combined Power", f"{total_power:.2f} MW")
                        with col3:
                            total_households = results_df['Households'].sum()
                            st.metric("Total Households", f"{total_households:,}")
                        
                        if FOLIUM_AVAILABLE:
                            fig = px.scatter_mapbox(
                                results_df,
                                lat='Latitude',
                                lon='Longitude',
                                size='Total_Annual_MWh',
                                color='Total_Annual_MWh',
                                hover_name='Location',
                                hover_data=['Households', 'Waterfall_MW', 'Geothermal_MW'],
                                color_continuous_scale='Viridis',
                                size_max=20,
                                zoom=5
                            )
                            
                            fig.update_layout(
                                mapbox_style="open-street-map",
                                title="Energy Potential Map - All Locations"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.download_button(
                            label="Download Batch Results",
                            data=results_df.to_csv(index=False),
                            file_name="batch_energy_analysis_results.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.info("Please make sure your CSV has the correct column names and format.")
with subtab3:
        st.title("LSTM Time-Series Energy Predictor")
        st.markdown("AI-powered seasonal forecasting using real climate data")

        has_geo_data = bool(st.session_state.geo_data and 'P_total_MW' in st.session_state.geo_data)

        if not has_geo_data:
            st.warning("No calculation data available from Geographic Calculator!")
            st.info("Please go to the Geographic Calculator and run calculations first.")
            st.markdown("---")
            st.markdown("### What This Tool Does:")
            st.markdown("""
            - Uses a trained LSTM neural network
            - Trained on 122 years of Bangladesh weather data (1901-2023)
            - Predicts monthly energy output based on seasonal climate patterns
            - Provides confidence intervals for predictions
            - Forecasts up to 24 months ahead
            """)
            st.stop()

        @st.cache_resource
        def load_lstm_model():
            try:
                model = load_model('energy_predictor.h5', compile=False)
                model.compile(
                    optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error']
                )
                with open('scaler_X.pkl', 'rb') as f:
                    scaler_X = pickle.load(f)
                with open('scaler_y.pkl', 'rb') as f:
                    scaler_y = pickle.load(f)
                return model, scaler_X, scaler_y, None
            except Exception as e:
                return None, None, None, str(e)

        if TENSORFLOW_AVAILABLE:
            with st.spinner("Loading LSTM model..."):
                model, scaler_X, scaler_y, error = load_lstm_model()
            
            if error:
                st.error(f"Could not load model: {error}")
                st.info("Make sure you've run train_lstm_model.py first!")
                st.stop()
            else:
                st.success("LSTM model loaded successfully!")
        else:
            st.error("TensorFlow not installed. Run: pip install tensorflow")
            st.stop()

        st.sidebar.header("Prediction Settings")

        forecast_months = st.sidebar.slider("Forecast Period (months)", 3, 24, 12)
        start_month = st.sidebar.selectbox("Starting Month", 
            ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November', 'December'],
            index=datetime.now().month - 1
        )

        st.sidebar.markdown("### Climate Scenario")
        climate_scenario = st.sidebar.radio(
            "Select scenario:",
            ["Normal", "Wetter (More Monsoon)", "Drier (Less Rain)", "Hotter"]
        )

        pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs(["LSTM Forecast", "Model Info", "Comparison", "Export"])

        with pred_tab1:
            st.header("LSTM Energy Forecast")
            
            base_power_mw = st.session_state.geo_data.get('P_total_MW', 0)
            waterfall_mw = st.session_state.geo_data.get('P_waterfall_MW', 0)
            geo_mw = st.session_state.geo_data.get('P_geo_MW', 0)
            location_name = st.session_state.geo_data.get('location_name', 'Selected Location')
            
            st.markdown(f"### Forecasting for: {location_name}")
            st.markdown(f"Base Power Capacity: {base_power_mw:.2f} MW")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Waterfall Component", f"{waterfall_mw:.2f} MW")
            with col2:
                st.metric("Geothermal Component", f"{geo_mw:.2f} MW")
            
            st.markdown("---")
            
            if st.button("Generate LSTM Forecast", type="primary"):
                with st.spinner("Running LSTM predictions..."):
                    month_map = {
                        'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }
                    
                    start_month_num = month_map[start_month]
                    
                    bangladesh_climate = {
                        1: (19, 20), 2: (22, 25), 3: (26, 50), 4: (28, 100),
                        5: (28, 250), 6: (28, 350), 7: (28, 400), 8: (28, 350),
                        9: (27, 300), 10: (26, 150), 11: (23, 30), 12: (20, 15)
                    }
                    
                    climate_adjustments = {
                        "Normal": (1.0, 1.0),
                        "Wetter (More Monsoon)": (1.0, 1.3),
                        "Drier (Less Rain)": (1.0, 0.7),
                        "Hotter": (1.1, 0.9)
                    }
                    
                    temp_mult, rain_mult = climate_adjustments[climate_scenario]
                    
                    months = []
                    temperatures = []
                    rainfalls = []
                    
                    for i in range(forecast_months):
                        month_num = (start_month_num + i - 1) % 12 + 1
                        year_offset = (start_month_num + i - 1) // 12
                        
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        month_label = f"{month_names[month_num-1]} Y{year_offset+1}"
                        months.append(month_label)
                        
                        base_temp, base_rain = bangladesh_climate[month_num]
                        
                        np.random.seed(42 + i)
                        temp = base_temp * temp_mult * np.random.normal(1.0, 0.05)
                        rain = base_rain * rain_mult * np.random.normal(1.0, 0.1)
                        
                        temperatures.append(temp)
                        rainfalls.append(rain)
                    
                    historical_months = []
                    for i in range(12, 0, -1):
                        month_num = (start_month_num - i) % 12
                        if month_num == 0:
                            month_num = 12
                        base_temp, base_rain = bangladesh_climate[month_num]
                        historical_months.append([base_temp, base_rain])
                    
                    predictions_mwh = []
                    confidence_lower = []
                    confidence_upper = []
                    
                    current_sequence = np.array(historical_months)
                    
                    for i in range(forecast_months):
                        sequence_scaled = scaler_X.transform(current_sequence)
                        sequence_scaled = sequence_scaled.reshape(1, 12, 2)
                        
                        prediction_scaled = model.predict(sequence_scaled, verbose=0)
                        prediction_mwh = scaler_y.inverse_transform(prediction_scaled)[0][0]
                        
                        user_monthly_mwh = (base_power_mw * 730)
                        scale_factor = user_monthly_mwh / 3500
                        
                        scaled_prediction = prediction_mwh * scale_factor
                        
                        predictions_mwh.append(scaled_prediction)
                        
                        confidence_lower.append(scaled_prediction * 0.85)
                        confidence_upper.append(scaled_prediction * 1.15)
                        
                        new_point = [temperatures[i], rainfalls[i]]
                        current_sequence = np.vstack([current_sequence[1:], new_point])
                    
                    hours_per_month = 730
                    predictions_mw = [e / hours_per_month for e in predictions_mwh]
                    
                    waterfall_ratio = waterfall_mw / base_power_mw if base_power_mw > 0 else 0.5
                    geo_ratio = geo_mw / base_power_mw if base_power_mw > 0 else 0.5
                    
                    waterfall_predictions = [p * waterfall_ratio for p in predictions_mwh]
                    geo_predictions = [p * geo_ratio for p in predictions_mwh]
                    
                    st.session_state.predictions = {
                        'months': months,
                        'temperatures': temperatures,
                        'rainfalls': rainfalls,
                        'waterfall_mw': [w / hours_per_month for w in waterfall_predictions],
                        'geo_mw': [g / hours_per_month for g in geo_predictions],
                        'total_mw': predictions_mw,
                        'waterfall_mwh': waterfall_predictions,
                        'geo_mwh': geo_predictions,
                        'total_mwh': predictions_mwh,
                        'confidence_lower': confidence_lower,
                        'confidence_upper': confidence_upper,
                        'total_annual_mwh': sum(predictions_mwh),
                        'location': location_name,
                        'climate_scenario': climate_scenario
                    }
                
                st.success("LSTM forecast generated successfully!")
            
            if 'months' in st.session_state.predictions:
                months = st.session_state.predictions['months']
                predictions_mw = st.session_state.predictions['total_mw']
                predictions_mwh = st.session_state.predictions['total_mwh']
                confidence_lower = st.session_state.predictions['confidence_lower']
                confidence_upper = st.session_state.predictions['confidence_upper']
                temperatures = st.session_state.predictions['temperatures']
                rainfalls = st.session_state.predictions['rainfalls']
                
                st.markdown("### Monthly Energy Forecast (LSTM)")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=months + months[::-1],
                    y=[c/730 for c in confidence_upper] + [c/730 for c in confidence_lower[::-1]],
                    fill='toself',
                    fillcolor='rgba(0,100,200,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval (±15%)',
                    showlegend=True
                ))
                
                fig.add_trace(go.Scatter(
                    x=months,
                    y=predictions_mw,
                    mode='lines+markers',
                    name='LSTM Prediction',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f"LSTM Energy Forecast - {location_name} ({st.session_state.predictions['climate_scenario']} Scenario)",
                    xaxis_title="Month",
                    yaxis_title="Power Output (MW)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Forecast Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_power = np.mean(predictions_mw)
                    st.metric("Average Power", f"{avg_power:.2f} MW")
                
                with col2:
                    total_energy = sum(predictions_mwh)
                    st.metric("Total Energy (Forecast Period)", f"{total_energy:,.0f} MWh")
                
                with col3:
                    peak_month = months[np.argmax(predictions_mw)]
                    peak_power = max(predictions_mw)
                    st.metric("Peak Month", peak_month)
                    st.caption(f"{peak_power:.2f} MW")
                
                with col4:
                    households = int(total_energy * 1000 / 7.2)
                    st.metric("Avg Households Powered", f"{households:,}")
                
                st.markdown("---")
                st.markdown("### Climate Inputs (LSTM Features)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(
                        x=months,
                        y=temperatures,
                        mode='lines+markers',
                        name='Temperature',
                        line=dict(color='red')
                    ))
                    fig_temp.update_layout(
                        title='Temperature Forecast',
                        xaxis_title='Month',
                        yaxis_title='Temperature (C)',
                        height=300
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                
                with col2:
                    fig_rain = go.Figure()
                    fig_rain.add_trace(go.Bar(
                        x=months,
                        y=rainfalls,
                        name='Rainfall',
                        marker_color='lightblue'
                    ))
                    fig_rain.update_layout(
                        title='Rainfall Forecast',
                        xaxis_title='Month',
                        yaxis_title='Rainfall (mm)',
                        height=300
                    )
                    st.plotly_chart(fig_rain, use_container_width=True)
                
                with st.expander("View Detailed Monthly Predictions"):
                    prediction_df = pd.DataFrame({
                        'Month': months,
                        'Temp (C)': [f"{t:.1f}" for t in temperatures],
                        'Rain (mm)': [f"{r:.0f}" for r in rainfalls],
                        'Power (MW)': [f"{p:.2f}" for p in predictions_mw],
                        'Energy (MWh)': [f"{e:,.0f}" for e in predictions_mwh],
                        'Households': [int(e * 1000 / 7.2) for e in predictions_mwh]
                    })
                    st.dataframe(prediction_df, use_container_width=True)
            else:
                st.info("Click 'Generate LSTM Forecast' button above to create predictions")

        with pred_tab2:
            st.header("LSTM Model Information")
            
            st.markdown("""
            ### About This Model
            
            This is a Long Short-Term Memory (LSTM) neural network trained to predict energy output 
            based on climate patterns.
            
            #### Training Data:
            - Source: Bangladesh Weather Dataset (Kaggle)
            - Period: 1901-2023 (122 years)
            - Features: Monthly temperature and rainfall
            - Records: 1,474 months of historical data
            
            #### Model Architecture:
            - Type: LSTM Recurrent Neural Network
            - Layers: 
              - LSTM Layer 1: 64 units
              - Dropout: 20%
              - LSTM Layer 2: 32 units
              - Dropout: 20%
              - Dense: 16 units
              - Output: 1 unit (energy prediction)
            - Input: 12-month sequences of temperature + rainfall
            - Output: Next month's energy output (MWh)
            
            #### How It Works:
            1. Takes historical climate data (temp + rainfall) for past 12 months
            2. LSTM learns seasonal patterns and trends
            3. Predicts energy output for next month
            4. Scales prediction to match your system capacity
            5. Repeats for each forecasted month
            
            #### Physical Basis:
            - Rainfall → Waterfall Power: More rain = higher water flow = more power
            - Temperature → Geothermal: Higher temp = better heat extraction
            - Model learns these relationships from 122 years of climate cycles
            """)

        with pred_tab3:
            st.header("LSTM vs Baseline Comparison")
            
            if 'total_annual_mwh' in st.session_state.predictions:
                predicted_annual = st.session_state.predictions['total_annual_mwh']
                actual_annual = st.session_state.geo_data.get('E_total_year_MWh', 0)
                
                st.markdown("### Predicted vs Baseline")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Baseline Annual Energy", f"{actual_annual:,.0f} MWh")
                    st.caption("From Geographic Calculator (constant)")
                
                with col2:
                    st.metric("LSTM Predicted Energy", f"{predicted_annual:,.0f} MWh")
                    st.caption(f"Accounting for {st.session_state.predictions.get('climate_scenario', 'Normal')} climate")
                
                with col3:
                    difference_pct = ((predicted_annual - actual_annual) / actual_annual * 100) if actual_annual > 0 else 0
                    st.metric("Difference", f"{difference_pct:+.1f}%")
                    st.caption("Due to seasonal/climate factors")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Baseline',
                    x=['Annual Energy'],
                    y=[actual_annual],
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name='LSTM Prediction',
                    x=['Annual Energy'],
                    y=[predicted_annual],
                    marker_color='darkblue'
                ))
                
                fig.update_layout(
                    title='Annual Energy Comparison',
                    yaxis_title='Energy (MWh)',
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Generate a forecast first!")

        with pred_tab4:
            st.header("Export Predictions")
            
            if 'months' in st.session_state.predictions:
                export_df = pd.DataFrame({
                    'Month': st.session_state.predictions['months'],
                    'Temperature_C': st.session_state.predictions['temperatures'],
                    'Rainfall_mm': st.session_state.predictions['rainfalls'],
                    'Total_MW': st.session_state.predictions['total_mw'],
                    'Monthly_Energy_MWh': st.session_state.predictions['total_mwh'],
                    'Confidence_Lower_MWh': st.session_state.predictions['confidence_lower'],
                    'Confidence_Upper_MWh': st.session_state.predictions['confidence_upper']
                })
                
                st.dataframe(export_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download LSTM Predictions (CSV)",
                        data=csv,
                        file_name=f"lstm_forecast_{st.session_state.predictions.get('location', 'location')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    summary = {
                        'model': 'LSTM Neural Network',
                        'location': st.session_state.predictions.get('location', 'N/A'),
                        'climate_scenario': st.session_state.predictions.get('climate_scenario', 'Normal'),
                        'forecast_period_months': len(st.session_state.predictions['months']),
                        'total_predicted_energy_mwh': st.session_state.predictions['total_annual_mwh'],
                        'average_monthly_mwh': np.mean(st.session_state.predictions['total_mwh']),
                        'predictions': export_df.to_dict('records')
                    }
                    
                    json_str = json.dumps(summary, indent=2)
                    st.download_button(
                        label="Download Report (JSON)",
                        data=json_str,
                        file_name=f"lstm_forecast_report_{st.session_state.predictions.get('location', 'location')}.json",
                        mime="application/json"
                    )
            else:
                st.warning("Generate a forecast first!")

st.markdown("---")
st.markdown("EnergyGuard Complete - Community Energy Toolkit")
st.caption("Built for sustainable community development")