# EnergyGuard Complete

**AI-Powered Renewable Energy Analysis & Forecasting Platform**

A comprehensive Streamlit application for analyzing, optimizing, and predicting renewable energy generation from waterfall turbines and geothermal systems with advanced LSTM-based seasonal forecasting.

---

## Features

### 1. **Household Energy Status Monitor**
- Real-time energy usage analysis and optimization
- AI-driven anomaly detection and alerts
- Continuous waste energy recovery system (80% capture efficiency)
- Smart recommendations based on:
  - Time of day and sunlight availability
  - Temperature conditions
  - Usage patterns and historical data
- Efficiency scoring and action planning

### 2. **PDF Document Analyzer**
- Automatic extraction of technical data from PDF documents
- Intelligent pattern recognition for:
  - Geographic coordinates (latitude/longitude)
  - Waterfall specifications (height, flow rate)
  - Geothermal data (temperature, drilling depth)
  - Material specifications and power outputs
- Data validation and quality checks
- Direct integration with Geographic Calculator

### 3. **Geographic Energy Calculator**
- Map-based renewable energy potential analysis
- Multiple input methods:
  - Manual coordinate entry
  - Interactive map clicking
  - PDF data import
  - Batch CSV processing
- Comprehensive calculations for:
  - Waterfall turbine power output
  - Geothermal energy generation
  - Waste energy recovery (30% capture, 80% recovery efficiency)
- Material recommendations based on temperature requirements
- Sensitivity analysis for key parameters
- Interactive visualization with Plotly and Folium maps

### 4. **LSTM Time-Series Energy Predictor**
- Advanced neural network forecasting using LSTM architecture
- Trained on 122 years of Bangladesh climate data (1901-2023)
- Seasonal pattern recognition and prediction
- Multiple climate scenario modeling:
  - Normal conditions
  - Wetter (increased monsoon)
  - Drier (reduced rainfall)
  - Hotter temperatures
- Confidence intervals for predictions
- Up to 24-month forecasting capability

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Dependencies

```bash
pip install streamlit pandas numpy matplotlib plotly PyPDF2 folium streamlit-folium tensorflow
```

### Optional Dependencies

For full functionality, install:
```bash
pip install scikit-learn
```

### Clone the Repository

```bash
git clone https://github.com/amrutaVelamuri/earth_prize_final.git
cd earth_prize_final
```

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

### Training the LSTM Model

Before using the time-series predictor, you need to train the LSTM model:

```bash
python train_lstm_model.py
```

This will:
- Load and preprocess the Bangladesh weather dataset
- Train the LSTM neural network
- Save the model as `energy_predictor.h5`
- Save scalers as `scaler_X.pkl` and `scaler_y.pkl`

---

## Application Workflow

### Basic Usage Flow

1. **Start with PDF Analyzer (Optional)**
   - Upload technical PDF documents
   - Automatically extract energy site specifications
   - Validate and review extracted data
   - Send data to Geographic Calculator

2. **Geographic Energy Calculator**
   - Enter location details (manual, map, or from PDF)
   - Input waterfall specifications (height, flow rate)
   - Input geothermal data (temperature, depth)
   - Calculate total energy potential
   - Review waste recovery calculations
   - Export results

3. **LSTM Time-Series Predictor**
   - Select forecast period (3-24 months)
   - Choose climate scenario
   - Generate LSTM predictions
   - Analyze seasonal variations
   - Export forecast data

4. **Household Energy Monitor**
   - Track real-time energy usage
   - Receive AI-powered optimization recommendations
   - Monitor waste recovery performance
   - View historical trends

---

## Technical Details

### Energy Calculations

#### Waterfall Turbine Power
```
P = ρ × g × Q × h × η
```
Where:
- ρ = water density (1000 kg/m³)
- g = gravitational acceleration (9.81 m/s²)
- Q = flow rate (m³/s)
- h = height (m)
- η = turbine efficiency (default 90%)

#### Geothermal Power
```
P_thermal = ṁ × c_p × ΔT
P_electrical = P_thermal × η_conversion
```
Where:
- ṁ = mass flow rate (50 kg/s default)
- c_p = specific heat of water (4.18 kJ/kg·C)
- ΔT = temperature difference (T_underground - T_surface)
- η_conversion = conversion efficiency (15% default)

#### Waste Energy Recovery
```
Waste_available = 30% × Total_generation
Waste_recovered = 80% × Waste_available
Waste_remaining = 20% × Waste_available (system reserve)
```

### LSTM Model Architecture

- **Input Layer**: 12-month sequences of [temperature, rainfall]
- **LSTM Layer 1**: 64 units with 20% dropout
- **LSTM Layer 2**: 32 units with 20% dropout
- **Dense Layer**: 16 units (ReLU activation)
- **Output Layer**: 1 unit (energy prediction in MWh)
- **Training Data**: 1,474 months (1901-2023)
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam

---

## File Structure

```
earth_prize_final/
├── main.py                        # Main Streamlit application
├── train_lstm_model.py            # LSTM training script
├── validation.py                  # Data validation utilities
├── requirements.txt               # Python dependencies
├── energy_predictor.h5            # Trained LSTM model
├── scaler_X.pkl                   # Feature scaler
├── scaler_y.pkl                   # Target scaler
├── training_history.png           # Model training visualization
├── dataset/                       # Training data directory
│   └── bangladesh_weather.csv     # Historical climate data
└── README.md                      # This file
```

---

## Data Sources

### Bangladesh Weather Dataset
- **Source**: Kaggle - [Bangladesh Weather Dataset](https://www.kaggle.com/)
- **Period**: 1901-2023 (122 years)
- **Features**: Monthly temperature (°C) and rainfall (mm)
- **Records**: 1,474 data points
- **Purpose**: Training LSTM model for seasonal energy forecasting
- **Location**: Place the CSV file in `dataset/bangladesh_weather.csv` before training the model

**Note**: This dataset is not included in the repository. Download it from Kaggle and place it in the `dataset/` folder before running `train_lstm_model.py`.

---

## Key Assumptions

### Default Parameters
- Household energy consumption: 7.2 kWh/day (2,628 kWh/year)
- Turbine efficiency: 90%
- Geothermal conversion efficiency: 15%
- Geothermal capacity factor: 85%
- Waste generation: 30% of primary output
- Waste recovery efficiency: 80%
- Geothermal flow rate: 50 kg/s

### Material Selection (Geothermal)
- Temperature < 300°C: Stainless Steel/Incoloy
- Temperature 300-600°C: Inconel alloys/Nickel-chromium
- Temperature > 600°C: Ceramic composites/SiC/Titanium alloys

---

## Features in Detail

### Interactive Mapping
- OpenStreetMap integration via Folium
- Click-to-select location coordinates
- Visual representation of energy potential
- Color-coded markers based on output capacity

### Batch Analysis
- Process multiple locations from CSV files
- Automated calculation pipeline
- Comparative analysis across sites
- Aggregated statistics and visualizations

### Export Capabilities
- CSV format for spreadsheet analysis
- JSON format for data integration
- Complete calculation reports
- LSTM forecast data with confidence intervals

### Validation System
- Coordinate range validation
- Physical parameter bounds checking
- Data completeness verification
- Warning system for unusual values

---

## Model Performance

### LSTM Predictor
- Training on 122 years of climate data
- Learns seasonal patterns and cyclical trends
- Confidence intervals: ±15% (default)
- Adapts predictions to user system capacity
- Accounts for climate scenario variations

### Waste Recovery System
- Continuous operation (independent of main systems)
- Captures thermal losses, friction, and mechanical inefficiencies
- 80% recovery rate from available waste
- 20% reserve for system stability

---

## Limitations

1. **Geographic Scope**: Climate model trained on Bangladesh data; predictions most accurate for South Asian monsoon climates
2. **Temporal Scope**: Forecasts limited to 24 months
3. **Model Assumptions**: Assumes consistent maintenance and operational efficiency
4. **Data Requirements**: LSTM requires historical climate patterns; may need retraining for different regions

---

## Future Enhancements

- Multi-region climate model support
- Real-time sensor data integration
- Advanced optimization algorithms
- Cost-benefit analysis module
- Environmental impact assessment
- Grid integration modeling
- Enhanced material selection algorithms
- Mobile application version

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request with detailed description

---

## License

This project is developed for the Earth Prize competition and community sustainable energy development.

---

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: amrutaVelamuri

---

## Acknowledgments

- Bangladesh Weather Dataset from Kaggle
- Streamlit framework for interactive web applications
- TensorFlow/Keras for LSTM implementation
- OpenStreetMap and Folium for mapping capabilities
- Plotly for advanced visualizations

---

## Version History

- **v1.0**: Initial release with full feature set
  - Household energy monitoring
  - PDF document analyzer
  - Geographic calculator
  - LSTM time-series predictor
  - Batch processing capabilities
  - Interactive mapping and visualization

---

**Built for sustainable community development and renewable energy optimization**
