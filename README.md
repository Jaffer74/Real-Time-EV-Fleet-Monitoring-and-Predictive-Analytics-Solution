# Real-Time EV Fleet Monitoring and Predictive Analytics Solution 🚗⚡

A state-of-the-art electric vehicle (EV) fleet management system designed to revolutionize how fleet managers monitor, analyze, and optimize their electric vehicle operations. Built with modern web technologies and powered by machine learning. 🌟
---
![Dashboard](static/route_optimization_image.jpg)

## 📑 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [ML Model Details](#ml-model-details)
- [Installation](#installation)
- [API Configuration](#api-configuration)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### 🔐 Authentication & User Management
- Secure login/registration system with email verification
- Role-based access (Fleet Managers & Drivers)
- Session management & secure authentication

### 🚙 Vehicle Registration & Monitoring
- Comprehensive vehicle registration system
- Real-time status tracking
- Interactive dashboard with live updates

### 🗺️ Smart Route Optimization
- Intelligent route planning with charging stops
- Real-time battery monitoring
- Dynamic route adjustments
- Charging station locator
- Map visualization with saved images
- Auto-calculation of required charging stops

### 🔋 Battery Health Prediction
Machine learning-powered analysis using Random Forest Regressor:

- **Input Features:**
  - Capacity (mAh)
  - Cycle Count
  - Voltage (V)
  - Temperature (°C)
  - Internal Resistance (mΩ)

- **Visual Health Indicators:**
  - Green: Good
  - Red: Needs Attention

- **Predictive Maintenance Alerts**

*The `battery_health_model.pkl` file is used after performing Random Forest Regressor to predict battery health and generate maintenance alerts.*

---

## 📂 Project Structure
```plaintext
ev-fleet-monitoring/
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
│   ├── battery_health_status.html
│   ├── battery_stat.html
│   ├── cost_energy_dashboard.html
│   ├── driver_dashboard.html
│   ├── home.html
│   ├── index.html
│   ├── login.html
│   ├── login_success.html
│   ├── maintenance_dashboard.html
│   ├── register.html
│   ├── register_vehicle.html
│   ├── report_generation.html
│   ├── route_optimization.html
│   └── vehicle_status.html
├── app.py
├── battery_data.csv
├── battery_health_model.pkl
├── requirements.txt
├── class_diagram_ev.png
├── README.md
└── LICENSE.md
```
---

## 🤖 ML Model Details
The battery health prediction model uses Random Forest Regressor trained on historical battery data:

```python
# Model Training Overview
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = pd.read_csv('battery_data.csv')
X = data[['Capacity', 'CycleCount', 'Voltage', 'Temperature', 'InternalResistance']]
y = data['HealthStatus']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save model
import pickle
with open('battery_health_model.pkl', 'wb') as file:
    pickle.dump(model, file)
```
---
### Steps:
1. **Clone this repository**:
   ```bash
   git clone https://github.com/Jaffer74/Real-Time-EV-Fleet-Monitoring-and-Predictive-Analytics-Solution.git cd Real-Time-EV-Fleet-Monitoring-and-Predictive-Analytics-Solution
   ```


2. Set Up Python Virtual Environment:
    ```bash 
   python -m venv venv source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
   
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
View the `requirements.txt` file [here](https://github.com/Jaffer74/Real-Time-EV-Fleet-Monitoring-and-Predictive-Analytics-Solution/blob/main/requirements.txt).


5. Run the Application:
   ```bash
   python app.py
   ```

---
## 🌐 API Configuration

```python
# API Endpoints
GRAPHHOPPER_BASE_URL = "https://graphhopper.com/api/1/route"
OPEN_CHARGE_MAP_URL = "https://api.openchargemap.io/v3/poi"
GEOCODE_BASE_URL = "https://api.opencagedata.com/geocode/v1/json"
```
---
## 🎨 Features Details

<details>
<summary>View Detailed Feature Information</summary>

### Cost & Energy Dashboard
- Total Energy Consumption tracking
- Cost per Mile Analysis
- CO₂ Emissions Saved calculation
- Fleet Efficiency Comparison

![Cost & Energy Dashboard](link-to-image)

### Route Optimization
- Intelligent route planning with charging stops
- Calculates how many times the vehicle should stop to reach the destination and charge based on input parameters (battery level, distance, charging station locations, etc.)
- Real-time tracking of route adjustments as battery levels change
- Charging station locator with real-time updates
- Displays dynamic routes and charging stop recommendations with images saved inside the `static` folder for visual reference

![Route Optimization](static/route_optimization_image.jpg)

### Driver Maintenance
- Real-time Driver Score
- Safety Metrics (95%)
- Efficiency Tracking (88%)
- Compliance Monitoring (93%)

![Driver Maintenance](link-to-image)

### Vehicle Monitoring
- Current Power Usage
- Average Battery Level
- Active Points Tracking
- Revenue Analysis

![Vehicle Monitoring](link-to-image)

### Class Diagram
- The class diagram represents the structure of the EV fleet monitoring system, showcasing key classes and their relationships.
- It includes classes for vehicles, drivers, maintenance schedules, route optimization, and battery health prediction.

![Class Diagram](link-to-class-diagram-image)

</details>

---
### 📄 License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
---
