# Real-Time EV Fleet Monitoring and Predictive Analytics Solution 🚗⚡

A state-of-the-art electric vehicle (EV) fleet management system designed to revolutionize how fleet managers monitor, analyze, and optimize their electric vehicle operations. Built with modern web technologies and powered by machine learning. 🌟
---
![Dashboard](static/ev_dashboard.png)
![Dashboard](static/ev_dashboard1.png)
![Dashboard](static/ev_dashboard2.png)

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
<div align="center">
⭐️ Welcome to EV Management !!
</div>

![Login](static/ev_welcome.png)
- Below, Images stating, Proper Form Validation
![Login](static/ev_login.png)
![Login](static/ev_registration.png)

### 🚙 Vehicle Registration & Monitoring
- Comprehensive vehicle registration system
- Real-time status tracking
- Interactive dashboard with live updates
![Vechicle](static/ev_vehicle_reg.png)
![Vehicle](static/ev_vehicle_status.png)

### 🗺️ Smart Route Optimization
- Intelligent route planning with charging stops
- Real-time battery monitoring
- Dynamic route adjustments
- Charging station locator
- Map visualization with saved images
- Auto-calculation of required charging stops
![Route](static/ev_route.png)

### 🔋 Battery Health Prediction
Machine learning-powered analysis using Random Forest Regressor:
*The `battery_health_model.pkl` file is used after performing Random Forest Regressor to predict battery health and generate maintenance alerts.*

- **Input Features:**
  - Capacity (mAh)
  - Cycle Count
  - Voltage (V)
  - Temperature (°C)
  - Internal Resistance (mΩ)

- **Visual Health Indicators:**
  - Green: Good
  - Red: Needs Attention
  ![Battery](static/ev_battery_good.png)
  ![Route](static/ev_battery_bad.png)

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

![Cost & Energy Dashboard](static/ev_cost.png)
![Cost & Energy Dashboard](static/ev_cost.png)

### Route Optimization
- Intelligent route planning with charging stops
- Calculates how many times the vehicle should stop to reach the destination and charge based on input parameters (battery level, distance, charging station locations, etc.)
- Real-time tracking of route adjustments as battery levels change
- Charging station locator with real-time updates
- Displays dynamic routes and charging stop recommendations with images saved inside the `static` folder for visual reference

![Route Optimization](static/ev_route_map1.png)

### Driver Maintenance
- Real-time Driver Score
- Safety Metrics (95%)
- Efficiency Tracking (88%)
- Compliance Monitoring (93%)

![Driver Maintenance](static/driver.png)

### Vehicle Monitoring
- Current Power Usage
- Average Battery Level
- Active Points Tracking
- Revenue Analysis

![Vehicle Monitoring](static/vehicle.png)
![Vehicle Monitoring](static/vehicle1.png)

## Report Generation

### Generate Customizable Reports
- **Select Vehicle**: Choose the vehicle for the report.
- **Report Type**: Select the type of report (Battery Reports, Driver Reports, Maintenance Reports, Cost & Energy Reports).
- **Date Range**: Select the start and end dates for the report (dd-mm-yyyy).
- **Generate Report**: Button to generate the report based on the selected options.

### Total Reports Generated
- Displays the total number of reports generated.
![Report](static/ev_report3.png)
![Report](static/ev_report1.png)

### Downloaded Reports History
| Date & Time | Vehicle | Report Type | Status (Success/Failure) | Action (Download Link) |
|-------------|---------|-------------|--------------------------|------------------------|
| dd-mm-yyyy  | Vehicle Name | Report Type | Success/Failure         | [Download Link]         |

### Save Reports as PDF
- The generated reports are saved in PDF format, which can be downloaded from the history section.
- This is the pdf generated Structure👇
![Report](static/ev_report2.png)

### Class Diagram
- The class diagram represents the structure of the EV fleet monitoring system, showcasing key classes and their relationships.
- It includes classes for vehicles, drivers, maintenance schedules, route optimization, and battery health prediction.

![Class Diagram](static/ev_class_diagram.png)
</details>

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
### 📄 License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<div align="center">
⭐️ Star this repo if you find it helpful!
Made with ❤️ for the EV community
</div>
