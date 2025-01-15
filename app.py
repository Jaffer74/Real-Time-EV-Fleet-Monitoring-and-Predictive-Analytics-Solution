from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import requests
from flask import Flask, request, jsonify
from datetime import datetime
import math
import json
from flask import Blueprint
from polyline import decode
import folium
import os
import logging
from typing import Tuple, List, Dict, Optional
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3  # Example with SQLite for simplicity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.secret_key = 'your_secret_key_here'


GRAPH_HOPPER_API_KEY = 'aa5cb70e-47dd-41a6-bde2-ef5e050e9ceb'
#FOR ROUTES

OPEN_CHARGE_MAP_API_KEY = '181ac90e-4538-4a4c-870f-09efd14661b7'
#FOR EV STATIONS IN REAL TIME

GEOCODE_API_KEY = '2d11c631f5234d00ae51286668dc66f0'
#FOR MAPS


GRAPHHOPPER_BASE_URL = "https://graphhopper.com/api/1/route"
OPEN_CHARGE_MAP_URL = "https://api.openchargemap.io/v3/poi"
GEOCODE_BASE_URL = "https://api.opencagedata.com/geocode/v1/json"

geolocator = Nominatim(user_agent="ev_route_optimizer")

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

db = SQLAlchemy(app)


# Dummy database to simulate user email storage
users_db = {'test@example.com': {'password': 'password123'}}

# Create a database model for the user
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(80))

    def __init__(self, username, password):
        self.username = username
        self.password = password


class Vehicle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.String(100), unique=True)
    owner = db.Column(db.String(100))  # This column should correctly store the owner name
    registration_number = db.Column(db.String(100))
    battery_status = db.Column(db.String(100))
    speed = db.Column(db.Float)
    location = db.Column(db.String(200))

    def __init__(self, vehicle_id, owner, registration_number, battery_status, speed, location):
        self.vehicle_id = vehicle_id
        self.owner = owner
        self.registration_number = registration_number
        self.battery_status = battery_status
        self.speed = speed
        self.location = location

# Load ML Model
model = None
model_path = "battery_health_model.pkl"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model:{e}")


# Home page (index)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/driver-dashboard')
def driver_behavior_dashboard():
    return render_template('driver_dashboard.html')

@app.route('/maintenance-alerts')
def maintenance_alert_dashboard():
    return render_template('maintenance_dashboard.html')

@app.route('/cost-energy')
def cost_energy():
    return render_template('cost_energy_dashboard.html')

@app.route('/report-generation')
def reportgen():
    return render_template('report_generation.html')

# Login route 
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return render_template('login.html', error="Please enter both email and password!")

        user = users_db.get(email)

        if user and user['password'] == password:
            session['logged_in'] = True  # Store login state in session
            return redirect(url_for('home'))  # Redirect to home page
        else:
            return render_template('login.html', error="Invalid credentials! Please try again.")

    return render_template('login.html')

# Home Dashboard
@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('home.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password or not confirm_password:
            return render_template('register.html', error="Please fill out all fields!")

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match!")

        if email in users_db:
            return render_template('register.html', error="Email already registered!")

        users_db[email] = {'username': username, 'password': password}
        return redirect(url_for('login'))

    return render_template('register.html')

def get_coordinates(address: str) -> Tuple[float, float]:
    """
    Get coordinates for an address using OpenCage Geocoding API.
    
    Args:
        address: String address to geocode
    Returns:
        tuple of (latitude, longitude) or (None, None) if not found
    """
    try:
        # Add India for better results with Indian addresses
        if "india" not in address.lower():
            address = f"{address}, India"
            
        params = {
            "q": address,
            "key": GEOCODE_API_KEY,
            "countrycode": "in",
            "limit": 1,
            "no_annotations": 1
        }
        
        response = requests.get(GEOCODE_BASE_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                lat = result["geometry"]["lat"]
                lon = result["geometry"]["lng"]
                logger.info(f"Geocoded {address} to {lat}, {lon}")
                return lat, lon
                
        logger.error(f"Geocoding failed for address: {address}")
        return None, None
            
    except Exception as e:
        logger.error(f"Error during geocoding: {str(e)}")
        return None, None

def get_route(start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> Dict:
    """
    Get route between two points using GraphHopper API.
    
    Args:
        start_coords: tuple of (lat, lon) for start location
        end_coords: tuple of (lat, lon) for end location
    Returns:
        dict with route information or error message
    """
    try:
        params = {
            'vehicle': 'car',
            'locale': 'en',
            'key': GRAPH_HOPPER_API_KEY,
            'points_encoded': True,
            'point': [
                f"{start_coords[0]},{start_coords[1]}",
                f"{end_coords[0]},{end_coords[1]}"
            ]
        }

        response = requests.get(GRAPHHOPPER_BASE_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Failed to fetch route. Status code: {response.status_code}"
            logger.error(error_msg)
            return {"error": error_msg}
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Route request failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate distance between two points using geodesic distance.
    
    Args:
        point1: tuple of (lat, lon)
        point2: tuple of (lat, lon)
    Returns:
        distance in kilometers
    """
    return geodesic(point1, point2).kilometers

def get_charging_stations(route_points: List[List[float]], max_range: float) -> List[Dict]:
    """
    Get or generate charging stations along the route.
    
    Args:
        route_points: List of [lat, lon] points
        max_range: Maximum vehicle range in km
    Returns:
        List of charging station dictionaries
    """
    # Calculate bounding box with buffer
    buffer = (max_range / 111) * 0.5
    min_lat = min(point[0] for point in route_points) - buffer
    max_lat = max(point[0] for point in route_points) + buffer
    min_lon = min(point[1] for point in route_points) - buffer
    max_lon = max(point[1] for point in route_points) + buffer
    
    try:
        # Try to get real charging stations
        response = requests.get(
            OPEN_CHARGE_MAP_URL,
            params={
                'key': OPEN_CHARGE_MAP_API_KEY,
                'boundingbox': f"{min_lat},{min_lon},{max_lat},{max_lon}",
                'maxresults': 100,
                'compact': True
            },
            timeout=10
        )
        
        if response.status_code == 200:
            stations = response.json()
            if stations:
                return stations
    except Exception as e:
        logger.warning(f"Failed to get charging stations from API: {e}")
    
    # Fallback: Generate evenly spaced stations
    logger.info("Using fallback generated stations")
    generated_stations = []
    num_segments = 5
    
    for i in range(1, num_segments):
        index = (i * len(route_points)) // num_segments
        if index >= len(route_points):
            continue
            
        point = route_points[index]
        generated_stations.append({
            'AddressInfo': {
                'Title': f'Charging Station {i}',
                'AddressLine1': f'Location {i}',
                'Latitude': point[0],
                'Longitude': point[1]
            }
        })
    
    return generated_stations








@app.route('/route_optimization', methods=['GET', 'POST'])
def route_optimization():
    if request.method == 'POST':
        try:
            # Get form inputs
            source = request.form['source']
            destination = request.form['destination']
            max_range = float(request.form['max_range'])
            battery_remaining = float(request.form['battery_remaining'])

            # Calculate actual driving range based on battery
            available_range = max_range * (battery_remaining / 100)
            
            # Safety buffer calculations
            SAFETY_MARGIN = 0.15  # 15% safety margin
            MINIMUM_BATTERY = 0.10  # 10% minimum battery level
            
            # Calculate effective safe range
            safe_range = available_range * (1 - SAFETY_MARGIN)
            
            # Get coordinates
            source_coords = get_coordinates(source)
            if not source_coords or not all(source_coords):
                return render_template('route_optimization.html', 
                                    error=f"Could not find coordinates for {source}")
                                    
            dest_coords = get_coordinates(destination)
            if not dest_coords or not all(dest_coords):
                return render_template('route_optimization.html', 
                                    error=f"Could not find coordinates for {destination}")

            # Get route
            route_data = get_route(source_coords, dest_coords)
            if "error" in route_data:
                return render_template('route_optimization.html', 
                                    error=f"Route error: {route_data['error']}")

            route_points = []
            total_distance = 0
            
            if 'paths' in route_data and route_data['paths']:
                path = route_data['paths'][0]
                route_points = decode(path['points'])
                total_distance = path['distance'] / 1000  # Convert to km

                # Calculate stops needed
                available_range = max_range * (battery_remaining / 100)
                safe_range = available_range * 0.85  # 15% safety margin
                stops_needed = max(0, math.ceil(total_distance / safe_range) - 1)

            # Create map
            route_map = folium.Map(location=source_coords, zoom_start=7)
            
            # Add route line
            folium.PolyLine(
                route_points,
                weight=4,
                color='blue',
                opacity=0.8
            ).add_to(route_map)

            # Add markers for source and destination
            folium.Marker(
                source_coords,
                popup=f'Start: {source}<br>Battery: {battery_remaining}%',
                icon=folium.Icon(color='green')
            ).add_to(route_map)
            
            folium.Marker(
                dest_coords,
                popup=f'Destination: {destination}',
                icon=folium.Icon(color='red')
            ).add_to(route_map)

            # Get and add charging stops
            if stops_needed > 0:
                stations = get_charging_stations(route_points, max_range)
                segment_length = total_distance / (stops_needed + 1)
                    
                for i in range(stops_needed):
                        # Calculate position for this stop
                    stop_distance = segment_length * (i + 1)
                    stop_index = int((stop_distance / total_distance) * len(route_points))
                        
                    if stop_index < len(route_points):
                        stop_point = route_points[stop_index]
                        folium.Marker(
                            [stop_point[0], stop_point[1]],
                            popup=f'Charging Stop {i+1}<br>Distance: {stop_distance:.1f} km',
                            icon=folium.Icon(color='blue', icon='plug', prefix='fa')
                        ).add_to(route_map)

                # Fit map bounds
            route_map.fit_bounds(route_map.get_bounds())

                # Save map
            map_filename = f'route_map_{datetime.now().strftime("%Y%m%d%H%M%S")}.html'
            map_path = os.path.join('static', map_filename)
            os.makedirs('static', exist_ok=True)
            route_map.save(map_path)

            return render_template(
                'route_optimization.html',
                route_map=f'/{map_path}',
                stops_needed=stops_needed,
                total_distance=f"{total_distance:.1f}",
                current_range=f"{available_range:.1f}",
                charging_stations=[{
                    'name': f'Charging Stop {i+1}',
                    'address': f'Along route, approximately {segment_length*(i+1):.1f} km from start',
                    'distance': segment_length*(i+1)
                } for i in range(stops_needed)]
            )

        except Exception as e:
            logger.error(f"Route optimization error: {str(e)}")
            return render_template('route_optimization.html', 
                            error=f"An error occurred: {str(e)}")

    return render_template('route_optimization.html')


# Helper function to find the nearest charging station
def find_nearest_charging_station(current_point, charging_stations):
    min_distance = float('inf')
    nearest_station = None

    for station in charging_stations:
        station_coords = (station.get('Longitude'), station.get('Latitude'))
        if station_coords[0] and station_coords[1]:  # Ensure valid coordinates
            distance = calculate_distance(current_point, station_coords)
            if distance < min_distance:
                min_distance = distance
                nearest_station = station

    return nearest_station

def get_bounding_box(route_points, buffer=0.1):
    """
    Get a bounding box around the route with a buffer zone.
    
    Args:
        route_points: List of [lat, lon] points
        buffer: Buffer size in degrees (default 0.1 ≈ 11km at equator)
    """
    min_lat = min(point[0] for point in route_points) - buffer
    max_lat = max(point[0] for point in route_points) + buffer
    min_lon = min(point[1] for point in route_points) - buffer
    max_lon = max(point[1] for point in route_points) + buffer
    
    return min_lat, min_lon, max_lat, max_lon




def get_coordinates(address):
    """
    Get coordinates for an address using OpenCage API.
    
    Args:
        address: String address to geocode
    Returns:
        tuple of (latitude, longitude) or (None, None) if not found
    """
    try:
        # Add India for better results with Indian cities
        if "india" not in address.lower():
            address = f"{address}, India"
            
        params = {
            "q": address,
            "key": GEOCODE_API_KEY,
            "countrycode": "in",  # Restrict to India
            "limit": 1,
            "no_annotations": 1
        }
        
        response = requests.get(GEOCODE_BASE_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                lat = result["geometry"]["lat"]
                lon = result["geometry"]["lng"]
                logging.info(f"Geocoded {address} to {lat}, {lon}")
                return lat, lon
            else:
                logging.error(f"No results found for address: {address}")
        else:
            logging.error(f"Geocoding failed with status code: {response.status_code}")
            
    except Exception as e:
        logging.error(f"Error during geocoding: {str(e)}")
        
    return None, None


def calculate_ev_stops(route_points, total_distance, current_range, charging_stations):
    """
    Calculate optimal charging stops along the route.
    
    Args:
        route_points: List of [lat, lon] points
        total_distance: Total route distance in km
        current_range: Current vehicle range in km
        charging_stations: List of charging stations
    """
    stops = []
    distance_covered = 0
    current_position = route_points[0]
    
    while distance_covered < total_distance:
        # Find next potential charging stop needed
        remaining_distance = total_distance - distance_covered
        
        if remaining_distance <= current_range:
            break  # Can reach destination with current charge
            
        # Find best charging station within range
        best_station = None
        best_progress = 0
        
        for station in charging_stations:
            station_lat = station['AddressInfo'].get('Latitude')
            station_lon = station['AddressInfo'].get('Longitude')
            
            if not station_lat or not station_lon:
                continue
                
            # Calculate distance to station
            distance_to_station = calculate_distance(
                current_position,
                (station_lat, station_lon)
            )
            
            if distance_to_station < current_range:
                # Calculate how far along the route this station is
                for i, point in enumerate(route_points):
                    dist = calculate_distance(
                        (station_lat, station_lon),
                        (point[0], point[1])
                    )
                    if dist < 1:  # Within 1km of route
                        progress = (i / len(route_points)) * total_distance
                        if progress > best_progress:
                            best_progress = progress
                            best_station = station
        
        if best_station:
            stops.append(best_station)
            current_position = (
                best_station['AddressInfo']['Latitude'],
                best_station['AddressInfo']['Longitude']
            )
            distance_covered = best_progress
            current_range = max_range  # Assume full charge after stop
        else:
            # No suitable station found within range
            logging.warning("No suitable charging station found within range")
            break
            
    return stops


# Haversine formula to calculate distance between two geographic points
def calculate_distance(point1, point2):
    from math import radians, sin, cos, sqrt, atan2
    lat1, lon1 = radians(point1[1]), radians(point1[0])
    lat2, lon2 = radians(point2[1]), radians(point2[0])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    R = 6371  # Radius of Earth in kilometers
    return R * c


@app.route('/battery_health_status/', methods=['GET', 'POST'], endpoint='battery_health_status')
def predict_battery_health():
    if request.method == 'POST':
        try:
            # Retrieve and validate form data
            capacity = float(request.form['capacity'])
            cycle_count = int(request.form['cycle_count'])
            voltage = float(request.form['voltage'])
            temperature = float(request.form['temperature'])
            internal_resistance = float(request.form['internal_resistance'])  # Update to match feature name

            if capacity <= 0 or cycle_count < 0 or voltage <= 0 or internal_resistance <= 0 or temperature < -50:
                raise ValueError("All input values must be within valid ranges.")

            # Prepare input data with matching feature names
            input_params = {
                "Capacity (mAh)": capacity,
                "Cycle Count": cycle_count,
                "Voltage (V)": voltage,
                "Temperature (°C)": temperature,
                "Internal Resistance (mΩ)": internal_resistance,  # Updated feature name
                "Distance (km)": 0,  # Provide default value if required by the model
            }
            input_data = pd.DataFrame([input_params])

            # Predict battery health
            prediction = model.predict(input_data)[0]

            return render_template(
                "battery_health_status.html",
                prediction=round(prediction, 2),
                data=input_params
            )
        except ValueError as ve:
            return render_template(
                "battery_health_status.html",
                error=f"Validation Error: {str(ve)}"
            )
        except Exception as e:
            return render_template(
                "battery_health_status.html",
                error=f"An error occurred: {str(e)}"
            )
    else:
        # Render the form for GET requests
        return render_template("battery_health_status.html")

# Logout route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index'))

# Register vehicle route (requires user to be logged in)
@app.route('/register_vehicle/', methods=['GET', 'POST'])
def register_vehicle():
    if not session.get('logged_in'):
        return redirect(url_for('login'))  # Redirect to login if not logged in

    if request.method == 'POST':
        try:
            vehicle_id = request.form.get('vehicle_id')
            owner = request.form.get('owner')
            registration_number = request.form.get('registration_number')
            battery_status = request.form.get('battery_status')
            speed = float(request.form.get('speed'))
            location = request.form.get('location')

            # Debug logs
            print("Vehicle ID:", vehicle_id)
            print("Owner:", owner)
            print("Registration Number:", registration_number)

            # Create new vehicle object
            vehicle = Vehicle(
                vehicle_id=vehicle_id,
                owner=owner,
                registration_number=registration_number,
                battery_status=battery_status,
                speed=speed,
                location=location
            )
            db.session.add(vehicle)
            db.session.commit()

            return render_template('home.html', message="Vehicle registered successfully!")
        except Exception as e:
            print("Error during vehicle registration:", e)  # Debugging log
            return render_template('home.html', message=f"Error: {str(e)}")

    return render_template('register_vehicle.html')


@app.route('/api/real_time_vehicle_status/', methods=['GET'])
def real_time_vehicle_status():
    try:
        vehicles = Vehicle.query.all()  # Fetch all vehicles
        return jsonify([{
            'vehicle_id': v.vehicle_id,
            'owner_name': v.owner,  # Use 'owner' instead of 'owner_name'
            'registration_number': v.registration_number,
            'battery_status': v.battery_status,
            'speed': v.speed
        } for v in vehicles])
    except Exception as e:
        print("Error fetching vehicles:", e)  # Debugging log
        return jsonify({"error": "Unable to fetch vehicle data"}), 500


# Vehicle Status Page
@app.route('/vehicle_status')
def vehicle_status():
    if not session.get('logged_in'):
        return redirect(url_for('login'))  # Redirect to login if not logged in

    return render_template('vehicle_status.html')

if __name__ == '__main__':
    app.run(debug=True)
