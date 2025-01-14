import requests
from geopy.distance import distance as geopy_distance
import os

# Fetch the API key from environment variable
API_KEY = os.getenv("OPEN_CHARGE_MAP_API_KEY")  # Make sure to set this environment variable

# Coordinates for cities
city_coords = {
    'Chennai': (13.0827, 80.2707),
    'Coimbatore': (11.0168, 76.9558),
    'Madurai': (9.9250, 78.1198),
    'Salem': (11.6643, 78.1460),
    'Trichy': (10.7905, 78.7047),
    'Erode': (11.3410, 77.7172),
    'Vellore': (12.9165, 79.1323),
    'Tirunelveli': (8.7277, 77.7063),
    'Puducherry': (11.9416, 79.8083),
    'Kanchipuram': (12.8333, 79.7056),
    'Tanjore': (10.7870, 79.1590),
    'Ramanagaram': (12.7324, 78.2495),
    'Dindigul': (10.3640, 77.9800),
    'Dharmapuri': (12.1189, 78.1393),
    'Tiruppur': (11.1085, 77.3411),
    'Theni': (10.0111, 77.4777),
    'Sivakasi': (9.4511, 77.7974),
    'Karur': (10.9576, 78.0807),
    'Nagapattinam': (10.7639, 79.8424),
    'Nagercoil': (8.1784, 77.4280),
    'Cuddalore': (11.7460, 79.7714),
    'Perambalur': (11.2333, 78.8667),
    'Krishnagiri': (12.5186, 78.2138),
    'Ariyalur': (11.1442, 79.0788)
}

# Calculate distance between two coordinates
def calculate_distance(coords1, coords2):
    return geopy_distance(coords1, coords2).km

# Get EV details from the user
def get_ev_details():
    while True:
        try:
            max_range = int(input("Enter the max range of the EV (in km): "))
            remaining_charge = int(input("Enter the remaining battery percentage: "))
            remaining_km = (max_range * remaining_charge) / 100
            return max_range, remaining_km
        except ValueError:
            print("Please enter valid numeric values for max range and remaining charge.")

# Fetch charging station data from the Open Charge Map API
def get_charging_stations(country_code="IN", max_results=10):
    # Using string interpolation for country_code and max_results, and API_KEY should be retrieved securely
    API_KEY = '4cc97fd0-c572-481d-aa87-ee767e1d84bd'  # This is just for example; avoid hardcoding keys
    url = f'https://api.openchargemap.io/v3/poi/?output=json&countrycode={country_code}&maxresults={max_results}&key={API_KEY}'

    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        # Check if the response was successful (2xx status code)
        response.raise_for_status()  # This will raise an error for 4xx or 5xx status codes
        stations = response.json()
        return stations
    except requests.exceptions.RequestException as e:
        print(f"Error fetching charging stations: {e}")
        return []  # Return an empty list if there's an error



# Find the optimized route
def find_optimized_route(source, destination, max_range, remaining_km, stations):
    current_location = city_coords[source]
    destination_coords = city_coords[destination]
    route = []
    visited_stations = set()
    iterations = 0

    while remaining_km < calculate_distance(current_location, destination_coords):
        iterations += 1
        print(f"\nIteration: {iterations}")
        print(f"Current location: {current_location}")
        print(f"Remaining km: {remaining_km}")

        best_station = None
        best_distance = 0

        for station in stations:
            if 'AddressInfo' not in station or 'Latitude' not in station['AddressInfo'] or 'Longitude' not in station['AddressInfo']:
                continue

            station_coords = (station['AddressInfo']['Latitude'], station['AddressInfo']['Longitude'])
            distance_to_station = calculate_distance(current_location, station_coords)

            if station['ID'] in visited_stations:
                continue

            if distance_to_station <= remaining_km and distance_to_station > best_distance:
                best_station = station
                best_distance = distance_to_station

        if best_station is None:
            print("No charging station found within the remaining distance. Cannot continue.")
            break

        # Recharge and update variables
        visited_stations.add(best_station['ID'])
        route.append(best_station)
        remaining_km = max_range
        current_location = (best_station['AddressInfo']['Latitude'], best_station['AddressInfo']['Longitude'])

        print(f"Recharging at station {best_station['ID']} located at {best_station['AddressInfo']['Latitude']}, {best_station['AddressInfo']['Longitude']}")
        print(f"Distance to station: {best_distance} km")

    # Add final destination to route
    if remaining_km >= calculate_distance(current_location, destination_coords):
        route.append({'station_id': 'Destination', 'latitude': destination_coords[0], 'longitude': destination_coords[1]})
        print("\nRoute optimization successful. Reached destination.")

    return route
