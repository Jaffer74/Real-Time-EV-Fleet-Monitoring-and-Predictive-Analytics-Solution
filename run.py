import os
import requests
from geopy.distance import distance as geopy_distance

# Fetch the Open Charge Map API Key from environment variable
API_KEY = os.getenv('OPEN_CHARGE_MAP_API_KEY')  # Ensure the environment variable is set

if not API_KEY:
    print("API Key is missing. Please set the OPEN_CHARGE_MAP_API_KEY environment variable.")
    exit(1)

# Coordinates for cities in Tamil Nadu
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
    max_range = int(input("Enter the max range of the EV (in km): "))
    remaining_charge = int(input("Enter the remaining battery percentage: "))
    remaining_km = (max_range * remaining_charge) / 100
    return max_range, remaining_km

# Fetch charging station data from the Open Charge Map API
def get_charging_stations(country_code="IN", max_results=10):
    url = f'https://api.openchargemap.io/v3/poi/?output=json&countrycode={country_code}&maxresults={max_results}&key={API_KEY}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        stations = response.json()
        return stations
    except requests.exceptions.RequestException as e:
        print(f"Error fetching charging stations: {e}")
        return []

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

# Example usage
if __name__ == "__main__":
    stations = get_charging_stations(country_code="IN", max_results=10)
    if stations:
        source = input("Enter the source city: ")
        destination = input("Enter the destination city: ")

        if source in city_coords and destination in city_coords:
            max_range, remaining_km = get_ev_details()
            route = find_optimized_route(source, destination, max_range, remaining_km, stations)
            print("\nOptimized Route:")
            for step in route:
                print(step)
        else:
            print("Invalid source or destination city.")
    else:
        print("No charging stations available.")
