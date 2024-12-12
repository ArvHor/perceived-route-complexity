import time
import pandas as pd
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.distance import geodesic
import overpy
# Initialize geolocator


origin_locations = pd.read_csv('./parameter_data/boeing_locations.csv')
#new_locations = pd.DataFrame(columns=['city_name','train_station_name', 'train_station_lat', 'train_station_lon'])
api = overpy.Overpass()

def get_train_station_coordinates(city):
  """
  Retrieves the coordinates of a train station in a given city using Overpass API.

  Args:
    city: The name of the city.

  Returns:
    A tuple containing the latitude and longitude of the train station as floats, 
    or None if no train station is found.
  """
  geolocator = Nominatim(user_agent="train_station_finder")
  api = overpy.Overpass()
  try:
    location = geolocator.geocode(city)
    if not location:
      print(f"Could not find the city: {city}")
      return None
    city_center_coords = (location.latitude, location.longitude)
    lat, lon = location.latitude, location.longitude
    result = api.query(f"""
      [out:json];
      node(around:10000,{lat},{lon})["railway"="station"];
      out center;
    """)
    if result.nodes:
      closest_station = min(result.nodes,key=lambda node: geodesic(city_center_coords, (float(node.lat), float(node.lon))).km)
      # Get the first train station found
      train_station_name = closest_station.tags.get("name:en")
      if not train_station_name:
        train_station_name = closest_station.tags.get("name", "Unknown")

      city_location = geolocator.reverse((lat, lon), exactly_one=True)
      address = city_location.raw.get('address', {})
      city_country = address.get('country', 'Unknown')
      #city_continent = address.get('continent', 'Unknown')
      #print(f"Train station found: {train_station_name}")
      return float(closest_station.lat), float(closest_station.lon), train_station_name, city_country,#city_continent  # Explicitly convert to floats
    else:
      print(f"No train station found in {city}")
      return None
  except GeocoderTimedOut:
    print(f"Geocoding timed out for city: {city}")
    return None
  except overpy.exception.OverpassTooManyRequests:
    print(f"Too many requests to Overpass API. Try again later.")
    return None
  except overpy.exception.OverpassGatewayTimeout:
    print(f"Timeout error occurred while querying Overpass API for {city}")
    return None
  
new_locations = []

for index, row in origin_locations.iterrows():
  city_name = row['city_name']
  train_station_coords = get_train_station_coordinates(city_name)
  if train_station_coords:
    location = {
      'country': train_station_coords[3],
      'city_name': city_name,
      'train_station_name': train_station_coords[2],
      'train_station_lat': train_station_coords[0],
      'train_station_lon': train_station_coords[1]
    }
    new_locations.append(location)
    print(f"Train station coordinates for {city_name}: {train_station_coords}")
  time.sleep(1)  # Sleep for 1 second to avoid making too many requests in a short period of time

# Save the new DataFrame to a new CSV file
new_locations = pd.DataFrame(new_locations)
new_locations.to_csv('./parameter_data/boeing_locations_with_stations.csv', index=False)
