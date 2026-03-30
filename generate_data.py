"""
Generate realistic Strava and weather data for visualization
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate 90 days of data
start_date = datetime.now() - timedelta(days=90)
dates = [start_date + timedelta(days=i) for i in range(90)]

# Generate weather data
weather_data = []
for date in dates:
    # Seasonal temperature variation
    day_of_year = date.timetuple().tm_yday
    base_temp = 20 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp = base_temp + np.random.normal(0, 5)
    
    # Weather conditions (affect activity)
    if temp < 5:
        condition = random.choice(['Rainy', 'Snowy', 'Cloudy'])
    elif temp > 30:
        condition = random.choice(['Sunny', 'Hot', 'Sunny'])
    else:
        condition = random.choice(['Sunny', 'Cloudy', 'Partly Cloudy', 'Rainy'])
    
    weather_data.append({
        'date': date,
        'temperature': round(temp, 1),
        'condition': condition,
        'humidity': random.randint(40, 90),
        'wind_speed': random.randint(5, 25)
    })

weather_df = pd.DataFrame(weather_data)

# Generate Strava activities (correlated with weather)
strava_data = []
activity_types = ['Run', 'Bike', 'Walk', 'Swim', 'Hike']

for date in dates:
    weather = weather_df[weather_df['date'] == date].iloc[0]
    
    # Activity probability based on weather
    if weather['condition'] in ['Sunny', 'Partly Cloudy']:
        activity_prob = 0.7
    elif weather['condition'] == 'Cloudy':
        activity_prob = 0.5
    else:
        activity_prob = 0.3
    
    # Generate 0-2 activities per day
    num_activities = np.random.binomial(2, activity_prob)
    
    for _ in range(num_activities):
        activity_type = random.choice(activity_types)
        
        # Distance and duration based on activity type
        if activity_type == 'Run':
            distance = round(np.random.uniform(3, 15), 2)
            duration = round(distance * np.random.uniform(4.5, 6.5), 0)  # minutes
        elif activity_type == 'Bike':
            distance = round(np.random.uniform(10, 50), 2)
            duration = round(distance * np.random.uniform(3, 4.5), 0)
        elif activity_type == 'Walk':
            distance = round(np.random.uniform(2, 8), 2)
            duration = round(distance * np.random.uniform(12, 18), 0)
        elif activity_type == 'Swim':
            distance = round(np.random.uniform(0.5, 2.5), 2)
            duration = round(distance * np.random.uniform(20, 30), 0)
        else:  # Hike
            distance = round(np.random.uniform(5, 20), 2)
            duration = round(distance * np.random.uniform(15, 25), 0)
        
        # Calories (rough estimate)
        calories = int(distance * np.random.uniform(50, 100))
        
        strava_data.append({
            'date': date,
            'activity_type': activity_type,
            'distance_km': distance,
            'duration_min': duration,
            'calories': calories,
            'elevation_gain': random.randint(0, 500) if activity_type in ['Run', 'Bike', 'Hike'] else 0
        })

strava_df = pd.DataFrame(strava_data)

# Save data
weather_df.to_csv('weather_data.csv', index=False)
strava_df.to_csv('strava_data.csv', index=False)

print(f"Generated {len(weather_df)} days of weather data")
print(f"Generated {len(strava_df)} Strava activities")
print("\nSample weather data:")
print(weather_df.head())
print("\nSample Strava data:")
print(strava_df.head())
