"""
Create a professional dashboard-style visualization and export as PNG
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime
import seaborn as sns

# Set style for modern dashboard look
plt.style.use('dark_background')
sns.set_palette("husl")

# Load data
weather_df = pd.read_csv('weather_data.csv', parse_dates=['date'])
strava_df = pd.read_csv('strava_data.csv', parse_dates=['date'])

# Prepare data
weather_df['day_of_week'] = weather_df['date'].dt.dayofweek
weather_df['week'] = weather_df['date'].dt.isocalendar().week
weather_df['month'] = weather_df['date'].dt.month

# Aggregate activities by date
daily_activities = strava_df.groupby('date').agg({
    'distance_km': 'sum',
    'calories': 'sum',
    'activity_type': 'count',
    'duration_min': 'sum'
}).rename(columns={'activity_type': 'num_activities'}).reset_index()

# Merge weather and activity data
merged = weather_df.merge(daily_activities, on='date', how='left')
merged['distance_km'] = merged['distance_km'].fillna(0)
merged['calories'] = merged['calories'].fillna(0)
merged['num_activities'] = merged['num_activities'].fillna(0)
merged['duration_min'] = merged['duration_min'].fillna(0)

# Activity and weather distributions
activity_dist = strava_df['activity_type'].value_counts()

# Weather grouping helpers
good_conditions = ['Sunny', 'Partly Cloudy', 'Hot']
merged['weather_group'] = np.where(
    merged['condition'].isin(good_conditions),
    'Good (Sunny/Hot)',
    'Poor (Rain/Cloud/Other)'
)

# Calculate key metrics focused on weather–activity relationship
total_distance = merged['distance_km'].sum()
total_calories = merged['calories'].sum()
total_activities = len(strava_df)
active_days = (merged['num_activities'] > 0).sum()
avg_temp_active = merged[merged['num_activities'] > 0]['temperature'].mean()
avg_temp_inactive = merged[merged['num_activities'] == 0]['temperature'].mean()

good_days = merged[merged['weather_group'] == 'Good (Sunny/Hot)']
poor_days = merged[merged['weather_group'] == 'Poor (Rain/Cloud/Other)']

good_active_rate = (good_days['num_activities'] > 0).mean() * 100 if len(good_days) > 0 else 0
poor_active_rate = (poor_days['num_activities'] > 0).mean() * 100 if len(poor_days) > 0 else 0

good_avg_daily_distance = good_days['distance_km'].mean() if len(good_days) > 0 else 0
poor_avg_daily_distance = poor_days['distance_km'].mean() if len(poor_days) > 0 else 0

distance_good_share = (
    good_days['distance_km'].sum() / total_distance * 100 if total_distance > 0 else 0
)

condition_activity = (
    merged.groupby('condition')['num_activities']
    .sum()
    .sort_values(ascending=False)
)
top_condition = condition_activity.index[0] if len(condition_activity) > 0 else 'N/A'
top_condition_acts = int(condition_activity.iloc[0]) if len(condition_activity) > 0 else 0

# Date range for subtitle
start_date = merged['date'].min()
end_date = merged['date'].max()

# Weekly aggregation (still used for temporal context in other analyses if needed)
merged['week_start'] = merged['date'].dt.to_period('W').dt.start_time
weekly_data = merged.groupby('week_start').agg({
    'distance_km': 'sum',
    'calories': 'sum',
    'num_activities': 'sum',
    'temperature': 'mean'
}).reset_index()

# Condition-level stats for weather charts
condition_stats = merged.groupby('condition').agg({
    'distance_km': 'sum',
    'num_activities': 'sum',
    'temperature': 'mean'
}).reset_index().sort_values('distance_km', ascending=False)

# Activity-level stats by weather (for bottom charts)
strava_with_weather = strava_df.merge(
    weather_df[['date', 'condition']],
    on='date',
    how='left'
)
strava_with_weather['weather_group'] = np.where(
    strava_with_weather['condition'].isin(good_conditions),
    'Good (Sunny/Hot)',
    'Poor (Rain/Cloud/Other)'
)
activity_weather_avg = (
    strava_with_weather.groupby(['activity_type', 'weather_group'])['distance_km']
    .mean()
    .unstack(fill_value=0)
)

# Humidity bands for new chart (average distance & activity by humidity level)
humidity_bins = pd.cut(merged['humidity'], bins=[40, 50, 60, 70, 80, 90, 100], right=False)
humidity_stats = merged.groupby(humidity_bins).agg({
    'distance_km': 'mean',
    'num_activities': 'mean'
}).reset_index()
humidity_labels = [f"{int(interval.left)}–{int(interval.right-1)}%" for interval in humidity_stats['humidity']]

# Day-of-week stats including weather
dow_stats = merged.groupby('day_of_week').agg({
    'distance_km': 'mean',
    'num_activities': 'mean',
    'temperature': 'mean',
    'condition': lambda c: np.mean(c.isin(good_conditions))
}).reset_index().rename(columns={'condition': 'sunny_share'})
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Create dashboard figure (larger size for better readability)
fig = plt.figure(figsize=(24, 14), facecolor='#0d1117')
# Grid: 1 row for KPIs + 2 rows for a 2×2 chart layout
gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.3, 
              left=0.05, right=0.95, top=0.90, bottom=0.06)

# Title (question the dashboard answers)
fig.suptitle('HOW DOES WEATHER INFLUENCE MY ACTIVITY PATTERNS?', fontsize=24, fontweight='bold', 
             color='white', y=0.97)

# Subtitle with date range (under the title, above subplots)
fig.text(
    0.5,
    0.935,
    f"{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}",
    ha='center',
    va='center',
    fontsize=12,
    color='#8b949e'
)

# Note explaining weather groups (between KPIs and charts)
fig.text(
    0.5,
    0.69,
    "*) Good weather = Sunny, Partly Cloudy, Hot   |   Poor weather = Rain, Cloudy, Snowy, and any other non-good condition",
    ha='center',
    va='center',
    fontsize=9,
    color='#8b949e'
)

# ========== TOP ROW: WEATHER-LINKED KPIS ==========
# Metric 1: Activity rate on good vs poor weather days
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#161b22')
ax1.text(0.5, 0.7, f'{good_active_rate:.0f}% vs {poor_active_rate:.0f}%', ha='center', va='center',
         fontsize=28, fontweight='bold', color='#58a6ff')
ax1.text(0.5, 0.42, 'Activity Rate (Good vs Poor Weather)*', ha='center', va='center',
         fontsize=13, color='#8b949e')
ax1.text(0.5, 0.2, 'Days with ≥ 1 activity', ha='center', va='center',
         fontsize=11, color='#58a6ff', style='italic')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Metric 2: Avg daily distance on good vs poor weather days
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#161b22')
ax2.text(0.5, 0.7, f'{good_avg_daily_distance:.1f} vs {poor_avg_daily_distance:.1f}', ha='center', va='center',
         fontsize=28, fontweight='bold', color='#f85149')
ax2.text(0.5, 0.42, 'Avg Daily Distance (km)', ha='center', va='center',
         fontsize=13, color='#8b949e')
ax2.text(0.5, 0.2, 'Good vs Poor Weather Days*', ha='center', va='center',
         fontsize=11, color='#f85149', style='italic')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')

# Metric 3: Share of distance on good-weather days
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#161b22')
ax3.text(0.5, 0.7, f'{distance_good_share:.0f}%', ha='center', va='center',
         fontsize=32, fontweight='bold', color='#3fb950')
ax3.text(0.5, 0.42, 'Of Distance on Good-Weather Days*', ha='center', va='center',
         fontsize=13, color='#8b949e')
ax3.text(0.5, 0.2, 'Share of total distance', ha='center', va='center',
         fontsize=11, color='#3fb950', style='italic')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')

# Metric 4: Most active weather condition
ax4 = fig.add_subplot(gs[0, 3])
ax4.set_facecolor('#161b22')
ax4.text(0.5, 0.7, top_condition, ha='center', va='center',
         fontsize=20, fontweight='bold', color='#d29922')
ax4.text(0.5, 0.42, 'Most Active Weather', ha='center', va='center',
         fontsize=13, color='#8b949e')
ax4.text(0.5, 0.2, f'{top_condition_acts} activities', ha='center', va='center',
         fontsize=11, color='#d29922', style='italic')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# ========== SECOND ROW: PATTERN-FOCUSED CHARTS ==========
# Chart 1: Distance vs temperature (2D density heatmap)
ax5 = fig.add_subplot(gs[1, 0:2])
ax5.set_facecolor('#161b22')
temp_values = merged['temperature'].values
dist_values = merged['distance_km'].values

if np.any(dist_values > 0):
    temp_bins = np.linspace(temp_values.min(), temp_values.max(), 10)
    dist_bins = np.linspace(0, dist_values.max(), 10)
    heatmap, xedges, yedges = np.histogram2d(
        temp_values,
        dist_values,
        bins=[temp_bins, dist_bins]
    )

    im5 = ax5.imshow(
        heatmap.T,
        origin='lower',
        aspect='auto',
        extent=[
            xedges[0],
            xedges[-1],
            yedges[0],
            yedges[-1]
        ],
        cmap='YlOrRd'
    )
    ax5.set_title(
        'Distance vs Temperature',
        fontsize=16,
        fontweight='bold',
        color='white',
        pad=15
    )
    ax5.set_xlabel('Temperature (°C)', fontsize=12, color='#8b949e')
    ax5.set_ylabel('Daily Distance (km)', fontsize=12, color='#8b949e')
    ax5.tick_params(colors='#8b949e')
    for spine in ax5.spines.values():
        spine.set_color('#30363d')
    cbar5 = plt.colorbar(im5, ax=ax5, pad=0.02)
    cbar5.set_label('Number of Days', color='#8b949e')
    cbar5.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(plt.getp(cbar5.ax.axes, 'yticklabels'), color='#8b949e')
else:
    ax5.text(
        0.5,
        0.5,
        'No distance data available',
        ha='center',
        va='center',
        fontsize=12,
        color='#8b949e'
    )
    ax5.set_axis_off()

# Chart 2: Good vs poor weather distance distribution
ax6 = fig.add_subplot(gs[1, 2:4])
ax6.set_facecolor('#161b22')
box_data = []
box_labels = []
box_colors = []
if len(good_days) > 0:
    box_data.append(good_days['distance_km'])
    box_labels.append('Good (Sunny/Hot)')
    box_colors.append('#58a6ff')
if len(poor_days) > 0:
    box_data.append(poor_days['distance_km'])
    box_labels.append('Poor (Rain/Cloud/Other)')
    box_colors.append('#f4a261')

if box_data:
    box = ax6.boxplot(
        box_data,
        patch_artist=True,
        tick_labels=box_labels
    )
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in box['medians']:
        median.set_color('white')

    ax6.set_title(
        'Daily Distance Distribution by Weather Group',
        fontsize=16,
        fontweight='bold',
        color='white',
        pad=15
    )
    ax6.set_ylabel('Daily Distance (km)', fontsize=12, color='#8b949e')
    ax6.tick_params(colors='#8b949e')
    ax6.grid(True, alpha=0.2, color='#30363d', axis='y')
    for spine in ax6.spines.values():
        spine.set_color('#30363d')
else:
    ax6.text(
        0.5,
        0.5,
        'No data for weather groups',
        ha='center',
        va='center',
        fontsize=12,
        color='#8b949e'
    )
    ax6.set_axis_off()

# ========== THIRD ROW: HUMIDITY IMPACT ==========
# Chart 3: Activity and humidity relationship
ax7 = fig.add_subplot(gs[2, 0:2])
ax7.set_facecolor('#161b22')
hum_pos = np.arange(len(humidity_stats))
width_h = 0.4
bars_h = ax7.bar(
    hum_pos,
    humidity_stats['distance_km'],
    width_h,
    label='Avg distance (km)',
    color='#58a6ff',
    alpha=0.85
)
ax7.set_title('Activity vs Humidity Level', fontsize=16, fontweight='bold', color='white', pad=15)
ax7.set_xlabel('Humidity Band', fontsize=12, color='#8b949e')
ax7.set_ylabel('Avg Distance (km)', fontsize=12, color='#58a6ff')
ax7.set_xticks(hum_pos)
ax7.set_xticklabels(humidity_labels, rotation=0, ha='center', color='#8b949e')
ax7.grid(True, alpha=0.2, color='#30363d', axis='y')
ax7.tick_params(colors='#8b949e')
for spine in ax7.spines.values():
    spine.set_color('#30363d')

ax7_twin = ax7.twinx()
line_h, = ax7_twin.plot(
    hum_pos,
    humidity_stats['num_activities'],
    color='#f4a261',
    marker='o',
    linewidth=2,
    label='Avg # activities'
)
ax7_twin.set_ylabel('Avg Number of Activities', fontsize=12, color='#f4a261')
ax7_twin.tick_params(colors='#f4a261')
for spine in ax7_twin.spines.values():
    spine.set_color('#30363d')

lines_h, labels_h = ax7.get_legend_handles_labels()
lines_h2, labels_h2 = ax7_twin.get_legend_handles_labels()
ax7.legend(
    lines_h + lines_h2,
    labels_h + labels_h2,
    loc='upper left',
    facecolor='#161b22',
    edgecolor='#30363d',
    labelcolor='white',
    fontsize=10
)

# ========== THIRD ROW (BOTTOM): WEATHER TOTALS ==========
# Chart 4: Total distance vs weather condition
ax9 = fig.add_subplot(gs[2, 2:4])
ax9.set_facecolor('#161b22')

if not condition_stats.empty:
    # use previously computed condition_stats (total distance per condition)
    totals = condition_stats.sort_values('distance_km', ascending=False)
    x_tot = np.arange(len(totals))
    bars_tot = ax9.bar(
        x_tot,
        totals['distance_km'],
        color='#58a6ff',
        alpha=0.85,
        label='Total distance (km)'
    )
    ax9.set_title('Total Distance vs Weather Condition', fontsize=16, fontweight='bold', color='white', pad=15)
    ax9.set_xlabel('Weather Condition', fontsize=12, color='#8b949e')
    ax9.set_ylabel('Total Distance (km)', fontsize=12, color='#58a6ff')
    ax9.set_xticks(x_tot)
    ax9.set_xticklabels(totals['condition'], rotation=30, ha='right', color='#8b949e')
    ax9.tick_params(colors='#8b949e')
    ax9.grid(True, alpha=0.2, color='#30363d', axis='y')
    for spine in ax9.spines.values():
        spine.set_color('#30363d')
    ax9.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=10)
else:
    ax9.text(
        0.5,
        0.5,
        'No distance data by weather condition',
        ha='center',
        va='center',
        fontsize=12,
        color='#8b949e'
    )
    ax9.set_axis_off()

# Save dashboard as PNG
plt.savefig('life_dashboard.png', dpi=300, facecolor='#0d1117', 
            bbox_inches='tight', edgecolor='none', pad_inches=0.2)
print("Dashboard saved as 'life_dashboard.png'")
print("\nDashboard Metrics:")
print(f"  • Total Distance: {total_distance:.1f} km")
print(f"  • Total Calories: {total_calories:,}")
print(f"  • Total Activities: {total_activities}")
print(f"  • Active Days: {active_days}")
