import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Load processed data
df = pd.read_csv("data/processed_vehicle_data.csv")

# Highlight anomalies in the dataset
df['anomaly_color'] = df['hard_braking'] | df['sudden_acceleration'] | df['sharp_turn']

# Create figures with anomaly markers
fig_acceleration = px.line(df, x='time', y='acceleration', title='Acceleration Over Time', markers=True)
fig_acceleration.add_scatter(x=df[df['sudden_acceleration']]['time'], y=df[df['sudden_acceleration']]['acceleration'], mode='markers', marker=dict(color='red', size=8), name='Sudden Acceleration')

fig_braking = px.line(df, x='time', y='brake_force', title='Braking Force Over Time', markers=True)
fig_braking.add_scatter(x=df[df['hard_braking']]['time'], y=df[df['hard_braking']]['brake_force'], mode='markers', marker=dict(color='red', size=8), name='Hard Braking')

fig_steering = px.line(df, x='time', y='steering_angle', title='Steering Angle Over Time', markers=True)
fig_steering.add_scatter(x=df[df['sharp_turn']]['time'], y=df[df['sharp_turn']]['steering_angle'], mode='markers', marker=dict(color='red', size=8), name='Sharp Turn')

# Layout with dropdown to filter anomalies
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1("Vehicle Performance Dashboard"),
    dcc.Dropdown(
        id='filter-dropdown',
        options=[
            {'label': 'Show All Data', 'value': 'all'},
            {'label': 'Show Only Anomalies', 'value': 'anomalies'}
        ],
        value='all',
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Graph(id='acceleration-plot', figure=fig_acceleration),
    dcc.Graph(id='braking-plot', figure=fig_braking),
    dcc.Graph(id='steering-plot', figure=fig_steering),
])

if __name__ == "__main__":
    app.run_server(debug=True)
