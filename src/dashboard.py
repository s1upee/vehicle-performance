import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Load processed data
df = pd.read_csv("data/processed_vehicle_data.csv")

# Initialize Dash app
app = dash.Dash(__name__)

# Create figures
fig_acceleration = px.line(df, x='time', y='acceleration', title='Acceleration Over Time')
fig_braking = px.line(df, x='time', y='brake_force', title='Braking Force Over Time')
fig_steering = px.line(df, x='time', y='steering_angle', title='Steering Angle Over Time')

# Layout
app.layout = html.Div(children=[
    html.H1("Vehicle Performance Dashboard"),
    dcc.Graph(figure=fig_acceleration),
    dcc.Graph(figure=fig_braking),
    dcc.Graph(figure=fig_steering),
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
