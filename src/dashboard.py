import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import time

# Load initial data
df = pd.read_csv("data/processed_vehicle_data.csv")

def load_data():
    """Reloads the latest processed vehicle data."""
    return pd.read_csv("data/processed_vehicle_data.csv")

# Initialize Dash app
app = dash.Dash(__name__)

# Layout with dropdown filter and real-time updates
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
    dcc.Interval(
        id='interval-component',
        interval=5000,  # Update every 5 seconds
        n_intervals=0
    ),
    dcc.Graph(id='acceleration-plot'),
    dcc.Graph(id='braking-plot'),
    dcc.Graph(id='steering-plot'),
])

# Callback to update graphs in real-time
@app.callback(
    [Output('acceleration-plot', 'figure'),
     Output('braking-plot', 'figure'),
     Output('steering-plot', 'figure')],
    [Input('interval-component', 'n_intervals'),
     Input('filter-dropdown', 'value')]
)
def update_graphs(n_intervals, filter_value):
    df = load_data()
    
    if filter_value == 'anomalies':
        df = df[df['hard_braking'] | df['sudden_acceleration'] | df['sharp_turn']]
    
    fig_acceleration = px.line(df, x='time', y='acceleration', title='Acceleration Over Time', markers=True)
    fig_acceleration.add_scatter(x=df[df['sudden_acceleration']]['time'], y=df[df['sudden_acceleration']]['acceleration'], mode='markers', marker=dict(color='red', size=8), name='Sudden Acceleration')
    
    fig_braking = px.line(df, x='time', y='brake_force', title='Braking Force Over Time', markers=True)
    fig_braking.add_scatter(x=df[df['hard_braking']]['time'], y=df[df['hard_braking']]['brake_force'], mode='markers', marker=dict(color='red', size=8), name='Hard Braking')
    
    fig_steering = px.line(df, x='time', y='steering_angle', title='Steering Angle Over Time', markers=True)
    fig_steering.add_scatter(x=df[df['sharp_turn']]['time'], y=df[df['sharp_turn']]['steering_angle'], mode='markers', marker=dict(color='red', size=8), name='Sharp Turn')
    
    return fig_acceleration, fig_braking, fig_steering

if __name__ == "__main__":
    app.run_server(debug=True)
