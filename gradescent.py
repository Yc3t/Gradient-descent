import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Define the computeCost function
def compute_cost(X, y, theta):
    m = len(y)
    J = np.sum((X.dot(theta) - y) ** 2) / (2 * m)
    return J

# Define the gradient descent function
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    theta_history = []

    for i in range(num_iters):
        error = X.dot(theta) - y
        theta = theta - (alpha / m) * (X.T.dot(error))
        theta_history.append(theta)
        J_history.append(compute_cost(X, y, theta))

    return np.array(theta_history), np.array(J_history)

# Sample data
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Gradient Descent Simulator"),
    dcc.Graph(id='gradient-plot'),
    html.Label("Initial theta_0"),
    dcc.Input(id='theta-0-input', type='number', value=0),
    html.Label("Initial theta_1"),
    dcc.Input(id='theta-1-input', type='number', value=0),
    html.Label("Learning rate (alpha)"),
    dcc.Slider(id='alpha-slider', min=0.001, max=0.3, step=0.001, value=0.01, marks={i/100: str(i/100) for i in range(1, 31)}),
    html.Label("Number of iterations"),
    dcc.Slider(id='iterations-slider', min=10, max=2000, step=10, value=100, marks={i: str(i) for i in range(100, 2001, 400)}),
])
@ app.callback(
    Output('gradient-plot', 'figure'),
    [Input('alpha-slider', 'value'),
     Input('iterations-slider', 'value'),
     Input('theta-0-input', 'value'),
     Input('theta-1-input', 'value')]
)
def update_figure(alpha, num_iters, theta_0, theta_1):
    # Check if theta_0 or theta_1 is None and provide a default value if so
    if theta_0 is None:
        theta_0 = 0
    if theta_1 is None:
        theta_1 = 0

    # Convert theta_0 and theta_1 to float to avoid TypeError
    theta_0 = float(theta_0)
    theta_1 = float(theta_1)
    # Perform gradient descent
    initial_theta = np.array([theta_0, theta_1])
    theta_history, J_history = gradient_descent(X, y, initial_theta, alpha, num_iters)

    # Create subplots: one for 3d surface, one for contour
    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{'type': 'surface'}, {'type': 'contour'}]],
                        subplot_titles=('3D Cost Function', 'Contour Plot'))

    # Define the cost function data
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
    J_vals = np.array([[compute_cost(X, y, [t0, t1]) for t1 in theta1_vals] for t0 in theta0_vals])

    # Add surface plot for cost function
    fig.add_trace(go.Surface(x=T0, y=T1, z=J_vals, colorscale='Viridis', name='Cost Function'), row=1, col=1)

    # Add contour plot for cost function
    fig.add_trace(go.Contour(x=theta0_vals, y=theta1_vals, z=J_vals, colorscale='Viridis', name='Cost Contour'), row=1, col=2)

    # Add gradient descent path to both plots
    fig.add_trace(go.Scatter3d(x=theta_history[:, 0], y=theta_history[:, 1], z=J_history, mode='lines+markers', marker=dict(color='red'), name='Gradient Descent Path'), row=1, col=1)
    fig.add_trace(go.Scatter(x=theta_history[:, 0], y=theta_history[:, 1], mode='lines+markers', marker=dict(color='red'), name='Gradient Descent Path'), row=1, col=2)

    # Update layout for a cleaner look
    fig.update_layout(autosize=True, title='Gradient Descent Visualization')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
