import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model import *
from data import *
from adjust import *

# Data control
FILE = "discharge.csv"
RCU = 80 # real cycle until 80 cycles (168 cycles in total in this example)
PRED_STEP = 3 # next cycle steps to be predicted (`-1` stands for the rest of all cycles)

# Training setting
epochs = 300
batch = 20
lr = 1e-3

# Load data
dataset = DischargeDataSet(FILE)

# Set interest of research
#   - real step of cycle until `RCU` -> for training
#   - steps to be predicted based on trained model -> interest of digital twin
dataset.set_real_predict_range(real_step=RCU, pred_step=PRED_STEP)
simcapa_in_real, simcapa_in_pred = dataset.get_sim_capacities()
realcapa_in_real, realcapa_in_pred = dataset.get_real_capacities()

# Supportive data to be used (e.g. calculation, plotting)
real_capa = dataset.get_real_capacity()
charge_index = dataset.get_index()
train_dataset, test_dataset = dataset()
MAXCAPA = real_capa.max()
MINCAPA = real_capa.min()

# Create pytorch DataLoader object for training and testing
train_dataloader = DataLoader(train_dataset, batch_size = batch, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size = batch, shuffle=False)


# Init neural network, loss function and optimizer
model = NeuralNetwork().double()
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# Training
train(model=model, epochs=epochs, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)

# Evaluate itself, only for plotting the approximation
preds = evaluate(model=model, dataloader=train_dataloader)
X_twin_real = simcapa_in_real + preds

# Evaluate unkown `twin value` for prediction
preds = evaluate(model=model, dataloader=test_dataloader)
X_twin_pred = simcapa_in_pred + weight(preds)



# Plot
fig = go.Figure()

# Speration line
fig.add_trace(go.Scatter(x=[RCU, RCU], 
                         y=[MINCAPA, MAXCAPA],
                         mode='lines',
                         line=dict(color='tomato', width=2),
                         name=f'Seperation of past data and predicted value'))


# KNOWN: Hybraid twin
fig.add_trace(go.Scatter(x=simcapa_in_real.index, 
                         y=X_twin_real,
                         mode='lines',
                         name=f'Data-driven Capacity Approximation',
                         line=dict(color='green', width=3)))
# KNOWN: Simulated
fig.add_trace(go.Scatter(x=realcapa_in_real.index, 
                         y=simcapa_in_real,
                         mode='lines',
                         name=f'Simulation based Capacity (for Training)',
                         line=dict(color='navy', 
                                   width=3,
                                   dash='dash')))

# Unkown: Simulated
fig.add_trace(go.Scatter(x=realcapa_in_pred.index, 
                         y=simcapa_in_pred,
                         mode='lines',
                         name=f'Simulation based Capacity (for Testing)',
                         line=dict(color='orange', 
                                   width=3,
                                   dash='dash')))
# Known: Real Value
fig.add_trace(go.Scatter(x=real_capa.index, 
                         y=real_capa,
                         mode='markers',
                         marker=dict(
                              size=4,
                              color='grey',
                              symbol='cross'
                                 ),
                         name=f'Real Capacity (Observed)',
                         line_color='navy'))

# UNKNOWN: hybrid twin = NN(sim_res) + sim_res -> to predict unkown real value
fig.add_trace(go.Scatter(x=simcapa_in_pred.index, 
                         y=X_twin_pred,
                         name=f'Data-driven Capacity Prediction',
                         mode='markers',
                         marker=dict(
                              size=5,
                              color='firebrick',
                              symbol='diamond'
                                 ),
                         line=dict(color='firebrick', width=5)))

fig.update_layout(
    title="Comparison of hybrid twin with other models",
    xaxis_title="Cycles",
    yaxis_title="Capacity in Ahr")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.95,
    xanchor="left",
    x=0.77
))
fig.update_layout({'plot_bgcolor': '#f2f8fd',
                  'paper_bgcolor': 'white',}, 
                   template='plotly_white')

fig.show()