# Filename: app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import datetime
import os

# 1. Load and preprocess data
@st.cache_data
def load_data():
    required_files = ["features.csv", "stores.csv", "train.csv"]
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"Missing file: `{file}`")
            st.stop()
        if os.stat(file).st_size == 0:
            st.error(f"The file `{file}` is empty.")
            st.stop()

    features = pd.read_csv("features.csv")
    stores = pd.read_csv("stores.csv")
    train = pd.read_csv("train.csv")

    data = pd.merge(train, features, on=['Store', 'Date'], how='left')
    data = pd.merge(data, stores, on='Store', how='left')

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(['Store', 'Dept', 'Date'])

    le_store = LabelEncoder()
    le_dept = LabelEncoder()
    data['Store_enc'] = le_store.fit_transform(data['Store'])
    data['Dept_enc'] = le_dept.fit_transform(data['Dept'])

    data['Temperature'] = data['Temperature'].fillna(method='ffill')

    data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)
    data['Month'] = data['Date'].dt.month
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    return data, le_store, le_dept

# 2. Dataset class
class WalmartDataset(Dataset):
    def __init__(self, df):
        self.features = torch.tensor(df[['Store_enc', 'Dept_enc', 'Temperature', 'Fuel_Price', 
                                         'CPI', 'Unemployment', 'WeekOfYear', 'Month', 'DayOfWeek']].values, dtype=torch.float32)
        self.targets = torch.tensor(df['Weekly_Sales'].values.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# 3. Model
class DemandModel(nn.Module):
    def __init__(self, input_dim=9):
        super(DemandModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# 4. Training

def train_model(model, dataloader, epochs=5, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    return model

# 5. Prediction

def predict(model, features):
    model.eval()
    with torch.no_grad():
        features_array = np.array(features, dtype=np.float32)
        if not np.all(np.isfinite(features_array)):
            raise ValueError("Input features contain NaNs or infinite values.")

        inputs = torch.tensor(features_array, dtype=torch.float32)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        preds = model(inputs).detach().numpy()
    return preds

# 6. Simulated real-time features

def simulate_real_time_features(date, store_enc):
    np.random.seed(int(date.strftime('%Y%m%d')) + store_enc)
    date = pd.Timestamp(date)
    temp = 60 + 20 * np.sin(date.dayofyear * 2 * np.pi / 365) + np.random.randn()
    fuel_price = 2.5 + 0.1 * np.random.randn()
    cpi = 200 + 5 * np.random.randn()
    unemployment = 5 + np.random.randn()
    return temp, fuel_price, cpi, unemployment

# --- Streamlit UI ---

st.title("🧠 AI-Driven Dynamic Restocking Assistant (DRA)")

# Load data and encoders
data, le_store, le_dept = load_data()

stores = le_store.classes_
departments = le_dept.classes_

selected_store = st.selectbox("Select Store", stores)
selected_dept = st.selectbox("Select Department", departments)
selected_date = st.date_input("Select Date", value=datetime.date(2012, 10, 26))

store_enc = le_store.transform([selected_store])[0]
dept_enc = le_dept.transform([selected_dept])[0]

temp, fuel_price, cpi, unemployment = simulate_real_time_features(selected_date, store_enc)

features = [[
    int(store_enc), int(dept_enc), float(temp), float(fuel_price),
    float(cpi), float(unemployment),
    int(selected_date.isocalendar()[1]),
    int(selected_date.month),
    int(selected_date.weekday())
]]

st.write("Features used for prediction:", features)

if st.button("Train Model (takes ~30 seconds)"):
    train_sample = data.sample(frac=0.05, random_state=42)
    train_sample = train_sample[[
        'Store_enc', 'Dept_enc', 'Temperature', 'Fuel_Price', 
        'CPI', 'Unemployment', 'WeekOfYear', 'Month', 'DayOfWeek', 'Weekly_Sales']]
    train_sample = train_sample.dropna()
    train_sample = train_sample[np.isfinite(train_sample).all(axis=1)]

    if train_sample.empty:
        st.error("Cleaned training sample is empty. Check for NaNs or invalid values in input data.")
        st.stop()

    dataset = WalmartDataset(train_sample)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = DemandModel()
    model = train_model(model, dataloader, epochs=5)
    st.session_state['model'] = model
    st.success("Model trained and saved in session!")

if 'model' in st.session_state:
    model = st.session_state['model']
    try:
        pred = predict(model, features)[0][0]
        st.metric(label=f"Predicted Weekly Sales for Store {selected_store} Dept {selected_dept}", value=f"{pred:,.2f}")

        if pred > 20000:
            restock_priority = "High 🔥"
        elif pred > 5000:
            restock_priority = "Medium ⚠️"
        else:
            restock_priority = "Low ✅"
        st.write(f"Suggested Restocking Priority: **{restock_priority}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Train the model first by clicking the button above.")

if st.checkbox("Show Historical Sales"):
    hist_data = data[(data['Store'] == selected_store) & (data['Dept'] == selected_dept)]
    if not hist_data.empty:
        st.line_chart(hist_data.set_index('Date')['Weekly_Sales'])
    else:
        st.warning("No historical sales data available for this selection.")
