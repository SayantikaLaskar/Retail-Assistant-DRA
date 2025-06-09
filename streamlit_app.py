# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import datetime

# Load data
@st.cache_data
def load_data():
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

class WalmartDataset(Dataset):
    def __init__(self, df):
        self.features = df[['Store_enc', 'Dept_enc', 'Temperature', 'Fuel_Price',
                            'CPI', 'Unemployment', 'WeekOfYear', 'Month', 'DayOfWeek']].values.astype(np.float32)
        self.targets = df['Weekly_Sales'].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DemandModel(nn.Module):
    def __init__(self, input_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, dataloader, epochs=5, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
    return model

def simulate_real_time_features(date, store_enc):
    np.random.seed(int(date.strftime('%Y%m%d')) + store_enc)
    doy = date.timetuple().tm_yday
    temp = 60 + 20 * np.sin(doy * 2 * np.pi / 365) + np.random.randn()
    fuel_price = 2.5 + 0.1 * np.random.randn()
    cpi = 200 + 5 * np.random.randn()
    unemployment = 5 + np.random.randn()
    return temp, fuel_price, cpi, unemployment

def predict(model, features):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        features = np.ascontiguousarray(features, dtype=np.float32)
        inputs = torch.from_numpy(features).to(device)
        preds = model(inputs).cpu().numpy()
    return preds

# --- Streamlit UI ---

st.title("🧠 AI-Driven Dynamic Restocking Assistant (DRA)")

data, le_store, le_dept = load_data()
if data is None or data.empty:
    st.stop()

stores = le_store.classes_
departments = le_dept.classes_

selected_store = st.selectbox("Select Store", stores)
selected_dept = st.selectbox("Select Department", departments)
selected_date = st.date_input("Select Date", value=datetime.date(2012, 10, 26))

store_enc = le_store.transform([selected_store])[0]
dept_enc = le_dept.transform([selected_dept])[0]

# Simulate features
temp, fuel_price, cpi, unemployment = simulate_real_time_features(selected_date, store_enc)
features = np.array([[
    store_enc, dept_enc, temp, fuel_price, cpi, unemployment,
    selected_date.isocalendar()[1],
    selected_date.month,
    selected_date.weekday()
]], dtype=np.float32)

# Train model
if st.button("Train Model"):
    st.write("Training model... please wait.")
    sample = data.sample(frac=0.05, random_state=42)
    dataset = WalmartDataset(sample)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = DemandModel()
    model = train_model(model, dataloader)
    st.session_state['model'] = model
    st.success("✅ Model trained and stored!")

# Predict
if 'model' in st.session_state:
    model = st.session_state['model']
    pred = predict(model, features)[0][0]
    st.metric(label=f"Predicted Weekly Sales for Store {selected_store} Dept {selected_dept}",
              value=f"${pred:,.2f}")

    if pred > 20000:
        priority = "High 🔥"
    elif pred > 5000:
        priority = "Medium ⚠️"
    else:
        priority = "Low ✅"
    st.markdown(f"**Restocking Priority:** {priority}")
else:
    st.info("Train the model to enable predictions.")

# Show historical sales
if st.checkbox("Show Historical Sales"):
    hist = data[(data['Store'] == selected_store) & (data['Dept'] == selected_dept)]
    st.line_chart(hist.set_index('Date')['Weekly_Sales'])
