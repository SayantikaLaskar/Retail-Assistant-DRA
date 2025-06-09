# Filename: app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import datetime

# 1. Load and preprocess data
@st.cache_data
def load_data():
    # Update paths as needed; Kaggle Walmart dataset files
    features = pd.read_csv("features.csv")
    stores = pd.read_csv("stores.csv")
    train = pd.read_csv("train.csv")

    # Merge datasets on Store, Date, Dept
    data = pd.merge(train, features, on=['Store', 'Date'], how='left')  # Remove 'Dept' here
    data = pd.merge(data, stores, on='Store', how='left')


    # Convert Date to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(['Store', 'Dept', 'Date'])

    # Label encode categorical columns
    le_store = LabelEncoder()
    le_dept = LabelEncoder()
    data['Store_enc'] = le_store.fit_transform(data['Store'])
    data['Dept_enc'] = le_dept.fit_transform(data['Dept'])
    
    # Fill missing values (e.g. Temperature)
    data['Temperature'] = data['Temperature'].fillna(method='ffill')

    # Create features: week of year, month, day of week
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)
    data['Month'] = data['Date'].dt.month
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    return data, le_store, le_dept

# 2. Dataset class for PyTorch
class WalmartDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.features = df[['Store_enc', 'Dept_enc', 'Temperature', 'Fuel_Price', 
                           'CPI', 'Unemployment', 'WeekOfYear', 'Month', 'DayOfWeek']].values.astype(np.float32)
        self.targets = df['Weekly_Sales'].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y

# 3. Simple Feedforward model
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

# 4. Training function
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

# 5. Predict function
def predict(model, features):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features.astype(np.float32))
        preds = model(inputs).numpy()
    return preds

# 6. Simulate real-time data inputs (weather/trends) — replace with real API calls later
def simulate_real_time_features(date, store_enc):
    # Mock example: small random fluctuation for temperature and fuel price
    np.random.seed(int(date.strftime('%Y%m%d')) + store_enc)
    date = pd.Timestamp(date)
    temp = 60 + 20 * np.sin(date.dayofyear * 2 * np.pi / 365) + np.random.randn()
    fuel_price = 2.5 + 0.1 * np.random.randn()
    cpi = 200 + 5 * np.random.randn()
    unemployment = 5 + np.random.randn()
    return temp, fuel_price, cpi, unemployment

# --- Streamlit App ---

st.title("🧠 AI-Driven Dynamic Restocking Assistant (DRA)")

data, le_store, le_dept = load_data()

stores = le_store.classes_
departments = le_dept.classes_

# User inputs
selected_store = st.selectbox("Select Store", stores)
selected_dept = st.selectbox("Select Department", departments)
selected_date = st.date_input("Select Date", value=datetime.date(2012, 10, 26))

store_enc = le_store.transform([selected_store])[0]
dept_enc = le_dept.transform([selected_dept])[0]

# Prepare input feature vector for prediction
temp, fuel_price, cpi, unemployment = simulate_real_time_features(selected_date, store_enc)

features = np.array([[
    store_enc, dept_enc, temp, fuel_price, cpi, unemployment,
    selected_date.isocalendar()[1],  # WeekOfYear
    selected_date.month,
    selected_date.weekday()
]], dtype=np.float32)

# Train or load model (for demo: train on a small subset to speed up)
if st.button("Train Model (takes ~30 seconds)"):
    train_sample = data.sample(frac=0.05, random_state=42)
    dataset = WalmartDataset(train_sample)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = DemandModel()
    model = train_model(model, dataloader, epochs=5)
    st.session_state['model'] = model
    st.success("Model trained and saved in session!")

if 'model' in st.session_state:
    model = st.session_state['model']
    pred = predict(model, features)[0][0]
    st.metric(label=f"Predicted Weekly Sales for Store {selected_store} Dept {selected_dept}", value=f"{pred:,.2f}")

    # Simple restocking priority suggestion
    if pred > 20000:
        restock_priority = "High 🔥"
    elif pred > 5000:
        restock_priority = "Medium ⚠️"
    else:
        restock_priority = "Low ✅"
    st.write(f"Suggested Restocking Priority: **{restock_priority}**")
else:
    st.info("Train the model first by clicking the button above.")

# Bonus: Show some sales history for selected store/dept
if st.checkbox("Show Historical Sales"):
    hist_data = data[(data['Store'] == selected_store) & (data['Dept'] == selected_dept)]
    st.line_chart(hist_data.set_index('Date')['Weekly_Sales'])

