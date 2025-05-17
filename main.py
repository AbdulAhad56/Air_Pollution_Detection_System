
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load Dataset
df = pd.read_csv('global_air_pollution_dataset.csv')

# Step 2: Preprocessing
df = df[['Country', 'City', 'PM2.5 AQI Value']]
df.dropna(inplace=True)
df['PM2.5 AQI Value'] = pd.to_numeric(df['PM2.5 AQI Value'], errors='coerce')

# Step 3: Create Labels
df['Label'] = df['PM2.5 AQI Value'].apply(lambda x: 0 if x < 100 else 1)

# Step 4: Features and Labels
X = df[['PM2.5 AQI Value']]
y = df['Label']

# Step 5: Split into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Define Prediction Function
def predict_survivability(location):
    location = location.strip().lower()
    
    # Check if location matches a country
    countries = df['Country'].str.lower().unique()
    
    if location in countries:
        # Aggregate for the country
        country_data = df[df['Country'].str.lower() == location]
        avg_pm_value = country_data['PM2.5 AQI Value'].mean()
        
        prediction = model.predict([[avg_pm_value]])[0]
        status = "Survivable ✅" if prediction == 0 else "Not Survivable ❌"
        country_name = country_data.iloc[0]['Country']
        print(f"Country: {country_name} | Average PM2.5 AQI: {avg_pm_value:.2f} | Prediction: {status}")
        
    else:
        # Search for a city match
        search = df[df['City'].str.lower() == location]
        if search.empty:
            print(f"No data found for '{location}'.")
        else:
            row = search.iloc[0]
            pm_value = row['PM2.5 AQI Value']
            prediction = model.predict([[pm_value]])[0]
            status = "Survivable ✅" if prediction == 0 else "Not Survivable ❌"
            print(f"City: {row['City']}, {row['Country']} | PM2.5 AQI: {pm_value} | Prediction: {status}")

# Step 8: Main
if __name__ == "__main__":
    print("=== Air Pollution Detection System ===")
    location = input("Enter a city or country name: ")
    predict_survivability(location)