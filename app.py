import pandas as pd
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and dataset
try:
    model_data = joblib.load('air_pollution_model.pkl')
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    data = pd.read_csv('global_air_pollution_dataset.csv')
except FileNotFoundError:
    tk.Tk().withdraw()  # Hide main window
    messagebox.showerror("Error", "Required files (air_pollution_model.pkl or global_air_pollution_dataset.csv) not found.")
    exit()

# Color mapping for AQI Category
AQI_COLORS = {
    'Good': '#00FF00',  # Green
    'Moderate': '#FFFF00',  # Yellow
    'Unhealthy for Sensitive Groups': '#FFA500',  # Orange
    'Unhealthy': '#FF0000',  # Red
    'Very Unhealthy': '#800080',  # Purple
    'Hazardous': '#8B0000',  # Dark Red
}

def update_results(city, country=None, override_country=None):
    # Clear previous results
    for widget in result_frame.winfo_children():
        widget.destroy()

    # Check if city exists in dataset
    city_exists = data[data['City'].str.lower() == city.lower()]
    
    if city_exists.empty:
        messagebox.showerror("Error", f"No city named '{city}' exists in the dataset.")
        return

    # Use override_country if provided (from user confirmation)
    if override_country:
        city_data = data[(data['City'].str.lower() == city.lower()) & (data['Country'] == override_country)]
    else:
        # Filter by selected country if specified
        if country and country != "All Countries":
            city_data = data[(data['City'].str.lower() == city.lower()) & (data['Country'] == country)]
        else:
            city_data = data[data['City'].str.lower() == city.lower()]

    if city_data.empty and country != "All Countries":
        # City exists in another country
        correct_country = city_exists['Country'].iloc[0]
        response = messagebox.askyesno(
            "City Not Found in Selected Country",
            f"The city '{city}' is not in '{country}' but exists in '{correct_country}'. "
            "Would you like to see results for this city in '{correct_country}'?"
        )
        if response:
            update_results(city, override_country=correct_country)
        return

    # Process city data
    try:
        city_row = city_data.iloc[0]
        aqi_values = city_row[['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
        max_aqi = aqi_values.max()
        features = pd.DataFrame([aqi_values.tolist() + [max_aqi]], 
                               columns=['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value', 'Max AQI'])
        country_display = city_row['Country']
        aqi_value = city_row['AQI Value']
        dataset_aqi_category = city_row['AQI Category']
        city_display = city

        # Predict AQI Category
        prediction = model.predict(features)[0]
        predicted_aqi_category = label_encoder.inverse_transform([prediction])[0]

        # Display results in a table-like format
        labels = [
            ("City", city_display),
            ("Country", country_display),
            ("AQI Value", f"{aqi_value:.2f}"),
            ("AQI Category (Dataset)", dataset_aqi_category),
            ("AQI Category (Predicted)", predicted_aqi_category),
            ("CO AQI Value", f"{features['CO AQI Value'].iloc[0]:.2f}"),
            ("Ozone AQI Value", f"{features['Ozone AQI Value'].iloc[0]:.2f}"),
            ("NO2 AQI Value", f"{features['NO2 AQI Value'].iloc[0]:.2f}"),
            ("PM2.5 AQI Value", f"{features['PM2.5 AQI Value'].iloc[0]:.2f}")
        ]

        for i, (label, value) in enumerate(labels):
            ttk.Label(result_frame, text=label + ":", font=("Arial", 10, "bold")).grid(row=i, column=0, sticky="e", padx=5, pady=2)
            ttk.Label(result_frame, text=value, font=("Arial", 10), 
                      background=AQI_COLORS.get(value, "#FFFFFF") if label == "AQI Category (Dataset)" else "#FFFFFF").grid(row=i, column=1, sticky="w", padx=5, pady=2)
    except (KeyError, IndexError) as e:
        messagebox.showerror("Error", f"Error accessing data for '{city}': {str(e)}. The dataset may be corrupted or missing required columns.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred while processing data for '{city}': {str(e)}")

def check_air_quality():
    city = city_entry.get().strip()
    country = country_combobox.get()

    if not city:
        messagebox.showerror("Error", "Please enter a city name.")
        return

    update_results(city, country)

# Create GUI
root = tk.Tk()
root.title("Air Pollution Detection System")
root.geometry("400x400")

# Input frame
input_frame = ttk.Frame(root)
input_frame.pack(pady=10)

# Country dropdown
ttk.Label(input_frame, text="Select Country:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
countries = ['All Countries'] + sorted(data['Country'].dropna().unique().tolist())
country_combobox = ttk.Combobox(input_frame, values=countries, state="readonly", width=25)
country_combobox.set("All Countries")
country_combobox.grid(row=0, column=1, padx=5)

# City input
ttk.Label(input_frame, text="Enter City Name:", font=("Arial", 12)).grid(row=1, column=0, padx=5, pady=5)
city_entry = ttk.Entry(input_frame, width=28)
city_entry.grid(row=1, column=1, padx=5, pady=5)

# Check button
check_button = ttk.Button(input_frame, text="Check Air Quality", command=check_air_quality)
check_button.grid(row=2, column=0, columnspan=2, pady=10)

# Result frame
result_frame = ttk.Frame(root)
result_frame.pack(pady=10)

# Start GUI
root.mainloop()