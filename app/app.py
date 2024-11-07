import streamlit as st
import folium
from streamlit_folium import st_folium  # Import st_folium for embedding Folium maps in Streamlit

st.title("Traffic Congestion Monitoring")
st.write("This is a test of your Streamlit setup with Folium integration.")

# Create a Folium map
m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
folium.Marker([40.7128, -74.0060], tooltip="New York").add_to(m)

# Display the Folium map in Streamlit
st_folium(m, width=700, height=500)
