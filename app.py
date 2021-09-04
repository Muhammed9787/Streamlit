import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage
from pages import predictions ,sir,dashboard# import your pages here

# Create an instance of the app 
app = MultiPage()

# Title of the main page
# display = Image.open('Logo.png')
# display = np.array(display)
# # st.image(display, width = 400)
# # st.title("Data Storyteller Application")
# col1, col2 = st.columns(2)
# col1.image(display, width = 400)
# col2.title("Data Storyteller Application")

# Add all your application here
#app.add_page("Upload Data", data_upload.app)
#app.add_page("Change Metadata", metadata.app)
#app.add_page("Machine Learning", machine_learning.app)
app.add_page("Dashboard",dashboard.app)
app.add_page("Covid-19 Predictions in Sudan",predictions.app)
app.add_page("SIR",sir.app)



# The main app
app.run()
