# kNN
kNN Regression using Streamlit

README for Concrete Compressive Strength kNN Regression Model
 - Current as of 20 Feburary 2021

Modeled and deployed in streamlit as part of EN605.645, Spring 2021, under 
Dr. Stephyn Butcher

**Author**: Kyle Tucker-Davis

**Introduction**: This model seeks to deploy the code from programming assignment 3 
into a UI that allows exploration
of the work done in the module.

**Accompanying files**:

1. ktucke28.py (main file)
2. concrete_compressive_strength.csv (data file)
3. background.png (UI background)

**Sections**:

0. *Background* (see below for reference) changed to picture to show ease of UI use.
1.  *Sidebar*.  Sidebar allows the user to input values for each of the features in 
this module.  Of note, each slider's
min & max & average are that of the training data set.  Default is set to average.
2.  *"Learn about kNN!"* contains links to 3 different YouTube videos to allow users 
to further explore kNN.
It is also included to showcase streamlit functionality for both columns and embedding 
videos.  They are jokingly referenced by length.
3.  *Prediction* returns the query, k value, and model estimation.
4.  *"Interested in regression estimates for other k values"* plots a st.line_chart for 
prediction values using k=1-20.
5.  *"Interested in Correlation/Heatmap?"* shows a heatmap of all features to all the user 
to better understand feature correlation for the 8 distinct features.
6.  *"See the code!"* includes the primary methods and calls used.  It does not include UI 
generation.


**References**:

Note - given eased restrictions on collaboration for this module, several external 
sources were used, as noted below.

 - Background image (before modification): https://wallpapercave.com/wp/ZTRsWNx.jpg

 - Code for using local image as background: 
https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/5

 - Streamlit documentation: https://docs.streamlit.io/en/stable/
