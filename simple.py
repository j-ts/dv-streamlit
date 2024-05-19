import streamlit as st
import pandas as pd
import altair as alt
import numpy as np


APP_TITLE = 'UK Young Adults Demographics: A Decade in Data '
APP_SUB_TITLE = 'Source: The National Archives under [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)'
st.set_page_config(APP_TITLE)
st.title(APP_TITLE)
st.caption(APP_SUB_TITLE)

with st.echo():
    st.write('This code will be printed.')

"""def main():
    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)"""

# Load synthetic data
@st.cache_data
def load_data():
    # Generate synthetic data
    locations = ['England', 'Scotland', 'Wales', 'Northern Ireland']
    ages = list(range(0, 101))  # Ages 0 to 100
    genders = ['Male', 'Female']
    years = [2011, 2022]

    data = []

    for year in years:
        for location in locations:
            for age in ages:
                for gender in genders:
                    population_density = np.random.randint(50, 500)  # Random population density
                    data.append([year, location, age, gender, population_density])

    df = pd.DataFrame(data, columns=['Year', 'Location', 'Age', 'Gender', 'Population Density'])
    return df

df = load_data()

# Title
# st.title("UK Population Density Dashboard")

# User selection for year and gender beside each other
col1, col2 = st.columns(2)
with col1:
    year = st.radio("Select Year", options=[2011, 2022])
with col2:
    male_selected = st.checkbox("Male", value=True)
    female_selected = st.checkbox("Female", value=True)

# Placeholder for the chart
title_placeholder = st.empty()
chart_placeholder = st.empty()

age_range = st.slider("Select Age", 18, 35, 18)

# Determine selected genders
selected_genders = []
if male_selected:
    selected_genders.append('Male')
if female_selected:
    selected_genders.append('Female')

# Filter data based on user selection
filtered_data = df[
    (df['Year'] == year) & 
    (df['Gender'].isin(selected_genders)) &
    (df['Age'] == age_range)
]

# Plot population density by location
title_placeholder.subheader("Population Density by Geographic Location")

# Check if any gender is selected to separate bars by gender
if selected_genders:
    chart1 = alt.Chart(filtered_data).mark_bar().encode(
        x='Location',
        y='Population Density',
        color='Gender'
    )
else:
    selected_genders = ['Male', 'Female']
    filtered_data = df[(df['Year'] == year) & (df['Gender'].isin(selected_genders)) & (df['Age'] == age_range)]

    # If no gender is selected, show full chart
    chart1 = alt.Chart(filtered_data).mark_bar().encode(
        x='Location',
        y='Population Density',
        color='Gender'
    )

chart_placeholder.altair_chart(chart1, use_container_width=True)

