import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards


COUNTRY_COLORS = {
    "ENGLAND": ["#ffffff", "#fff7f0", "#ffefdf", "#ffe7cf", "#ffdfbe", "#ffd6a5", "#ffcd83", "#ffc461", "#ffbc3f", "#ffb31e"],
    "NORTHERN IRELAND": ["#ffffff", "#f2ffe7", "#e4ffcf", "#d7ffb6", "#caff9e", "#bdf685", "#afe06d", "#a2cb55", "#95b63c", "#87a124"],
    "SCOTLAND": ["#ffffff", "#ffe6e6", "#ffcccc", "#ffb3b3", "#ff9999", "#ff8080", "#ff6666", "#ff4d4d", "#ff3333", "#ff1a1a"],
    "WALES": ["#ffffff", "#ffffeb", "#ffffd6", "#ffffc2", "#fffdad", "#fffa99", "#fff785", "#fff471", "#fff05d", "#ffec49"]
}


APP_TITLE = 'UK Young Adults Demographics: A Decade in Data '
APP_SUB_TITLE = ''
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.write("#")
st.title(APP_TITLE)
st.caption(APP_SUB_TITLE)

st.markdown('This code will be printed.')
st.divider()

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


@st.cache_data
def plot_animated_population_pie_chart(df):
    # Aggregate the population by country and year
    df_grouped = df.groupby(['Year', 'name'])['Population'].sum().reset_index()

    # Get unique years
    years = df_grouped['Year'].unique()

    # Initialize figure
    fig = go.Figure()

    # Add traces, one for each frame
    for year in years:
        df_year = df_grouped[df_grouped['Year'] == year]
        fig.add_trace(go.Pie(
            labels=df_year['name'],
            values=df_year['Population'],
            name='',
            marker=dict(colors=[COUNTRY_COLORS[country][5] for country in df_year['name']]),
            visible=(year == years[0])  # Only the first year is visible initially
        ))

    # Create frames
    frames = []
    for year in years:
        frame = go.Frame(
            data=[go.Pie(
                labels=df_grouped[df_grouped['Year'] == year]['name'],
                values=df_grouped[df_grouped['Year'] == year]['Population'],
                marker=dict(colors=[COUNTRY_COLORS[country][5] for country in df_grouped[df_grouped['Year'] == year]['name']])
            )],
            name=str(year)
        )
        frames.append(frame)
    fig.frames = frames

    fig.update_layout(
        updatemenus=[{
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Year:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{
                'args': [[year], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}],
                'label': str(year),
                'method': 'animate'
            } for year in years]
        }]
    )
    st.subheader('Population Dynamics by Country')
    fig.update_layout(showlegend=True,
        legend=dict(title='', traceorder='normal', orientation='h'))

    #fig.show()
    return fig

@st.cache_data
def plot_population_by_sex(df):
    # Group by name and sex, then sum the Population
    grouped = df.groupby(['name', 'sex'])['Population'].sum().reset_index()
    # Calculate the total population for each county
    total_population = grouped.groupby('name')['Population'].transform('sum')
    # Normalize the population values so that M + F = 100% for each county
    grouped['Population'] = (grouped['Population'] / total_population) * 100

    # Separate the data for males and females
    males = grouped[grouped['sex'] == 'M']
    females = grouped[grouped['sex'] == 'F']
    
    fig = go.Figure()
    # Add male bars
    for i, row in males.iterrows():
        country_name = row['name']
        sex = 'Male'
        color = COUNTRY_COLORS[country_name][4]
        text_color = 'gray' if country_name == 'WALES' else 'white'
        fig.add_trace(go.Bar(
            x=[row['Population']],
            y=[row['name']],
            name=f'{sex} – {country_name}',
            orientation='h',
            marker=dict(color=color),
            hoverinfo='text',
            width=0.7,
            hovertext=f'{country_name}<br>{sex}: {row["Population"]:.1f}%',
            #text=f'{sex}: {row["Population"]:.1f}% of {country_name}'
        ))

    
    # Add female bars
    for i, row in females.iterrows():
        country_name = row['name']
        sex = 'Female'
        color = COUNTRY_COLORS[country_name][8]
        text_color = 'gray' if country_name == 'WALES' else 'gray'
        fig.add_trace(go.Bar(
            x=[row['Population']],
            y=[row['name']],
            width=0.7,
            name=f'{sex} – {country_name}',
            orientation='h',
            marker=dict(color=color),
            hoverinfo='text',
            hovertext=f'{country_name}<br>{sex}: {row["Population"]:.1f}%',
            hoverlabel={"bgcolor":'white'}
        ))
    st.subheader('Sex Distribution by Country')
    # Set the title and labels
    fig.update_layout(
        #title='Normalized Population by Sex in 2011',
        xaxis_title='Population (%)',
        yaxis_title='',
        #barmode='stack',
        legend=dict(title='', traceorder='normal', orientation='v'),)
    #fig.show()
    return fig

@st.cache_data
def plot_animated_population_by_sex(df):
    # Aggregate the population by country, sex, and year
    df_grouped = df.groupby(['Year', 'name', 'sex'])['Population'].sum().reset_index()
    
    total_population = df_grouped.groupby(['Year', 'name'])['Population'].transform('sum')
    # Normalize the population values so that M + F = 100% for each county
    
    df_grouped['Population'] = (df_grouped['Population'] / total_population) * 100
    
    # Get unique years
    years = df_grouped['Year'].unique()

    # Initialize figure
    fig = go.Figure()

    # Add traces, one for each frame (initial year)
    initial_year = years[0]
    df_year = df_grouped[df_grouped['Year'] == initial_year]
    
    # Separate the data for males and females for the initial year
    males = df_year[df_year['sex'] == 'M']
    females = df_year[df_year['sex'] == 'F']
    
    for i, row in males.iterrows():
        country_name = row['name']
        sex = 'Male'
        color = COUNTRY_COLORS[country_name][4]
        fig.add_trace(go.Bar(
            x=[row['Population']],
            y=[row['name']],
            name=f'{sex} – {country_name}',
            orientation='h',
            marker=dict(color=color),
            hoverinfo='text',
            width=0.7,
            hovertext=f'{country_name}<br>{sex}: {row["Population"]:.1f}%',
            visible=(row['Year'] == initial_year)  # Only the first year is visible initially
        ))

    for i, row in females.iterrows():
        country_name = row['name']
        sex = 'Female'
        color = COUNTRY_COLORS[country_name][8]
        fig.add_trace(go.Bar(
            x=[row['Population']],
            y=[row['name']],
            width=0.7,
            name=f'{sex} – {country_name}',
            orientation='h',
            marker=dict(color=color),
            hoverinfo='text',
            hovertext=f'{country_name}<br>{sex}: {row["Population"]:.1f}%',
            visible=(row['Year'] == initial_year)  # Only the first year is visible initially
        ))

    # Create frames
    frames = []
    for year in years:
        df_year = df_grouped[df_grouped['Year'] == year]
        males = df_year[df_year['sex'] == 'M']
        females = df_year[df_year['sex'] == 'F']
        
        frame_data = []
        for i, row in males.iterrows():
            country_name = row['name']
            sex = 'Male'
            color = COUNTRY_COLORS[country_name][4]
            frame_data.append(go.Bar(
                x=[row['Population']],
                y=[row['name']],
                name=f'{sex} – {country_name}',
                orientation='h',
                marker=dict(color=color),
                hoverinfo='text',
                width=0.7,
                hovertext=f'{country_name}<br>{sex}: {row["Population"]:.1f}%',
            ))
        
        for i, row in females.iterrows():
            country_name = row['name']
            sex = 'Female'
            color = COUNTRY_COLORS[country_name][8]
            frame_data.append(go.Bar(
                x=[row['Population']],
                y=[row['name']],
                width=0.7,
                name=f'{sex} – {country_name}',
                orientation='h',
                marker=dict(color=color),
                hoverinfo='text',
                hovertext=f'{country_name}<br>{sex}: {row["Population"]:.1f}%',
            ))

        frames.append(go.Frame(data=frame_data, name=str(year)))
    fig.frames = frames

    fig.update_layout(
        updatemenus=[{
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Year:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{
                'args': [[year], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}],
                'label': str(year),
                'method': 'animate'
            } for year in years]
        }]
    )
    st.subheader('Sex Distribution by Country')
    fig.update_layout(
        xaxis_title='Population (%)',
        yaxis_title='',
        legend=dict(title='', traceorder='normal', orientation='v'),
        showlegend=True
    )

    return fig

@st.cache_data
def calculate_population_change(df):
    df_2011, df_2022 = df[df['Year'] == 2011], df[df['Year'] == 2022]
    pop_2011, pop_2022 = df_2011.groupby('name')['Population'].sum().reset_index()['Population'].sum(), df_2022.groupby('name')['Population'].sum().reset_index()['Population'].sum()
    delta = (pop_2022 - pop_2011) / pop_2011 * 100
    return pop_2022, delta


pie_df = pd.read_csv('/Users/jts/Documents/college/git-hub/dv-streamlit/test_pie.csv')


col = st.columns((3.5, 5, 1.9), gap='medium')
with col[0]:
    st.plotly_chart(plot_animated_population_pie_chart(pie_df), use_container_width=True)

with col[1]:
    st.plotly_chart(plot_animated_population_by_sex(pie_df), use_container_width=True)
    #st.plotly_chart(plot_population_by_sex(pie_df), use_container_width=True)


@st.cache_data
def calculate_sex_ratio(df):
    # Group by year and sex, then sum the Population
    grouped = df.groupby(['Year', 'sex'])['Population'].sum().reset_index()

    # Separate the data for males and females
    males = grouped[grouped['sex'] == 'M']
    females = grouped[grouped['sex'] == 'F']

    # Merge male and female populations based on year
    merged = pd.merge(males, females, on='Year', suffixes=('_male', '_female'))

    # Calculate sex ratio
    merged['Sex Ratio'] = (merged['Population_male'] / merged['Population_female']).round(2)
    ratio_2011 = merged.loc[merged['Year'] == 2011, 'Sex Ratio'].values[0]
    ratio_2022 = merged.loc[merged['Year'] == 2022, 'Sex Ratio'].values[0]
    delta = ratio_2022 - ratio_2011
    return ratio_2022, delta

@st.cache_data
def metrics():
        col1 = st.columns(1)[0]
        pop, d_pop = calculate_population_change(pie_df)
        d_pop = round(d_pop, 2)
        col1.metric(label="Population", value="{:,}".format(pop), delta=f'{d_pop}%')


        col2 = st.columns(1)[0]
        sex_ratio, sex_del = calculate_sex_ratio(pie_df)
        sex_del = round(sex_del, 2)
        col2.metric(label="Male/Female Ratio", value=sex_ratio, delta=sex_del)
        style_metric_cards(border_left_color="#8C83D1") #0074D9


with col[2]:
    #donut_chart_greater = make_donut(4.4, 'Inbound', 'green')
    #st.write('**Population change**')
    #st.altair_chart(donut_chart_greater)
    #st.write(calculate_sex_ratio(pie_df))

    metrics()


    with st.expander('About', expanded=True):
        st.write('''
            - **Data**: The National Archives under [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
            - **Population**: Total Population of the targeted age group in 2022.
            - **Male/Female Ratio**: Ratio of males to females in the targeted age group.
            ''')





st.markdown("<h2> </h2>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: black;'>Population Density by Geographic Location</h2>",
    unsafe_allow_html=True)

# User selection for year and gender beside each other
col1, col2 = st.columns(2)
with col1:
    year = st.radio("Select Year", options=[2011, 2022])
with col2:
    male_selected = st.checkbox("Male", value=True)
    female_selected = st.checkbox("Female", value=True)

# Placeholder for the chart
#title_placeholder = st.empty()
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
#title_placeholder.subheader("Population Density by Geographic Location")

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

st.write(pie_df)
