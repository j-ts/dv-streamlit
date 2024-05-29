import streamlit as st
import pandas as pd
import numpy as np

import geopandas as gpd
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import streamlit.components.v1 as components
from streamlit_extras.stylable_container import stylable_container

COUNTRY_COLORS = {
    "ENGLAND": ["#ffffff", "#fff7f0", "#ffefdf", "#ffe7cf", "#ffdfbe", "#ffd6a5", "#ffcd83", "#ffc461", "#ffbc3f", "#ffb31e"],
    "NORTHERN IRELAND": ["#ffffff", "#f2ffe7", "#e4ffcf", "#d7ffb6", "#caff9e", "#bdf685", "#afe06d", "#a2cb55", "#95b63c", "#87a124"],
    "SCOTLAND": ["#ffffff", "#ffe6e6", "#ffcccc", "#ffb3b3", "#ff9999", "#ff8080", "#ff6666", "#ff4d4d", "#ff3333", "#ff1a1a"],
    "WALES": ["#ffffff", "#ffffeb", "#ffffd6", "#ffffc2", "#fffdad", "#fffa99", "#fff785", "#fff471", "#fff05d", "#ffec49"]
}


APP_TITLE = 'UK Young Adults Demographics: A Decade in Data '
APP_SUB_TITLE = ''
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
st.write("#")



# First SCREEN

@st.cache_data
def scroll_to_basic_charts():
    js = '''
    <script>
        var element = window.parent.document.querySelector("#basic-charts");
        if (element) {
            element.scrollIntoView({behavior: 'smooth'});
        }
    </script>
    '''
    #st.write(js, unsafe_allow_html=True)
    components.html(js)


st.markdown(f"<h1 style='text-align: center; color: black;'>{APP_TITLE}</h2>",
    unsafe_allow_html=True)

#st.title(APP_TITLE)
st.caption(APP_SUB_TITLE)

__, __, col5, __, __ = st.columns(5)
with col5:
    st.markdown('This code will be printed.')

    #m = st.markdown("""<style>div.stButton > button:first-child {background-color:#8C83D1; font-style: italic;}</style>""", unsafe_allow_html=True) #background-color: #0099ff;
    # Button to scroll to the "Basic charts" section
    st.button("Explore the dashboard", on_click=scroll_to_basic_charts, type="primary")

st.divider()
st.image('jonny-unsplash.png', caption='Photo by Jonny Gios on Unsplash.',
            use_column_width="always")

# Second SCREEN
st.markdown("<h2> </h2><h2> </h2><h2 style='text-align: center;'>Basic charts</h2>", unsafe_allow_html=True)



# Function to pivot data by year, with suffix
@st.cache_data
def process_population_data(df):
    def pivot_data(df, year, suffix):
        df_year = df[df['year'] == year]
        total_population = df_year.pivot_table(index='area_name', values='population', aggfunc='sum')
        
        population = df_year.pivot_table(index='area_name', values='population', aggfunc='sum').rename(columns={'population': f'population_{suffix}'})
        
        sex = df_year.pivot_table(index='area_name', values='population', columns='sex', aggfunc='sum').div(total_population['population'], axis=0) * 100
        sex.columns = [f"{col}_{suffix}" for col in sex.columns]
        age_sex = df_year.pivot_table(index='area_name', values='population', columns='age_sex', aggfunc='sum').div(total_population['population'], axis=0) * 100
        age_sex.columns = [f"{col}_{suffix}" for col in age_sex.columns]
        age_year = df_year.pivot_table(index='area_name', values='population', columns='age_year', aggfunc='sum').div(total_population['population'], axis=0) * 100
        #age_year.columns = [f"{col}_{suffix}" for col in age_year.columns]
        return population, sex, age_sex, age_year

    df_copy = df.copy()
    df_copy.rename(columns={'name': 'area_name'}, inplace=True)
    df_copy['sex'] = df_copy['sex'].map({'M': 'Male', 'F': 'Female'})
    df_copy['age_sex'] = df_copy['age'].astype(str) + '_' + df_copy['sex']
    df_copy['age_year'] = df_copy['age'].astype(str) + '_' + df_copy['year'].astype(str)

    pop_2011, sex_2011, age_sex_2011, age_year_2011 = pivot_data(df_copy, 2011, '2011')
    pop_2022, sex_2022, age_sex_2022, age_year_2022 = pivot_data(df_copy, 2022, '2022')

    result_df = pd.concat([pop_2011, pop_2022, sex_2011, sex_2022, age_sex_2011, age_sex_2022, age_year_2011, age_year_2022], axis=1).reset_index()

    return result_df

# Load and process the data
@st.cache_data
def load_data():

    gh_folder = '/Users/jts/Documents/college/git-hub/dv-streamlit/'
    pie_df = pd.read_csv(gh_folder + 'pie_data.csv')
    # DELETE
    pie_df.rename(columns={'Year': 'year', 'Population': 'population'}, inplace=True)

    # prepare data for map
    new_df_copy = pie_df.copy()
    new_df_copy.rename(columns={'year': 'year', 'population': 'population'}, inplace=True)
    result_df = process_population_data(new_df_copy)


    #st.write(result_df.columns)
    # load gjson
    with open(gh_folder + "data-simplified.json", "r") as f:
        gdf = gpd.read_file(f)

    concatenated_df = pd.concat([gdf, result_df], axis=1)
    concatenated_df = concatenated_df.loc[:,~concatenated_df.columns.duplicated()].copy()

    # Convert the concatenated DataFrame to a GeoDataFrame
    gdf_choro = gpd.GeoDataFrame(concatenated_df, geometry=gdf.geometry)

    return pie_df, gdf_choro

pie_df, gdf_choro = load_data()

@st.cache_data
def plot_animated_population_pie_chart(df):
    # Aggregate the population by country and year
    df_grouped = df.groupby(['year', 'name'])['population'].sum().reset_index()

    # Get unique years
    years = df_grouped['year'].unique()

    # Initialize figure
    fig = go.Figure()

    # Add traces, one for each frame
    for year in years:
        df_year = df_grouped[df_grouped['year'] == year]
        fig.add_trace(go.Pie(
            labels=df_year['name'],
            values=df_year['population'],
            name='',
            marker=dict(colors=[COUNTRY_COLORS[country][5] for country in df_year['name']]),
            visible=(year == years[0])  # Only the first year is visible initially
        ))

    # Create frames
    frames = []
    for year in years:
        frame = go.Frame(
            data=[go.Pie(
                labels=df_grouped[df_grouped['year'] == year]['name'],
                values=df_grouped[df_grouped['year'] == year]['population'],
                marker=dict(colors=[COUNTRY_COLORS[country][5] for country in df_grouped[df_grouped['year'] == year]['name']])
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
                #'prefix': 'Year: ',
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
    st.subheader('Population Dynamics')
    fig.update_layout(showlegend=True,
        legend=dict(title='', traceorder='normal', orientation='h'))

    #fig.show()
    return fig

@st.cache_data
def plot_population_by_sex(df):
    # Group by name and sex, then sum the population
    grouped = df.groupby(['name', 'sex'])['population'].sum().reset_index()
    # Calculate the total population for each county
    total_population = grouped.groupby('name')['population'].transform('sum')
    # Normalize the population values so that M + F = 100% for each county
    grouped['population'] = (grouped['population'] / total_population) * 100

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
            x=[row['population']],
            y=[row['name']],
            name=f'{sex} – {country_name}',
            orientation='h',
            marker=dict(color=color),
            hoverinfo='text',
            width=0.7,
            hovertext=f'{country_name}<br>{sex}: {row["population"]:.1f}%',
            showlegend=False,
            #text=f'{sex}: {row["population"]:.1f}% of {country_name}'
        ))

    
    # Add female bars
    for i, row in females.iterrows():
        country_name = row['name']
        sex = 'Female'
        color = COUNTRY_COLORS[country_name][8]
        text_color = 'gray' if country_name == 'WALES' else 'gray'
        fig.add_trace(go.Bar(
            x=[row['population']],
            y=[row['name']],
            width=0.7,
            name=f'{sex} – {country_name}',
            orientation='h',
            marker=dict(color=color),
            hoverinfo='text',
            hovertext=f'{country_name}<br>{sex}: {row["population"]:.1f}%',
            hoverlabel={"bgcolor":'white'},
            showlegend=False,
        ))
    # Set the title and labels
    fig.update_layout(
        #title='Normalized Population by Sex in 2011',
        xaxis_title='Population (%)',
        yaxis_title='',
        #barmode='stack',
        showlegend=False,
        #legend=dict(title='', traceorder='normal', orientation='v'),
        )
    #fig.show()
    return fig

@st.cache_data
def plot_animated_population_by_sex(df):
    # Aggregate the population by country, sex, and year
    df_grouped = df.groupby(['year', 'name', 'sex'])['population'].sum().reset_index()
    
    total_population = df_grouped.groupby(['year', 'name'])['population'].transform('sum')
    # Normalize the population values so that M + F = 100% for each county
    
    df_grouped['population'] = (df_grouped['population'] / total_population) * 100
    
    # Get unique years
    years = df_grouped['year'].unique()

    # Initialize figure
    fig = go.Figure()

    # Add traces, one for each frame (initial year)
    initial_year = years[0]
    df_year = df_grouped[df_grouped['year'] == initial_year]
    
    # Separate the data for males and females for the initial year
    males = df_year[df_year['sex'] == 'M']
    females = df_year[df_year['sex'] == 'F']
    
    for i, row in males.iterrows():
        country_name = row['name']
        sex = 'Male'
        color = COUNTRY_COLORS[country_name][4]
        fig.add_trace(go.Bar(
            x=[row['population']],
            y=[row['name']],
            name=f'{sex} – {country_name}',
            orientation='h',
            marker=dict(color=color),
            hoverinfo='text',
            width=0.7,
            hovertext=f'{country_name}<br>{sex}: {row["population"]:.1f}%',
            visible=(row['year'] == initial_year)  # Only the first year is visible initially
        ))

    for i, row in females.iterrows():
        country_name = row['name']
        sex = 'Female'
        color = COUNTRY_COLORS[country_name][8]
        fig.add_trace(go.Bar(
            x=[row['population']],
            y=[row['name']],
            width=0.7,
            name=f'{sex} – {country_name}',
            orientation='h',
            marker=dict(color=color),
            hoverinfo='text',
            hovertext=f'{country_name}<br>{sex}: {row["population"]:.1f}%',
            visible=(row['year'] == initial_year)  # Only the first year is visible initially
        ))

    # Create frames
    frames = []
    for year in years:
        df_year = df_grouped[df_grouped['year'] == year]
        males = df_year[df_year['sex'] == 'M']
        females = df_year[df_year['sex'] == 'F']
        
        frame_data = []
        for i, row in males.iterrows():
            country_name = row['name']
            sex = 'Male'
            color = COUNTRY_COLORS[country_name][4]
            frame_data.append(go.Bar(
                x=[row['population']],
                y=[row['name']],
                name=f'{sex} – {country_name}',
                orientation='h',
                marker=dict(color=color),
                hoverinfo='text',
                width=0.7,
                hovertext=f'{country_name}<br>{sex}: {row["population"]:.1f}%',
            ))
        
        for i, row in females.iterrows():
            country_name = row['name']
            sex = 'Female'
            color = COUNTRY_COLORS[country_name][8]
            frame_data.append(go.Bar(
                x=[row['population']],
                y=[row['name']],
                width=0.7,
                name=f'{sex} – {country_name}',
                orientation='h',
                marker=dict(color=color),
                hoverinfo='text',
                hovertext=f'{country_name}<br>{sex}: {row["population"]:.1f}%',
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
                #'prefix': 'Year: ',
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
    fig.update_layout(
        xaxis_title='Population (%)',
        yaxis_title='',
        legend=dict(title='', traceorder='normal', orientation='v'),
        showlegend=False
    )

    return fig

@st.cache_data
def calculate_population_change(df):
    df_2011, df_2022 = df[df['year'] == 2011], df[df['year'] == 2022]
    pop_2011, pop_2022 = df_2011.groupby('name')['population'].sum().reset_index()['population'].sum(), df_2022.groupby('name')['population'].sum().reset_index()['population'].sum()
    delta = (pop_2022 - pop_2011) / pop_2011 * 100
    return pop_2022, delta



col = st.columns((0.8,1,1.2), gap='small')
with col[0]:
    st.subheader('Total Age Distribution')
    colors = {2011: COUNTRY_COLORS['NORTHERN IRELAND'][-1], 2022: COUNTRY_COLORS['SCOTLAND'][-1]}

    # Prepare data for each year
    data_2011 = pie_df[pie_df['year'] == 2011].groupby('age')['population'].sum().reset_index()
    data_2022 = pie_df[pie_df['year'] == 2022].groupby('age')['population'].sum().reset_index()
    min_population = min(data_2011['population'].min(), data_2022['population'].min())
    max_population = max(data_2011['population'].max(), data_2022['population'].max())

    # Create initial figure with data from 2011
    fig = go.Figure()

    hov_temp = 'Age: %{x}<br>Population: %{y}<extra></extra>'
    # Add 2011 data
    fig.add_trace(go.Scatter(x=data_2011['age'],y=data_2011['population'], 
                    mode='lines+markers', name='2011', line=dict(color=colors[2011]),
                    hovertemplate='2011<br>'+hov_temp))

    # Add 2022 data
    fig.add_trace(go.Scatter(x=data_2022['age'], y=data_2022['population'],
        mode='lines+markers', name='2022', visible=True, line=dict(color=colors[2022]),
        hovertemplate='2022<br>'+hov_temp))

    # Define buttons for switching between years
    

    # Update layout with buttons
    fig.update_layout(
        #updatemenus=[dict(type="buttons", buttons=buttons_years, x=.99, y=1.15)],
        xaxis_title="Age",
        yaxis=dict(
            title='Total Population',
            range=[min_population-20000, max_population+30000]
    ),
    )

    st.plotly_chart(fig, use_container_width=True, height=400)
with col[1]:
    st.plotly_chart(plot_animated_population_pie_chart(pie_df), use_container_width=True)

with col[2]:
    st.subheader('Sex Distribution')
    st.plotly_chart(plot_animated_population_by_sex(pie_df), use_container_width=True)
    


@st.cache_data
def calculate_sex_ratio(df):
    # Group by year and sex, then sum the population
    grouped = df.groupby(['year', 'sex'])['population'].sum().reset_index()

    # Separate the data for males and females
    males = grouped[grouped['sex'] == 'M']
    females = grouped[grouped['sex'] == 'F']

    # Merge male and female populations based on year
    merged = pd.merge(males, females, on='year', suffixes=('_male', '_female'))

    # Calculate sex ratio
    merged['Sex Ratio'] = (merged['population_male'] / merged['population_female']).round(2)
    ratio_2011 = merged.loc[merged['year'] == 2011, 'Sex Ratio'].values[0]
    ratio_2022 = merged.loc[merged['year'] == 2022, 'Sex Ratio'].values[0]
    delta = ratio_2022 - ratio_2011
    return ratio_2022, delta

__, Total_delta = calculate_population_change(pie_df)

@st.cache_data
def metrics():
        col_1, col_2 = st.columns(2)
        pop, d_pop = calculate_population_change(pie_df)
        d_pop = round(d_pop, 2)

        col_1.metric(label="Population", value="{:,}".format(pop), delta=f'{d_pop}%')

        #col2 = st.columns(1)[0]
        sex_ratio, sex_del = calculate_sex_ratio(pie_df)
        sex_del = round(sex_del, 2)
        col_2.metric(label="Male/Female Ratio", value=sex_ratio, delta=sex_del)
        #style_metric_cards(border_left_color="#8C83D1") #0074D9

# with col[2]:
    

st.markdown("<h2> </h2>", unsafe_allow_html=True)

#st.divider()
# Function to get population for a specific year
@st.cache_data
def get_population(df, year):
    pop_data = df[df['year'] == year]
    return pop_data['population'].sum() if not pop_data.empty else 0

@st.cache_data
def gender_gap(df):
    gender_gap = df.groupby(['name', 'age', 'sex'])['population'].sum().unstack()
    gender_gap['Gender Gap'] = abs((gender_gap['M'] - gender_gap['F'])/gender_gap['M']*100)

    # Find the maximum gender gap for each region
    max_gender_gap = gender_gap['Gender Gap'].groupby('name').idxmax().apply(lambda x: (x[1], gender_gap.loc[x, 'Gender Gap']))

    # Convert to a DataFrame for better readability
    max_gender_gap_df = pd.DataFrame(list(max_gender_gap.items()), columns=['Country', 'Age and Gap'])
    max_gender_gap_df[['Age', 'Gender Gap, %']] = max_gender_gap_df['Age and Gap'].apply(pd.Series)
    max_gender_gap_df.drop(columns='Age and Gap', inplace=True)
    max_gender_gap_df['Age'] = max_gender_gap_df['Age'].astype(int)
    max_gender_gap_df.sort_values('Gender Gap, %', ascending=False, inplace=True)
    return max_gender_gap_df


col, col1, col2, col3 = st.columns([1,1,1.2,0.8])
# Dropdown to select country
countries = gdf_choro['area_name'].unique()

with col1:
    st.subheader('Dynamics in numbers')
    metric_place = st.empty()
    selected_country = st.selectbox('', countries)
    # Filter data for the selected country
    df_filtered = pie_df[pie_df['name'] == selected_country.upper()]
    # Metrics for selected country
    pop_2011 = get_population(df_filtered, 2011)
    #st.metric(label=f'{selected_country} in 2011', value=f"{pop_2011:,}", delta=0, delta_color='off')
    pop_2022 = get_population(df_filtered, 2022)
    metric_place.metric(label=f'{selected_country} in 2022', value=f"{pop_2022:,}", delta=f"{int(pop_2022-pop_2011):,}")


with col:
    st.subheader('Biggest Gender Gaps by Age')
    #st.write('max_gender_gap_df')
    gender_gap_df = gender_gap(pie_df)
    st.dataframe(gender_gap_df,
                 #column_order=("states", "population"),
                 hide_index=True,
                 width=None,
                 column_config={
                 "Age": st.column_config.NumberColumn(
                    help="""Age group with biggest **Gender Gap**.""",),
                    "Gender Gap, %": st.column_config.ProgressColumn(
                        "Gender Gap, %",
                        help="""Normalized Difference between **Males** and **Females** in indicated *Age*.""",
                        format="%.2f",
                        min_value=0,
                        max_value=max(gender_gap_df['Gender Gap, %']),
                     )}
                 )


######################
with col2:
    st.markdown("<h3 style='text-align: center; margin-left:-70px;'>Metrics</h3>", unsafe_allow_html=True)
    #st.subheader('Metrics')
    metrics()
with col3:
    with st.expander('About', expanded=False):
        st.write('''
            - **Data**: The National Archives under [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
            - **Population**: Total population of the targeted age group in 2022.
            - **Male/Female Ratio**: Ratio of males to females in the targeted age group.
            ''')
    #pass
    

st.markdown("<h2> </h2><h2> </h2><h2> </h2><h2> </h2><h2> </h2>", unsafe_allow_html=True)

#################################
# Third SCREEN
st.markdown("<h2 style='text-align: center; color: black;'>Population Density by Geographic Location</h2>",
    unsafe_allow_html=True)

# Side filters
col1, col2 = st.columns([1,5])
with col1:
    st.subheader("Filters")
    year = st.selectbox("Select year", [2011, 2022])
    gender = st.selectbox("Select Gender", ["Both", "Male", "Female"])
    # age
    if 'age' not in st.session_state:
        st.session_state['age'] = None
    if 'slider_interacted' not in st.session_state:
        st.session_state['slider_interacted'] = False

    # Slider for selecting age
    slider_age = st.slider("Select Age", min_value=18, max_value=35, value=18)

    # Check if slider has been interacted with
    if not st.session_state['slider_interacted'] and slider_age != 18:
        st.session_state['slider_interacted'] = True
        st.session_state['age'] = slider_age
    elif st.session_state['slider_interacted']:
        st.session_state['age'] = slider_age

    age = st.session_state['age']

    cc1, cc2 = st.columns(2)
    with cc1:
        age_display = st.empty()
        age_display.write(f'Age selected: {age}')
    with cc2:
        ageb_placeholder = st.empty()
        #m = st.markdown("""<style>div.stButton > button:first-child {background-color:#FAF9F6; font-style: italic;}</style>""", unsafe_allow_html=True) #background-color: #0099ff;
    if ageb_placeholder.button("Reset age", type="secondary"):
        age = None
        st.session_state['slider_interacted'] = False
        st.session_state['age'] = None
        age_display.write(f'Age selected: {age}')

    #st.markdown('To interact with map, please, use fiters above.')


# Create the initial figure
fig = go.Figure()

# Add trace based on the selection
@st.cache_data
def get_column_name(year, gender, age=None):
    if gender == "Both":
        if age:
            return f'{age}_{year}'
        else:
            return f'population_{year}'
    else:
        if age:
            return f'{age}_{gender}_{year}'
        return f'{gender}_{year}'

column_name = get_column_name(year, gender, age)
z_data = gdf_choro[column_name]


if gender == "Both" and age:
    hovertemplate = f'<b>%{{customdata[0]}}</b><br>%{{z:.2f}}% of people aged {age} in {year}<extra></extra>'
elif gender == "Both":
    hovertemplate = '<b>%{customdata[0]}</b><br>%{z} people<extra></extra>'
elif age:
    hovertemplate = f'<b>%{{customdata[0]}}</b><br>%{{z:.2f}} % of {gender.lower()}s aged {age} in {year}<extra></extra>'
else:
    hovertemplate = f'<b>%{{customdata[0]}}</b><br>%{{z:.2f}} % of {gender.lower()}s in {year}<extra></extra>'

fig.add_trace(go.Choroplethmapbox(
    geojson=gdf_choro.__geo_interface__,
    locations=gdf_choro.index,
    z=z_data,
    colorscale= 'Cividis',# "Viridis",
    marker_opacity=0.5,
    marker_line_width=0,
    hovertemplate=hovertemplate,
    customdata=gdf_choro[['area_name']].values
))

# Update the layout
fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center={"lat": 54.5, "lon": -3.4},
        zoom=4
    ),
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

# show the map
with col2:
    st.plotly_chart(fig, use_container_width=True, height=1800)

st.divider()
st.subheader('Insights')
st.write(f"""
    * Total Population increased on {Total_delta:.2f}%.
    * Population in each country decreased by 2022, except England.
    * In 2022, the highest populations are in the 28-35 age range.
    * In 2011, the population had spikes every 3 years between 20 and 29 age range.
    """)


