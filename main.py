import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
# These might be helpful:
from iso3166 import countries

pd.options.display.float_format = '{:,.2f}'.format
df_data = pd.read_csv('mission_launches.csv')

# Shape of df_data
print("Shape of df_data:", df_data.shape)

# Number of rows and columns
num_rows = df_data.shape[0]
num_cols = df_data.shape[1]
print("Number of rows:", num_rows)
print("Number of columns:", num_cols)

# Column names
column_names = df_data.columns
print("Column names:", column_names)

# Check for NaN values
print("Number of NaN values:\n", df_data.isna().sum())

# Check for duplicates
num_duplicates = df_data.duplicated().sum()
print("Number of duplicates:", num_duplicates)

# Identify columns containing junk data
junk_columns = ['Sno', 'SNO']  # Replace with actual column names to be removed

# Drop the junk columns
df_data = df_data.drop(junk_columns, axis=1)

# Verify the updated DataFrame
print("Updated DataFrame after removing junk columns:")
print(df_data.head())

# Compute descriptive statistics
statistics = df_data.describe()

# Print the statistics
print("Descriptive Statistics:")
print(statistics)

# Count the number of launches per company
launches_by_company = df_data['Organisation'].value_counts().reset_index()

# Rename the columns
launches_by_company.columns = ['Organisation', 'Number of Launches']

# Sort the data by number of launches in descending order
launches_by_company = launches_by_company.sort_values('Number of Launches', ascending=False)

# Create the bar chart
plt.figure(figsize=(12, 8))  # Increase figure size for better visibility
sns.barplot(x='Number of Launches', y='Organisation', data=launches_by_company, palette='viridis')
plt.xlabel('Number of Launches')
plt.ylabel('Organisation')
plt.title('Number of Space Mission Launches by Organisation')


plt.tight_layout()  # Add spacing between plot elements

'''plt.show()'''

# Count the number of active and retired rockets
rocket_status_counts = df_data['Rocket_Status'].value_counts()

# Display the counts
print("Number of Active Rockets:", rocket_status_counts['StatusActive'])
print("Number of Retired Rockets:", rocket_status_counts['StatusRetired'])

# Count the number of successful and failed missions
mission_status_counts = df_data['Mission_Status'].value_counts()

# Display the counts
print("Number of Successful Missions:", mission_status_counts['Success'])
print("Number of Failed Missions:", mission_status_counts['Failure'])

# Filter out missing values in the Price column
launch_prices = df_data['Price'].dropna()

# Plot the histogram with wider spacing on the x-axis
plt.figure(figsize=(10, 8))
sns.histplot(launch_prices, bins=20, kde=True)

# Set plot title and axis labels
plt.title('Distribution of Launch Prices')
plt.xlabel('Price (USD millions)')
plt.ylabel('Count')

# Adjust x-axis spacing
plt.xticks(rotation=45, ha='right')

# Display the plot
'''plt.show()'''


# Extract country from location
def extract_country(location):
    words = location.split()
    country_name = words[-1]
    return country_name


df_data['Country'] = df_data['Location'].apply(extract_country)

# Wrangle country names
country_mapping = {
    'Russia': 'RUS',
    'New Mexico': 'USA',
    'Yellow Sea': 'China',
    'Shahrud Missile Test Site': 'Iran',
    'Pacific Missile Range Facility': 'USA',
    'Barents Sea': 'RUS',
    'Gran Canaria': 'USA'
}

df_data['Country'] = df_data['Country'].replace(country_mapping)

# Group by country and count the number of launches
launches_by_country = df_data['Country'].value_counts().reset_index()
launches_by_country.columns = ['Country', 'Launches']

# Create choropleth map
fig = px.choropleth(
    launches_by_country,
    locations='Country',
    locationmode='ISO-3',
    color='Launches',
    color_continuous_scale='matter',
    title='Number of Launches by Country'
)

# Update layout and show the map
fig.update_layout(
    margin=dict(l=0, r=0, t=60, b=0),
    title_font=dict(size=20),
    coloraxis_colorbar=dict(
        title='Launches',
        thicknessmode='pixels', thickness=100,
        lenmode='pixels', len=300,
        yanchor='middle', y=0.5
    )
)

'''fig.show()'''

# Group the data by country and count the failures
failures_by_country = df_data[df_data['Mission_Status'] == 'Failure'].groupby('Country').size().reset_index(name='Failures')

# Create a choropleth map figure
fig = go.Figure(data=go.Choropleth(
    locations=failures_by_country['Country'],
    z=failures_by_country['Failures'],
    text=failures_by_country['Country'],
    colorscale='Reds',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title='Failures'
))

# Set the title and layout of the figure
fig.update_layout(
    title_text='Number of Failures by Country',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

# Show the figure
'''fig.show()'''


# Group the data by countries, organizations, and mission status
sunburst_data = df_data.groupby(['Location', 'Organisation', 'Mission_Status']).size().reset_index(name='Count')

fig = go.Figure(go.Sunburst(
    labels=sunburst_data['Location'] + ' - ' + sunburst_data['Organisation'] + ' - ' + sunburst_data['Mission_Status'],
    parents=[''] * len(sunburst_data),
    values=sunburst_data['Count'],
))

fig.update_layout(
    margin=dict(t=0, l=0, r=0, b=0),
    width=800,
    height=800,
    title='Sunburst Chart of Countries, Organizations, and Mission Status',
)

'''fig.show()'''

# Convert the 'Price' column to numeric
df_data['Price'] = pd.to_numeric(df_data['Price'], errors='coerce')

# Group the data by organization and calculate the sum of money spent
money_spent_by_organisation = df_data.groupby('Organisation')['Price'].sum().reset_index()

# Sort the data in descending order based on the total amount spent
money_spent_by_organisation = money_spent_by_organisation.sort_values('Price', ascending=False)

# Create a bar plot to visualize the total amount spent by each organization
fig = px.bar(money_spent_by_organisation, x='Organisation', y='Price',
             labels={'Organisation': 'Organisation', 'Price': 'Total Amount Spent'},
             title='Total Amount of Money Spent by Organisation on Space Missions')

'''fig.show()'''

# Convert the 'Price' column to numeric
df_data['Price'] = pd.to_numeric(df_data['Price'], errors='coerce')

# Group the data by organisation and calculate the sum of money spent and the number of launches
money_spent_by_organisation = df_data.groupby('Organisation').agg({'Price': 'sum', 'Mission_Status': 'size'}).reset_index()
money_spent_by_organisation.columns = ['Organisation', 'Total Amount Spent', 'Number of Launches']

# Calculate the amount spent per launch
money_spent_by_organisation['Amount Spent per Launch'] = money_spent_by_organisation['Total Amount Spent'] / money_spent_by_organisation['Number of Launches']

# Sort the data in descending order based on the amount spent per launch
money_spent_by_organisation = money_spent_by_organisation.sort_values('Amount Spent per Launch', ascending=False)

# Create a bar plot to visualize the amount spent per launch by each organisation
fig = px.bar(money_spent_by_organisation, x='Organisation', y='Amount Spent per Launch',
             labels={'Organisation': 'Organisation', 'Amount Spent per Launch': 'Amount Spent per Launch'},
             title='Amount of Money Spent per Launch by Organisation')

'''fig.show()'''

# Extract the date without the time component
df_data['Date'] = df_data['Date'].str.split(' ').str[:4].str.join(' ')

# Convert the 'Date' column to datetime format
df_data['Date'] = pd.to_datetime(df_data['Date'], format='%a %b %d, %Y', utc=True)

# Extract the year from the 'Date' column
df_data['Year'] = df_data['Date'].dt.year

# Group the data by year and count the number of launches
launches_per_year = df_data.groupby('Year').size().reset_index(name='Number of Launches')

# Create a line plot to visualize the number of launches per year
fig = px.line(launches_per_year, x='Year', y='Number of Launches',
              labels={'Year': 'Year', 'Number of Launches': 'Number of Launches'},
              title='Number of Launches per Year')

'''fig.show()'''


# Convert the 'Date' column to datetime format
df_data['Date'] = pd.to_datetime(df_data['Date'], format='%a %b %d, %Y %H:%M %Z')

# Extract the month and year from the 'Date' column
df_data['Month'] = df_data['Date'].dt.strftime('%Y-%m')

# Group the data by month and count the number of launches
launches_per_month = df_data.groupby('Month').size().reset_index(name='Number of Launches')

# Calculate the rolling average with a window size of 3 months
launches_per_month['Rolling Average'] = launches_per_month['Number of Launches'].rolling(window=3).mean()

# Find the month with the highest number of launches
max_month = launches_per_month.loc[launches_per_month['Number of Launches'].idxmax(), 'Month']
max_launches = launches_per_month['Number of Launches'].max()

# Create the month-on-month chart with rolling average
fig = px.line(launches_per_month, x='Month', y='Number of Launches', 
              title='Number of Launches Month-on-Month',
              labels={'Month': 'Month', 'Number of Launches': 'Number of Launches'})

fig.add_scatter(x=launches_per_month['Month'], y=launches_per_month['Rolling Average'],
                mode='lines', name='Rolling Average')

# Highlight the month with the highest number of launches
fig.add_annotation(x=max_month, y=max_launches,
                   text=f'Max: {max_month} ({max_launches} launches)',
                   showarrow=True, arrowhead=1)

'''fig.show()'''


# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')
df_data['Date'] = pd.to_datetime(df_data['Date'], format='%a %b %d, %Y %H:%M %Z')
df_data['Month'] = df_data['Date'].dt.month
df_data['Year'] = df_data['Date'].dt.year

# Calculate the number of launches per month
launches_per_month = df_data.groupby(['Month', 'Year']).size().reset_index(name='Launches')

# Calculate the average number of launches per month
average_launches_per_month = launches_per_month.groupby('Month').mean().reset_index()

# Sort the data by the average number of launches
sorted_data = average_launches_per_month.sort_values('Launches', ascending=False)

# Visualize the launches per month
fig = px.bar(sorted_data, x='Month', y='Launches', labels={'Month': 'Month', 'Launches': 'Avg. Number of Launches'})
fig.update_layout(title='Average Launches per Month')
'''fig.show()'''

# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')

# Calculate the number of launches per organization per year
launches_by_organization_year = df_data.groupby(['Year', 'Organisation']).size().reset_index(name='Number of Launches')
top_10_organizations = df_data['Organisation'].value_counts().nlargest(10).index
launches_by_organization_year_top_10 = launches_by_organization_year[launches_by_organization_year['Organisation'].isin(top_10_organizations)]

# Visualize the number of launches over time by the top 10 organizations
fig = px.line(launches_by_organization_year_top_10, x='Year', y='Number of Launches', color='Organisation',
              labels={'Year': 'Year', 'Number of Launches': 'Number of Launches', 'Organisation': 'Organization'},
              title='Number of Launches Over Time by Top 10 Organizations')
'''fig.show()'''

# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')

# Filter the data for launches during the Cold War period (up until 1991)
df_cold_war = df_data[df_data['Year'] <= 1991]

# Filter the data for launches by the top two organizations: NASA and RVSN USSR
top_2_organizations = ['NASA', 'RVSN USSR']
df_cold_war_top_2 = df_cold_war[df_cold_war['Organisation'].isin(top_2_organizations)]

# Calculate the number of launches per year by the top two organizations
launches_by_year_top_2 = df_cold_war_top_2.groupby(['Year', 'Organisation']).size().reset_index(name='Number of Launches')

# Visualize the number of launches over time by the top two organizations
fig = px.line(launches_by_year_top_2, x='Year', y='Number of Launches', color='Organisation',
              labels={'Year': 'Year', 'Number of Launches': 'Number of Launches', 'Organisation': 'Organization'},
              title='Number of Launches Over Time: USA vs USSR')
'''fig.show()'''


# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')

# Filter the data for launches by the USSR and the USA, including launches from Kazakhstan
countries = ['USA', 'USSR']
df_countries = df_data[df_data['Country'].isin(countries) | df_data['Location'].str.contains('Kazakhstan')]

# Replace the country names to combine Kazakhstan under the USSR category
df_countries['Country'] = df_countries['Country'].replace({'Kazakhstan': 'USSR'})

# Calculate the total number of launches by country
launches_by_country = df_countries['Country'].value_counts().reset_index()
launches_by_country.columns = ['Country', 'Number of Launches']

# Visualize the total number of launches using a pie chart
fig = px.pie(launches_by_country, values='Number of Launches', names='Country',
             title='Total Number of Launches: USSR vs USA',
             labels={'Number of Launches': 'Number of Launches', 'Country': 'Country'})

'''fig.show()'''


# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')

# Filter the data for launches by the USA and the USSR
countries = ['USA', 'USSR']
df_countries = df_data[df_data['Country'].isin(countries)]

# Extract the year from the date column
df_countries['Year'] = pd.to_datetime(df_countries['Date']).dt.year

# Calculate the total number of launches year-on-year by country
launches_yearly = df_countries.groupby(['Country', 'Year']).size().reset_index(name='Number of Launches')

# Create a line chart to visualize the total number of launches year-on-year
fig = px.line(launches_yearly, x='Year', y='Number of Launches', color='Country',
              title='Total Number of Launches Year-On-Year: USA vs USSR',
              labels={'Year': 'Year', 'Number of Launches': 'Number of Launches', 'Country': 'Country'})

'''fig.show()'''


# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')

# Filter the data for mission failures
failure_conditions = ['Failure', 'Partial Failure', 'Prelaunch Failure']
df_failures = df_data[df_data['Mission_Status'].isin(failure_conditions)].copy()

# Extract the year from the date column
df_failures['Year'] = pd.to_datetime(df_failures['Date']).dt.year

# Calculate the total number of mission failures year-on-year
failures_yearly = df_failures.groupby('Year').size().reset_index(name='Number of Failures')

# Create a line chart to visualize the total number of mission failures year-on-year
fig = px.line(failures_yearly, x='Year', y='Number of Failures',
              title='Total Number of Mission Failures Year-On-Year',
              labels={'Year': 'Year', 'Number of Failures': 'Number of Failures'})

'''fig.show()'''


# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')

# Filter the data for mission failures
failure_conditions = ['Failure', 'Partial Failure', 'Prelaunch Failure']
df_failures = df_data[df_data['Mission_Status'].isin(failure_conditions)].copy()

# Extract the year from the date column
df_failures['Year'] = pd.to_datetime(df_failures['Date']).dt.year

# Calculate the total number of launches year-on-year
launches_yearly = df_data.groupby(df_data['Date'].dt.year).size().reset_index(name='Number of Launches')

# Calculate the total number of failures year-on-year
failures_yearly = df_failures.groupby(df_failures['Year']).size().reset_index(name='Number of Failures')

# Merge the number of launches and number of failures DataFrames
df_merged = pd.merge(launches_yearly, failures_yearly, how='left', left_on='Date', right_on='Year')

# Calculate the percentage of failures over time
df_merged['Percentage of Failures'] = (df_merged['Number of Failures'] / df_merged['Number of Launches']) * 100

# Create a line chart to visualize the percentage of failures over time
fig = px.line(df_merged, x='Date', y='Percentage of Failures',
              title='Percentage of Failures Over Time',
              labels={'Date': 'Year', 'Percentage of Failures': 'Percentage of Failures'})

'''fig.show()'''

# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')

# Filter the data for launches up to and including 2020
df_filtered = df_data[df_data['Date'].dt.year <= 2020]

# Calculate the total number of launches per year and country
launches_by_year_country = df_filtered.groupby(['Date', 'Country']).size().reset_index(name='Number of Launches')

# Find the country in the lead in terms of the total number of launches each year
df_lead_country = launches_by_year_country.groupby('Date')['Number of Launches'].idxmax()
lead_country_per_year = launches_by_year_country.loc[df_lead_country]

# Create a bar chart to visualize the lead country in terms of the total number of launches each year
fig = px.bar(lead_country_per_year, x='Date', y='Number of Launches', color='Country',
             title='Lead Country in Total Number of Launches Each Year (All Launches)',
             labels={'Date': 'Year', 'Number of Launches': 'Number of Launches', 'Country': 'Country'})

'''fig.show()'''

# Filter the data for successful launches
df_success = df_filtered[df_filtered['Mission_Status'] == 'Success']

# Calculate the total number of successful launches per year and country
success_by_year_country = df_success.groupby(['Date', 'Country']).size().reset_index(name='Number of Successful Launches')

# Find the country in the lead in terms of the total number of successful launches each year
df_lead_success_country = success_by_year_country.groupby('Date')['Number of Successful Launches'].idxmax()
lead_success_country_per_year = success_by_year_country.loc[df_lead_success_country]

# Create a bar chart to visualize the lead country in terms of the total number of successful launches each year
fig_success = px.bar(lead_success_country_per_year, x='Date', y='Number of Successful Launches', color='Country',
                     title='Lead Country in Total Number of Successful Launches Each Year',
                     labels={'Date': 'Year', 'Number of Successful Launches': 'Number of Successful Launches',
                             'Country': 'Country'})

'''fig_success.show()'''



# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')

# Filter the data for launches in the desired decades and years
df_filtered = df_data[df_data['Date'].dt.year.isin(range(1970, 1980 + 1)) | df_data['Date'].dt.year.isin([2018, 2019, 2020])]

# Calculate the number of launches per year and organization
launches_by_year_org = df_filtered.groupby(['Date', 'Organisation']).size().reset_index(name='Number of Launches')

# Find the organization with the most launches each year
df_lead_org = launches_by_year_org.groupby('Date')['Number of Launches'].idxmax()
lead_org_per_year = launches_by_year_org.loc[df_lead_org]

# Create a bar chart to visualize the organization with the most launches each year
fig = px.bar(lead_org_per_year, x='Date', y='Number of Launches', color='Organisation',
             title='Organization with the Most Launches Each Year',
             labels={'Date': 'Year', 'Number of Launches': 'Number of Launches', 'Organisation': 'Organization'})

'''fig.show()'''


# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')
df_data['Date'] = pd.to_datetime(df_data['Date'], format='%a %b %d, %Y %H:%M %Z')
df_data['Year'] = df_data['Date'].dt.year

# Calculate the average price of rocket launches per year
average_price_by_year = df_data.groupby('Year')['Price'].mean().reset_index()

# Visualize the average price of rocket launches over time
fig = px.line(average_price_by_year, x='Year', y='Price', labels={'Year': 'Year', 'Price': 'Average Launch Price'})
fig.update_layout(title='Average Launch Price Over Time')
fig.show()