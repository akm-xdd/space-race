# Load the data into a DataFrame (assuming the data is already loaded into 'df_data')

# Filter the data for launches by the USSR and the USA, including launches from Kazakhstan
countries = ['USA', 'USSR', 'KAZAKHSTAN']
df_countries = df_data[df_data['Country'].isin(countries) | df_data['Location'].str.contains('Kazakhstan')]

# Calculate the total number of launches by country
launches_by_country = df_countries.groupby('Country').size().reset_index(name='Number of Launches')

# Visualize the total number of launches using a pie chart
fig = px.pie(launches_by_country, values='Number of Launches', names='Country',
             title='Total Number of Launches: USSR vs USA',
             labels={'Number of Launches': 'Number of Launches', 'Country': 'Country'})

fig.show()
