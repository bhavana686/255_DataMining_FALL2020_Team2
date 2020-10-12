# Team Members
Akhil Reddy Mandadi  - (014608451) - akhilmandadi
Jagan Gadamsetty (014636531) - jagan695
Bhavana (014618513) - bhavana686

# Police Killings Analysis
### Data Source

> **Link**: https://github.com/fivethirtyeight/data/blob/master/police-killings/police_killings.csv

**Attributes**:
name: Name of deceased
age:Age of deceased
gender:Gender of deceased
raceethnicity: Race/ethnicity of deceased
month: Month of killing
day:Day of incident
year:Year of incident
streetaddress:Address/intersection where incident occurred
city:City where incident occurred
state:State where incident occurred
latitude:Latitude, geocoded from address
longitude:Longitude, geocoded from address
state_fp:State FIPS code
county_fp:County FIPS code
tract_ce:Tract ID code
geo_id:Combined tract ID code
county_id:Combined county ID code
namelsad:Tract description
lawenforcementagency:Agency involved in incident
cause:Cause of death
armed:How/whether deceased was armed
pop:Tract population
share_white:Share of pop that is non-Hispanic white
share_bloack:Share of pop that is black
share_hispanic:Share of pop that is Hispanic/Latino
p_income:Tract-level median personal income
h_income:Tract-level median household income
county_income:County-level median household income
comp_income:h_income/county_income
county_bucket:Household income, quintile within county
nat_bucket:Household income, quintile nationally
pov:Tract-level poverty rate
urate:Tract-level unemployment rate
college:Share of 25+ pop with BA or higher

## Description of the problem you’ll solve or the question you’ll investigate
1. Find out the age bucket where most of the deceased fall.
2. Find out the gender which most of the deceased belongs to.
3. Distribution of police killings yearly overtime and the find the trend.
4. Find out the race/ethnicity which most of the deceased belongs to.
5. Find out the city and state where most of the incidents take place.
6. Find out whether most of the deceased hold arms or not.
7. Find out the law enforcement agency that involved in most of the incidents.
8. Derive the correlation between median household income,median personal income and the number of incidents in a city.
9. Derive the correlation between unemployment rate and the number of incidents in a city.
10. Derive the correlation between literacy rate and the number of incidents in a city.
11. Derive the correlation between poverty rate and the number of incidents in a city. 

## Potential methods you will consider apply
Preprocess the data by removing outliers and filling out missing values in an appropriate manner and process the data to find the useful information and also visualize it for clear understanding. Correlate the columns that we are interested in and mine useful relation between them.

## How will you measure success?

We should be able to findout the relations between various dimensions of the data and mine useful insights from the raw data.

