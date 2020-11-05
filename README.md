
# Team Members

 - Akhil Reddy Mandadi  - (014608451) - akhilmandadi 
 - Jagan Gadamsetty (014636531) - jagan695 
 - Bhavana (014618513) - bhavana686

# Police Killings Analysis
### Data Source

> **Link**: https://github.com/fivethirtyeight/data/blob/master/police-killings/police_killings.csv

**Attributes**:

 - name: Name of deceased 
 - age:Age of deceased 
 - gender:Gender of deceased
 - raceethnicity: Race/ethnicity of deceased 
- month: Month of killing
- day:Day of incident 
- year:Year of incident
- streetaddress:Address/intersection where incident occurred 
- city:City
- where incident occurred 
- state:State where incident occurred
- latitude:Latitude, geocoded from address 
- longitude:Longitude, geocoded from address 
- state_fp:State FIPS code 
- county_fp:County FIPS
- code tract_ce:Tract ID code 
- geo_id:Combined tract ID code
- county_id:Combined county ID code 
- namelsad:Tract description
- lawenforcementagency:Agency involved in incident 
- cause:Cause of death
- armed:How/whether deceased was armed 
- pop:Tract population
- share_white:Share of pop that is non-Hispanic white
- share_bloack:Share of pop that is black 
- share_hispanic:Share of pop that is Hispanic/Latino 
- p_income:Tract-level median personal income
- h_income:Tract-level median household income
- county_income:County-level median household income
- comp_income:h_income/county_income 
- county_bucket:Household income quintile within county 
- nat_bucket:Household income, quintile
- nationally pov:Tract-level poverty rate 
- urate:Tract-level unemployment rate 
- college:Share of 25+ pop with BA or higher

## Description of the problem you’ll solve or the question you’ll investigate
United States have a higher rate of police killings with 3 people being killed per day on an average. We will try to understand the killings happened across the United States and better analyse and understand the victims of the killings, and factors that impact killings. We will try to focus on how the killings might have been impacted by certain factors like ethnicity, income levels, education etc and how these attributes might have impacted the possibility of being an victim of the killings. We will apply Classification and Clustering techniques to analyse the killings.
Below are some of the findings we are looking at:
1. Classify the data of police killings to find out if the killing is unwanted or not
2. Generate clusters from the data and try to analyze them and label then based on the properties of the clusters.
3. Find out the age bucket where most of the deceased fall.
4. Find out the gender which most of the deceased belongs to.
5. Distribution of police killings yearly overtime and the find the trend.
6. Find out the race/ethnicity which most of the deceased belongs to.
7. Find out the city and state where most of the incidents take place.
8. Find out whether most of the deceased hold arms or not.
9. Find out the law enforcement agency that involved in most of the incidents.
10. Derive the correlation between median household income,median personal income and the number of incidents in a city.
11. Derive the correlation between unemployment rate and the number of incidents in a city.
12. Derive the correlation between literacy rate and the number of incidents in a city.
13. Derive the correlation between poverty rate and the number of incidents in a city. 

## Potential methods you will consider apply
Preprocessing is one of the most essential step in the process of data mining and we are targetting to clean the data by techniques like removing outliers, eliminating noise and filling out missing values in an appropriate manner. Then process the data to find the useful information by visualizing it for clear understanding by using histograms, scatter plots etc. 
We will be using Classification techniques such as Decision tree classifier to analyse whether the killing is unwanted or not and
Clustering techniques such as Agglomeration clustering to cluster the data and analyze the clusters and label them accordingly based on the properties of the cluster.
And also correlate the columns that we are interested in and mine useful relationships between them.

## How will you measure success?

We will evaluate the accuracy of our decision tree classifier model by dividing our data into training and test sets. We will be able to find some key relationships and patterns on how few attributes like ethicity, income and education can correlate to a killing.
We should be able to findout the relations between various dimensions of the data and mine useful insights from the raw data.

