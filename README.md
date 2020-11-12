

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

## Preliminary Analysis Write-up

Performed the below tasks as part of the preprocessing process:

1. Removing Unnecessary Columns: Dropped the name, streetaddress, day, latitude and longitude attributes as they are not necessary for the analysis and hence dropping these columns will not affect our analysis.

2. Filling the missing and invalid data: Identified the columns which has invalid or null data. Counted the unique/distinct occurrences of values for string data type values to identify the invalid values, and for numeric attributes filled the null data with mean/median by plotting the graph with the values and examining the distribution graph and filling invalid data with appropriate values. Dropped the rows when few attributes like race ethnicity, armed and cause has missing or invalid data as it would be difficult to perform classification in the later stages.

 #### Data Analysis and Visualizations:

After pre-processing the data it's important to analyze the data to know the relation between attributes. So, we have analyzed the important attributes such as age bucket, gender, city, state, law enforcement agency, killings by month, armed, race and ethnicity and for simple understanding and visualization we represented data with appropriate graphical visualizations such as bar graph and pie charts.

#### The breakup of data based on gender:
![The breakup of data based on gender](/images/gender.png)
#### Top five cities in which police killings took place:
![Top five Cities](/images/city.png)
#### Top ten age buckets subjected to police killings :
![Top ten Age Buckets](/images/age.png)
#### Top five law enforcement agencies involved in police killings:
![Top five law enforcement agencies](/images/lawenforcement.png)
#### Trend of police killings in the year of 2015 on a monthly basis:
![Trend of police killings](/images/month.png)
#### The breakup of data based on race:
![The breakup of data based on race](/images/race.png)
#### Top five states in which police killings took places:
![ Top five states](/images/state.png)
#### The breakup of data based on armed status:
![The breakup of data based on armed status](/images/armed.png)
#### The following table represents the various correlation coefficients :
![Correlation](/images/correlation.png)

The number of incidents taken place in a city are influenced by the following factors such as average poverty rate, average household income, average unemployment rate, average personal income, average literacy rate in the order of decreasing influence.

#### Potential methods we are considering applying to mine patterns in the data:

We are planning to use decision tree classifier to generate a model which can be used to classify the police killings data to classify whether the killing is unwanted or not.

We are planning to use clustering algorithms such as K – means and Hierarchical clustering algorithms by choosing the best cluster number . At the end of the process we will be having a cluster label for each of the record and the properties of the clusters formed can be analyzed to mine interesting patterns in the data.

