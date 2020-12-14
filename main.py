import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data(filename):
    """Loading a csv file into a Pandas dataframe.
        filename - string
        return Pandas dataframe
    """
    return pd.read_csv(filename, encoding='ISO-8859-1')


def remove_unnecessary_columns(data):
    data.drop(
        ['name', 'streetaddress', 'day', 'latitude', 'longitude', 'geo_id', 'county_bucket', 'nat_bucket', 'tract_ce',
         'county_id', 'county_fp', 'state_fp'],
        axis=1, inplace=True)


def fill_missing_values(data):
    # Identifying the rows of age with non numeric values
    age = pd.to_numeric(data.age, errors='coerce')
    idx = age.isna()
    # There are total of 4 rows with age as Unknown. Found it by printing rows with non numeric values
    # Printing Non numeric rows
    print(data[idx])
    # Replacing Unknown values with zero
    data["age"] = data["age"].replace("Unknown", '0')
    data[["age"]] = data[["age"]].apply(pd.to_numeric)

    # Plotted the number of distinct age values on x-axis and occurrences on y-axis and the graph was
    # skewed distribution. So median is preferred to replace the missing values
    plt.plot(data["age"].value_counts())
    plt.title('Distribution graph of attribute - age')
    plt.show()
    median_age = data['age'].median()
    data["age"] = data["age"].replace(0, round(median_age))
    print(data.dtypes)

    # Dropping the rows with values as unknown for the columns raceethnicity, armed and cause
    print(data["gender"].value_counts())
    print(data["raceethnicity"].value_counts())
    data = data[data.raceethnicity != 'Unknown']
    print(data["armed"].value_counts())
    print(data["cause"].value_counts())
    data = data[data.armed != 'Unknown']
    data = data[data.cause != 'Unknown']
    print(len(data))

    urate = pd.to_numeric(data.urate, errors='coerce')
    idx = urate.isna()
    # There are total of 2 rows with urate as NaN. Found it by printing rows with non numeric values
    # Printing Non numeric rows
    print(data[idx])
    data[["urate"]] = data[["urate"]].apply(pd.to_numeric)
    plt.plot(data["urate"].value_counts())
    plt.title('Distribution graph of attribute - unemployment rate')
    plt.show()
    # The graph is skewed distribution, replacing null values with median
    median_urate = data['urate'].median()
    data.fillna({'urate': median_urate}, inplace=True)

    college = pd.to_numeric(data.college, errors='coerce')
    idx = college.isna()
    # There are total of 2 rows with college as NaN. Found it by printing rows with non numeric values
    # Printing Non numeric rows
    print(data[idx])
    data[["college"]] = data[["college"]].apply(pd.to_numeric)
    plt.plot(data["college"].value_counts())
    plt.title('Distribution graph of attribute - literacy rate')
    plt.show()
    # The graph is skewed distribution, replacing null values with median
    median_college = data['college'].median()
    data.fillna({'college': median_college}, inplace=True)

    share_white = pd.to_numeric(data.share_white, errors='coerce')
    idx = share_white.isna()
    # Printing Non numeric rows
    print(data[idx])
    # There are few rows of data with share_white, share_black, share_hispanic  as -.
    # Found it by printing rows with non numeric values and replacing with zero
    data["share_white"] = data["share_white"].replace("-", '0')
    data["share_black"] = data["share_black"].replace("-", '0')
    data["share_hispanic"] = data["share_hispanic"].replace("-", '0')

    # Handling Missing values of p_income by replacing invalid values and replacing with median
    data["p_income"] = data["p_income"].replace("-", '0')
    data[["p_income"]] = data[["p_income"]].apply(pd.to_numeric)
    median_p_income = data['p_income'].median()
    data["p_income"] = data["p_income"].replace(0, round(median_p_income))

    # Filling Missing values of h_income with median by using fillna function
    median_h_income = data['h_income'].median()
    data.fillna({'h_income': median_h_income}, inplace=True)

    # Filling Missing values of comp_income with median by using fillna function
    median_comp_income = data['comp_income'].median()
    data.fillna({'comp_income': median_comp_income}, inplace=True)

    # Filling Missing values of pov with median by using replace function
    data["pov"] = data["pov"].replace("-", '0')
    data[["pov"]] = data[["pov"]].apply(pd.to_numeric)
    median_pov = data['pov'].median()
    data["pov"] = data["pov"].replace(0, round(median_pov))

    return data


def data_analysis(data):
    # Finding the breakup of what arms the deceased holds at the time of death like gun, knife etc.
    arms_breakup = data["armed"].value_counts()
    print(arms_breakup)
    armed_labels = 'Firearm', 'No', 'Knife', 'Other', 'Vehicle', 'Non-lethal firearm', 'Disputed'
    plt.pie(arms_breakup, labels=armed_labels, autopct='%1.1f%%')
    plt.title('Breakdown by armed:')
    plt.axis('equal')
    plt.show()

    # Finding correlation between number of incidents taken place in a city and the average personal income of the city.
    incidents_per_city = data["city"].value_counts()
    average_personal_income_of_cities = data[['city', 'p_income']]
    average_personal_income_of_cities = average_personal_income_of_cities.groupby(
        [average_personal_income_of_cities["city"]]).mean()
    average_personal_income_and_number_of_incidents_per_city = pd.concat(
        [incidents_per_city, average_personal_income_of_cities], axis=1)
    print(average_personal_income_and_number_of_incidents_per_city)
    correlation = average_personal_income_and_number_of_incidents_per_city['city'].corr(
        average_personal_income_and_number_of_incidents_per_city['p_income'])
    print(
        'The correlation between number of incidents taken place in a city and the average personal income in the city is : ' + str(
            correlation))

    # Finding correlation between number of incidents taken place in a city and the average household income of the city.
    average_household_income_of_cities = data[['city', 'h_income']]
    average_household_income_of_cities = average_household_income_of_cities.groupby(
        [average_household_income_of_cities["city"]]).mean()
    average_household_income_and_number_of_incidents_per_city = pd.concat(
        [incidents_per_city, average_household_income_of_cities], axis=1)
    print(average_household_income_and_number_of_incidents_per_city)
    correlation = average_household_income_and_number_of_incidents_per_city['city'].corr(
        average_household_income_and_number_of_incidents_per_city['h_income'])
    print(
        'The correlation between number of incidents taken place in a city and the average household income in the city is : ' + str(
            correlation))

    # Finding correlation between number of incidents taken place in a city and the average unemployment rate of the city.
    average_unemployment_rate_of_cities = data[['city', 'urate']]
    average_unemployment_rate_of_cities = average_unemployment_rate_of_cities.groupby(
        [average_unemployment_rate_of_cities["city"]]).mean()
    average_unemployment_rate_and_number_of_incidents_per_city = pd.concat(
        [incidents_per_city, average_unemployment_rate_of_cities], axis=1)
    print(average_unemployment_rate_and_number_of_incidents_per_city)
    correlation = average_unemployment_rate_and_number_of_incidents_per_city['city'].corr(
        average_unemployment_rate_and_number_of_incidents_per_city['urate'])
    print(
        'The correlation between number of incidents taken place in a city and the average unemployment rate in the city is : ' + str(
            correlation))

    # Finding correlation between number of incidents taken place in a city and the average literacy rate of the city.
    average_literacy_rate_of_cities = data[['city', 'college']]
    average_literacy_rate_of_cities = average_literacy_rate_of_cities.groupby(
        [average_literacy_rate_of_cities["city"]]).mean()
    average_literacy_rate_and_number_of_incidents_per_city = pd.concat(
        [incidents_per_city, average_literacy_rate_of_cities], axis=1)
    print(average_literacy_rate_and_number_of_incidents_per_city)
    correlation = average_literacy_rate_and_number_of_incidents_per_city['city'].corr(
        average_literacy_rate_and_number_of_incidents_per_city['college'])
    print(
        'The correlation between number of incidents taken place in a city and the average literacy rate in the city is : ' + str(
            correlation))

    # Finding correlation between number of incidents taken place in a city and the average poverty rate of the city.
    average_poverty_rate_of_cities = data[['city', 'pov']]
    average_poverty_rate_of_cities = average_poverty_rate_of_cities.groupby(
        [average_poverty_rate_of_cities["city"]]).mean()
    average_poverty_rate_and_number_of_incidents_per_city = pd.concat(
        [incidents_per_city, average_poverty_rate_of_cities], axis=1)
    print(average_poverty_rate_and_number_of_incidents_per_city)
    correlation = average_poverty_rate_and_number_of_incidents_per_city['city'].corr(
        average_poverty_rate_and_number_of_incidents_per_city['pov'])
    print(
        'The correlation between number of incidents taken place in a city and the average poverty rate in the city is : ' + str(
            correlation))

    # Analysing the data by breaking down with reference to age group and ploting bar graph
    ages = data["age"].value_counts(bins=10)
    print(ages)
    ages_labels = '(30.2, 37.3]', '(23.1, 30.2] ', '(37.3, 44.4]', '(15.928, 23.1]', '(44.4, 51.5]', '(51.5, 58.6]', '(58.6, 65.7]', '(65.7, 72.8] ', '(72.8, 79.9]', '(79.9, 87.0] '
    plt.bar(x=ages_labels,
            height=ages)
    plt.hist(ages_labels, rwidth=10)
    plt.xticks(rotation=30)
    plt.title('Breakdown by Ages')
    plt.show()

    # Analysing the data by breaking down with reference to gender group and ploting pie chart
    gender_values = data["gender"].value_counts()
    print(gender_values)
    if (gender_values.Male > gender_values.Female):
        print('Most of the deceased belongs to Male group')
    else:
        print('Most of the deceased belongs to Female group')
    gender_labels = 'Male', 'Female'
    plt.pie(gender_values, labels=gender_labels, autopct='%1.1f%%')
    plt.title('Breakdown by Gender')
    plt.axis('equal')
    plt.show()

    # Analysing the data by breaking down with reference to race group and ploting pie chart
    race_values = data["raceethnicity"].value_counts()
    print(race_values)
    race_labels = 'White', 'Black', 'Hispanic/Latino', 'Asian/Pacific Islander', 'Native American'
    plt.pie(race_values, labels=race_labels, autopct='%1.1f%%')
    plt.title('Breakdown by Race')
    plt.axis('equal')
    plt.show()

    # Analysing the data by breaking down with reference to month and ploting bar graph
    month_values = data["month"].value_counts()
    print(month_values)
    month_labels = 'March', 'April', 'February', 'January', 'May', 'June'
    plt.bar(x=month_labels,
            height=month_values)
    plt.title('Breakdown by Month')
    plt.show()

    # Analysing the data by breaking down with reference to city and ploting bar graph
    city_values = data["city"].value_counts().head(5)
    print(city_values.head(5))
    city_labels = 'Los Angeles', 'Houston', 'Phoenix', 'New York', 'Oklahoma City'
    plt.bar(x=city_labels, height=city_values)
    plt.xticks(rotation=45)
    plt.title('Breakdown by City')
    plt.show()

    # Analysing the data by breaking down with reference to state and ploting bar graph
    state_values = data["state"].value_counts().head(5)
    print(state_values.head(5))
    state_labels = 'CA', 'TX', 'FL', 'AZ', 'OK'
    plt.bar(x=state_labels, height=state_values)
    plt.title('Breakdown by State')
    plt.show()

    # Analysing the data by breaking down with reference to law enforcement group and ploting bar graph
    lawenforcement_values = data["lawenforcementagency"].value_counts().head(5)
    print(lawenforcement_values.head(5))
    lawenforcement_labels = 'Los Angeles Police ', 'Oklahoma City Police', 'US Marshals Service ', 'Los Angeles County Sheriffs', 'Indianapolis Metropolitan'
    plt.bar(x=lawenforcement_labels, height=lawenforcement_values)
    plt.xticks(rotation=45)
    plt.title('Breakdown by Law Enforcement')
    plt.show()

def decision_tree_classification(data):
    numeric_data = data.select_dtypes(include=np.number)
    X = numeric_data  # independent columns
    y = data.iloc[:, 2]  # target column i.e ethnicity
    # Checking for null values in columns as presence of null values in data leads to
    # problem in fitting the data in the function used to extract significant features.
    print(X.isnull().sum())
    # apply SelectKBest class to extract top 8 best features
    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Factors', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(5, 'Score'))  # print 8 best features

    dt_numeric_data = numeric_data[['h_income', 'county_income', 'p_income', 'pop', 'pov']]
    # ['raceethnicity', 'gender', 'cause',]
    dt_data = data[['raceethnicity', 'armed', 'cause']]
    dt_data = pd.concat([dt_numeric_data, dt_data], axis=1)
    print(dt_data)

    print(dt_data['cause'].unique())


def main():
    pd.set_option('display.width', 800)
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    """Cleaning up the Police Killings dataset
    """
    # Loading data to data frame.
    data = load_data('police_killings.csv')

    # Removing the name, streetaddress, day, latitude and longitude attributes
    # as they are not necessary for the analysis
    # So dropping these columns will not affect the dataset
    remove_unnecessary_columns(data)

    # Printing the missing values
    print(data.isnull().sum())

    # Filling the missing values and cleaning the data
    data = fill_missing_values(data)

    data_analysis(data)

    decision_tree_classification(data)


if __name__ == "__main__":
    main()
