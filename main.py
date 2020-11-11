import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename):
    """Loading a csv file into a Pandas dataframe.
        filename - string
        return Pandas dataframe
    """
    return pd.read_csv(filename, encoding='ISO-8859-1')


def remove_unnecessary_columns(data):
    data.drop(['name', 'streetaddress', 'day', 'latitude', 'longitude', 'geo_id'], axis=1, inplace=True)


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

    return data


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


if __name__ == "__main__":
    main()
