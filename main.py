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

    print(data["gender"].value_counts())
    print(data["raceethnicity"].value_counts())
    data = data[data.raceethnicity != 'Unknown']
    print(data["armed"].value_counts())
    print(data["cause"].value_counts())
    data = data[data.armed != 'Unknown']
    data = data[data.cause != 'Unknown']
    print(len(data))
    return data

def main():
    pd.set_option('display.width', 800)
    # pd.set_option('display.max_rows', None)
    """Cleaning up the Police Killings dataset
    """
    # Loading data to data frame.
    data = load_data('police_killings.csv')

    # Removing the name, streetaddress, day, latitude and longitude attributes
    # as they are not necessary for the analysis
    # So dropping these columns will not affect the dataset
    remove_unnecessary_columns(data)
    data = fill_missing_values(data)
    # Printing the missing values
    print(data.isnull().sum())



if __name__ == "__main__":
    main()
