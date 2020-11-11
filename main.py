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


if __name__ == "__main__":
    main()
