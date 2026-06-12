import pandas as pd
import unicodedata


if __name__ == "__main__":
    # If data.py is run by itself load data from original csv to process a clean version
    def remove_accents(s): # Changes "Ōsaka" to "Osaka" just in case
        return ''.join(
            c for c in unicodedata.normalize('NFKD', s)
            if not unicodedata.combining(c)
        )


    df_cities = pd.read_csv("data/japan_cities.csv")
    df_cities = df_cities.iloc[:, [0, 1, 2, 7]] # Only keep the info we need
    df_cities["city_clean"] = df_cities["city"].apply(remove_accents)
    df_cities.to_csv("data/cleaned.csv", index=False)

cities = pd.read_csv("data/cleaned.csv") # From any file use `from data import cities` to get city data