"""
Extract populated places from the GeoNames allCountries.txt database and
generate structured city data for prompt creation.

The script filters GeoNames records with feature class 'P' (populated places)
and keeps only locations with a population greater than the specified threshold.
The extracted data includes geographic coordinates, administrative hierarchy,
country information, feature codes, and population statistics.

Download and extract GeoNames database:
    wget -nc https://download.geonames.org/export/dump/allCountries.zip
    python -c "import zipfile; zipfile.ZipFile('allCountries.zip').extractall('.')"

Example:
    python extract.py --file allCountries.txt --min_pop 100000
"""

import pandas as pd

def admin_level_helper(admins):
    for i, admin in enumerate(admins, 0):
        if(admin == ""):
            return i
    return 4


def extract_populated_places(data_path, min_popluation):
    print(f"Extracting cities...")

    cities_prompts = []

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            parts = line.strip().split("\t")

            feature_class = parts[6]

            geoname_id = int(parts[0])
            name = parts[1]
            asciiname = parts[2]
            lat = float(parts[4])
            lon = float(parts[5])
            feature_code = parts[7]
            country_code = parts[8]
            admin1 = parts[10]
            admin2 = parts[11]
            admin3 = parts[12]
            admin4 = parts[13]
            admin_level = admin_level_helper([admin1, admin2, admin3, admin4])
            population = int(parts[14]) if parts[14] else 0

            if feature_class == "P":
                if population > min_popluation:
                    cities_prompts.append((
                        geoname_id,
                        name,
                        asciiname,
                        lat,
                        lon,
                        feature_class,
                        feature_code,
                        country_code,
                        admin1,
                        admin2,
                        admin3,
                        admin4,
                        population,
                        admin_level
                    ))

            if i % 1_000_000 == 0:
                print(f"    Processed {i:,} lines...")

    return cities_prompts


def main():
    import argparse

    parser = argparse.ArgumentParser(
            description="""Extract cities (P feature class) with more than <min_pop> population from GeoNames allCountries.txt file.\nDownload data there:
            wget -nc https://download.geonames.org/export/dump/allCountries.zip 
            python -c import zipfile; zipfile.ZipFile('allCountries.zip').extractall('.')
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            usage="python %(prog)s --file allCountries.txt --min_pop 10000 [--output_path prompt_world.pkl]",
    )
    parser.add_argument("--file",           type=str,   required=True,                  help="Path to GeoNames allCountries.txt file")
    parser.add_argument("--min_pop",        type=int,   default=10000,                  help="Minimum population threshold")
    parser.add_argument("--output_path",    type=str,   default="prompts_world.pkl",        help="Path to save output prompts as pickle file")

    args = parser.parse_args()
    
    columns_name =["geoname_id", "name", "asciiname", "lat", "lon", "feature_class", "feature_code", 
                   "country_code", "admin1", "admin2", "admin3", "admin4", "population", "admin_level"]

    cities_prompts = extract_populated_places(args.file, args.min_pop)
    df_cities = pd.DataFrame(cities_prompts, columns=columns_name)

    df_cities.to_pickle(args.output_path)
    print(f"{df_cities.shape[0]} prompts saved at {args.output_path}")

if __name__ == "__main__":
    main()