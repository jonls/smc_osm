import json
import time
import urllib.request


def main():
    url_base = 'https://gis.smcgov.org/maps/rest/services'
    feature = 'ISD/COUNTY_GIS/FeatureServer/0'

    base_data = None
    offset = 0
    count = 1000
    while True:
        full_url = f'{url_base}/{feature}/query?f=geojson&resultRecordCount={count}&resultOffset={offset}&where=1%3D1&orderByFields=OBJECTID&outFields=*'
        print(full_url)

        response = urllib.request.urlopen(full_url)
        data = response.read()
        geo_data = json.loads(data)

        if len(geo_data['features']) < count:
            break

        if base_data is None:
            base_data = geo_data
        else:
            base_data['features'].extend(geo_data['features'])

        offset += count
        time.sleep(0.1)

    with open('data/smc_address.geojson', 'w') as f:
        json.dump(base_data, f)


if __name__ == '__main__':
    main()
