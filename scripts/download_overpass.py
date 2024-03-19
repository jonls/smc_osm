import argparse
import urllib.request
from urllib.parse import quote_plus


def download_overpass(query: str, output_file: str) -> None:
    base_url = 'https://overpass-api.de/api/interpreter'
    data_param = quote_plus(query)
    full_url = f'{base_url}?data={data_param}'
    response = urllib.request.urlopen(full_url)
    data = response.read()

    with open(output_file, 'wb') as f:
        f.write(data)


def download_buildings():
    download_overpass(
        query='''
            [out:json];
            area[name="San Mateo County"];
            (
                way[building](area);
                relation[building](area);
            );
            (._;>;);
            out;
        ''',
        output_file='data/osm_buildings.json',
    )


def download_address_points():
    download_overpass(
        query='''
            [out:json];
            area[name="San Mateo County"];
            node["addr:housenumber"]["addr:street"](area);
            out;
        ''',
        output_file='data/osm_address_points.json',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('download', choices=['buildings', 'address_points'])
    args = parser.parse_args()

    if args.download == 'buildings':
        download_buildings()
    elif args.download == 'address_points':
        download_address_points()
    else:
        parser.error('Invalid download')


if __name__ == '__main__':
    main()
