import json
import math
import re
from collections import Counter
from itertools import islice
from typing import Iterable
from typing import Iterator
from typing import TypeVar

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree

T = TypeVar('T')

EARTH_RADIUS = 6_371  # km
EARTH_CIRCUMFERENCE = EARTH_RADIUS * 2 * math.pi  # km


def batched(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch


def normalize_street_type(t: str) -> str | None:
    standard_types = {
        'RD': 'Road',
        'WAY': 'Way',
        'CT': 'Court',
        'CIR': 'Circle',
        'AVE': 'Avenue',
        'PL': 'Place',
        'ST': 'Street',
        'DR': 'Drive',
        'LN': 'Lane',
        'BLVD': 'Boulevard',
        'HWY': 'Highway',
        'TER': 'Terrace',
        'ROW': 'Row',
        'AV': 'Avenue',
        'WY': 'Way',
        'LANE': 'Lane',
        'PKWY': 'Parkway',
        'CRES': 'Crescent',
        'LNDG': 'Landing',
        'SQ': 'Square',
        'CV': 'Cove',
        'TRL': 'Trail',
        'ALLEY': 'Alley',
        'PASS': 'Pass',
        'ISLE': 'Isle',
        'BND': 'Bend',
        'CTR': 'Center',
        'PLZ': 'Plaza',
        'VIEW': 'View',
        'CENT': 'Center',
        'RDG': 'Ridge',
        'WALK': 'Walk',
        'HIGHWAY': 'Highway',
        'VISTA': 'Vista',
        'XING': 'Crossing',
        'COVE': 'Cove',
        'GLN': 'Glen',
    }
    return standard_types.get(t.upper())


def normalize_street_direction(d: str) -> str | None:
    standard_dirs = {
        'N': 'North',
        'S': 'South',
        'E': 'East',
        'W': 'West',
    }
    return standard_dirs.get(d.upper())


def title_case_street_name(s: str) -> str:
    def title(s: str) -> str:
        if re.match(r'\d+', s):
            return s.lower()
        return s.title()
    return ' '.join(title(x) for x in s.split())


def is_closer_than(
    coords1: np.array,
    coords2: np.array,
    dist_km: float,
) -> bool:
    approx_ang_dist = dist_km / EARTH_CIRCUMFERENCE * 360
    return np.linalg.norm(coords1 - coords2) <= approx_ang_dist


def iter_as_circle(iterable: Iterable[T]) -> Iterator[T]:
    """Iterate but yield the first element again as the last element."""
    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration:
        return

    yield first
    yield from it
    yield first


def get_local_space(l: np.array) -> np.array:
    """Create local space for input lat/lon

    :return: local space (shape: 3x3)
    """
    theta = math.pi/2 + -np.deg2rad(l[0])
    phi = np.deg2rad(l[1])

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    local_v1 = np.array([
        -cos_phi * cos_theta,
        -sin_phi * cos_theta,
        sin_theta,
    ])
    local_v2 = np.array([
        -sin_phi,
        cos_phi,
        0,
    ])
    normal = np.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta,
    ])
    return np.column_stack([local_v1, local_v2, normal])


def to_local_plane(l: np.array, local_space: np.array) -> np.array:
    """Transform input (lat, lon)s into local plane.

    :param l: Lat/lon pairs in degrees (shape: Nx2)
    :param local_space: Local space (shape: 3x3)
    :return: local plane coordinates (shape: Nx2)
    """
    theta = np.pi/2 - np.deg2rad(l[:,0])
    phi = np.deg2rad(l[:,1])
    sin_theta = np.sin(theta)
    x = EARTH_RADIUS * np.column_stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        np.cos(theta),
    ])
    u = local_space.T.dot(x.T).T
    return u[:,:2]


def to_world_latlon(u: np.array, local_space: np.array) -> np.array:
    """Transform us in local space in (lat, lon) coords.

    :param u: Local space coordinates (shape: Nx2)
    :param local_space: Local space (shape: 3x3)
    :return: World lat/lon coordinates in degrees (shape: Nx2)
    """
    u_len = np.linalg.norm(u.T, axis=0)
    f = np.sqrt((EARTH_RADIUS - u_len) * (EARTH_RADIUS + u_len))
    x = local_space.dot(np.column_stack([u, f]).T).T
    theta = np.arccos(x[:,2] / EARTH_RADIUS)
    phi = np.arctan2(x[:,1], x[:,0])
    return np.column_stack([
        (math.pi/2 - theta)*180/math.pi,
        phi*180/math.pi,
    ])


def main():
    # Parse OSM buildings
    with open('data/osm_buildings.json', 'r') as f:
        buildings_data = json.load(f)

    global_osm_nodes = {}
    global_osm_ways = {}

    buildings = {}
    building_address_index = {}
    buildings_address_tags = {}
    buildings_all_tags = {}
    for osm_elem in buildings_data['elements']:
        osm_id = osm_elem['id']
        osm_type = osm_elem['type']
        tags = osm_elem.get('tags', {})
        if osm_type == 'node':
            global_osm_nodes[osm_id] = np.array([
                osm_elem['lat'],
                osm_elem['lon'],
            ], dtype=float)
        elif osm_type == 'way':
            global_osm_ways[osm_id] = np.array(
                osm_elem['nodes'],
                dtype=int,
            )
            if tags.get('building', 'no') != 'no':
                buildings[osm_id] = 'way', None
        elif osm_type == 'relation':
            relation_type = tags.get('type')
            if tags.get('building', 'no') == 'no':
                name_tag = tags.get('name', '<unknown>')
                print(
                    f'Skip relation {osm_id} ({name_tag}):'
                    f' Not a building!'
                )
            elif relation_type != 'multipolygon':
                name_tag = tags.get('name', '<unknown>')
                print(
                    f'Skip relation {osm_id} ({name_tag}):'
                    f' Unknown relation type: {relation_type}!'
                )
            else:
                outer_members = []
                for member in osm_elem['members']:
                    if member.get('type') == 'way' and member.get('role') == 'outer':
                        outer_members.append(member)

                if len(outer_members) != 1:
                    name_tag = tags.get('name', '<unknown>')
                    print(
                        f'Skip relation {osm_id} ({name_tag}):'
                        f' Multiple outer roles not supported: {len(outer_members)}!'
                    )
                else:
                    buildings[osm_id] = 'relation', outer_members[0]['ref']
        else:
            print(f'Unhandled type {osm_id}: {osm_type}')

        if (
            (house_number := tags.get('addr:housenumber')) and
            (street := tags.get('addr:street'))
        ):
            city = tags.get('addr:city')
            if city:
                building_address_index[house_number, street, city] = osm_id
            buildings_address_tags[osm_id] = house_number, street, city

        buildings_all_tags[osm_id] = tags

    print(f'Registered {len(buildings)} buildings')

    # Parse OSM address points
    with open('data/osm_address_points.json', 'r') as f:
        address_points_data = json.load(f)

    point_address_index = {}
    point_address_tags = {}
    for osm_elem in address_points_data['elements']:
        osm_id = osm_elem['id']
        osm_type = osm_elem['type']
        if osm_type == 'node':
            global_osm_nodes[osm_id] = np.array([
                osm_elem['lat'],
                osm_elem['lon'],
            ], dtype=float)

        tags = osm_elem.get('tags', {})
        if (
            (house_number := tags.get('addr:housenumber')) and
            (street := tags.get('addr:street'))
        ):
            city = tags.get('addr:city')
            if city:
                point_address_index[house_number, street, city] = osm_id
            point_address_tags[osm_id] = house_number, street, city

    print(f'Registered {len(point_address_index)} point addresses')

    # Calculate building bounds and centroids
    buildings_bounds = {}
    buildings_centroids = {}
    buildings_major_axis = {}
    for building_id, (osm_type, ref) in buildings.items():
        if osm_type == 'way':
            nodes = global_osm_ways[building_id]
        elif osm_type == 'relation':
            nodes = global_osm_ways[ref]
        else:
            continue

        if len(nodes) < 4 or nodes[0] != nodes[-1]:
            print(f'Invalid building geometry: {building_id}')
            continue

        points = np.array([global_osm_nodes[node_id] for node_id in nodes])
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        buildings_bounds[building_id] = np.array([mins, maxs]).T

        centroid = (mins + maxs) / 2
        buildings_centroids[building_id] = centroid

        local_space = get_local_space(centroid)
        local_points = to_local_plane(l=points, local_space=local_space)

        # Convex hull
        hull = ConvexHull(local_points)

        min_bounding_box_perimeter = float('inf')
        min_values = None
        for a, b, _ in hull.equations:
            rotation = np.column_stack([[a, b], [-b, a]])
            rot_ps = rotation.T.dot(hull.points[hull.vertices].T).T
            rot_mins = np.min(rot_ps, axis=0)
            rot_maxs = np.max(rot_ps, axis=0)
            perimeter = 2 * np.sum(rot_maxs - rot_mins)
            if perimeter < min_bounding_box_perimeter:
                min_bounding_box_perimeter = perimeter
                min_normal = np.array([a, b])
                min_bounds = np.array([rot_mins, rot_maxs]).T
                min_values = local_space, min_normal, min_bounds

        if min_values is not None:
            buildings_major_axis[building_id] = min_values

    # Parse GeoJson address data
    with open('data/smc_address.geojson', 'r') as f:
        address_data = json.load(f)

    addresses = {}
    building_far_from_address = {}
    point_address_far_from_address = {}
    buildings_matched_to_address = {}
    for feature in address_data['features']:
        feature_id = feature['id']
        lon, lat = feature['geometry']['coordinates']
        coords = np.array([lat, lon])
        properties = feature['properties']

        # Street
        if not (street := properties['STREET']) or street.strip() == '':
            print(f'No street for ID={feature_id}: full address: {properties.get("FULL_ADDR")}')
            continue

        # Check for NA markers
        if street.lower() == 'no data available':
            print(f'No street for ID={feature_id}: full address: {properties.get("FULL_ADDR")}')
            continue

        # Title case street
        street = title_case_street_name(street)

        # Special cases for street name correction
        street = re.sub(r'De Las Pulgas$', 'de las Pulgas', street)
        street = re.sub(r'Governor\'S', 'Governors', street)
        street = re.sub(r'St (Francis|Martin|Mary|James)', 'Saint \g<1>', street)
        street = re.sub(r'Corriente Pointe', 'Corriente Point', street)
        street = re.sub(r'Mc ', 'Mc', street)

        # House number
        if not (house_number := properties['HOUSE_NUM']) or house_number.strip == '':
            print(f'No house number for ID={feature_id}: full address: {properties.get("FULL_ADDR")}')
            continue

        # Check for NA markers
        if house_number.lower() == 'no data available':
            print(f'No house number for ID={feature_id}: full address: {properties.get("FULL_ADDR")}')
            continue

        # City
        if not (city := properties['CITY']) or city.strip() == '':
            print(f'No city for ID={feature_id}: full address: {properties.get("FULL_ADDR")}')
            continue

        # Check for NA markers
        if city.lower() == 'no data available':
            print(f'No city for ID={feature_id}: full address: {properties.get("FULL_ADDR")}')
            continue

        city = ' '.join(s.title() for s in city.split())

        # Normalize street type
        if (street_type := properties['TYPE']):
            if street_type.strip() == '':
                street_type = None
            else:
                norm_type = normalize_street_type(street_type)
                if norm_type is None:
                    # Special case REAL
                    if street_type.upper() == 'REAL':
                        street_type = None
                        if not street.upper().endswith(' REAL'):
                            street += ' Real'
                    else:
                        print(f'Could not normalize street type: {street_type}, street: {street}, full address: {properties.get("FULL_ADDR")}')
                        continue

                street_type = norm_type

        # Direction
        if (direction := properties['DIRECTION']):
            if direction.strip() == '':
                direction = None
            else:
                norm_dir = normalize_street_direction(direction)
                if norm_dir is None:
                    print(f'Could not normalize street direction: {direction}, street: {street}, full address: {properties.get("FULL_ADDR")}')
                    continue

                direction = norm_dir

        # Fraction
        if (fraction := properties['FRACTION']):
            if fraction.strip() == '':
                fraction = None
            else:
                print(f'Address has a fraction: {fraction}, street: {street}, full address: {properties.get("FULL_ADDR")}')
                continue

        full_street = street
        if direction:
            full_street = direction + ' ' + full_street
        if street_type:
            full_street = full_street + ' ' + street_type

        # Lookup in building address index and match
        if (building_id := building_address_index.get((house_number, full_street, city))):
            # Check if this is the expected location
            if (
                building_id in buildings_centroids and
                not is_closer_than(coords, buildings_centroids[building_id], dist_km=0.3)
            ):
                building_far_from_address[building_id] = (
                    f'{house_number} {full_street}, {city}',
                    coords,
                )
            else:
                buildings_matched_to_address[building_id] = feature_id

            continue

        # Lookup in address point index
        if (point_address_id := point_address_index.get((house_number, full_street, city))):
            # Check if this is the expected location
            if (
                point_address_id in global_osm_nodes and
                not is_closer_than(coords, global_osm_nodes[point_address_id], dist_km=0.3)
            ):
                point_address_far_from_address[point_address_id] = (
                    f'{house_number} {full_street}, {city}',
                    coords,
                )

            continue

        addresses[feature_id] = (
            coords,
            (house_number, full_street, city),
            properties,
        )

    count_issues = len(building_far_from_address) + len(point_address_far_from_address)
    print(f'Writing building_far_from_address issues... ({count_issues} issues)')
    with open('issues/building_far_from_address_issues.geojson', 'w') as f:
        # Write buildings
        for building_id, (address, coords) in sorted(building_far_from_address.items()):
            f.write('\x1e')

            osm_type, ref = buildings[building_id]
            if osm_type == 'way':
                nodes = global_osm_ways[building_id]
            elif osm_type == 'relation':
                nodes = global_osm_ways[ref]
            else:
                continue

            latlon_coords = np.array([
                global_osm_nodes[node_id]
                for node_id in nodes
            ])
            polygon_coords = np.column_stack((latlon_coords[:,1], latlon_coords[:,0])).tolist()

            geo_doc = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [
                                polygon_coords,
                            ],
                        },
                        'properties': {
                            'id': building_id,
                            'parsed_address': address,
                            'point_url': f'https://www.openstreetmap.org/search?query={coords[0]}%2C%20{coords[1]}',
                        },
                    },
                ],
            }
            json.dump(geo_doc, f)
            f.write('\n')

        # Write point addresses
        for node_id, (address, coords) in point_address_far_from_address.items():
            f.write('\x1e')

            lat, lon = global_osm_nodes[node_id]
            geo_doc = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [
                                lon, lat
                            ],
                        },
                        'properties': {
                            'id': node_id,
                            'parsed_address': address,
                            'point_url': f'https://www.openstreetmap.org/search?query={coords[0]}%2C%20{coords[1]}',
                        },
                    },
                ],
            }
            json.dump(geo_doc, f)
            f.write('\n')

    print(f'Registered {len(addresses)} addresses')

    print('Building spatial index of buildings...')

    centroids = []
    centroids_index = {}
    for building_id, coords in buildings_centroids.items():
        centroids_index[len(centroids)] = building_id
        centroids.append(coords)

    centroids_kdtree = KDTree(np.array(centroids))

    # Search parameters
    search_dist = 200 # meters
    # Approximate angular radius for the given distance
    search_dist_ang = (search_dist / 1000) / EARTH_CIRCUMFERENCE * 360

    print('Searching against spatial index of buildings...')

    # Find building matches for OSM addresses
    osm_point_to_building_matches = {}
    building_to_osm_point_matches = {}
    for osm_batch in batched(point_address_tags.items(), n=1000):
        address_query = [global_osm_nodes[osm_id] for osm_id, _ in osm_batch]
        batch_result = centroids_kdtree.query_ball_point(x=address_query, r=search_dist_ang)
        for (osm_id, (house_number, street, _)), matches in zip(osm_batch, batch_result):
            coord = global_osm_nodes[osm_id]
            for match in matches:
                building_id = centroids_index[match]

                # Check if the search point is inside the oriented bounding box of the building
                local_space, (a, b), bounds = buildings_major_axis[building_id]
                rotation = np.column_stack([[a, b], [-b, a]])
                u, = to_local_plane(l=np.array([coord]), local_space=local_space)
                rot_coord = rotation.T.dot(u.T).T
                if (bounds[:,0] <= rot_coord).all() and (rot_coord <= bounds[:,1]).all():
                    building_to_osm_point_matches.setdefault(building_id, {}).setdefault((house_number, street), []).append(osm_id)
                    osm_point_to_building_matches.setdefault(osm_id, []).append(building_id)

    # Find building matches for external addresses
    missing_building_areas = Counter()
    address_to_building_matches = {}
    building_to_address_matches = {}
    for address_batch in batched(addresses.items(), n=1000):
        address_query = [coord for _, (coord, _, _) in address_batch]
        batch_result = centroids_kdtree.query_ball_point(x=address_query, r=search_dist_ang)
        for (feature_id, (coord, (house_number, street, _), _)), matches in zip(address_batch, batch_result):
            if len(matches) == 0:
                # Cluster locations within a square
                c_lat_sq = math.floor(coord[0] / search_dist_ang)
                c_lon_sq = math.floor(coord[1] / search_dist_ang)
                missing_building_areas[c_lat_sq, c_lon_sq] += 1
                continue

            for match in matches:
                building_id = centroids_index[match]
                # Check if the search point is inside the oriented bounding box of the building
                local_space, (a, b), bounds = buildings_major_axis[building_id]
                rotation = np.column_stack([[a, b], [-b, a]])
                u, = to_local_plane(l=np.array([coord]), local_space=local_space)
                rot_coord = rotation.T.dot(u.T).T
                if (bounds[:,0] <= rot_coord).all() and (rot_coord <= bounds[:,1]).all():
                    building_to_address_matches.setdefault(building_id, {}).setdefault((house_number, street), []).append(feature_id)
                    address_to_building_matches.setdefault(feature_id, []).append(building_id)

    print(f'Found {len(osm_point_to_building_matches)} osm point to building matches')
    print(f'Found {len(building_to_osm_point_matches)} building to osm point matches')

    print(f'Writing building_missing_from_area issues... ({len(missing_building_areas)} issues)')
    with open('issues/building_missing_from_area.geojson', 'w') as f:
        # Write area
        for (lat_sq, lon_sq), count in sorted(missing_building_areas.most_common()):
            f.write('\x1e')

            min_lat = lat_sq * search_dist_ang
            max_lat = (lat_sq + 1) * search_dist_ang
            min_lon = lon_sq * search_dist_ang
            max_lon = (lon_sq + 1) * search_dist_ang

            external_id = f'{search_dist}/{lat_sq}/{lon_sq}'
            boundary_coords = [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
            ]

            geo_doc = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [boundary_coords],
                        },
                        'properties': {
                            'external_id': external_id,
                            'count': count,
                        },
                    },
                ],
                'attachments': [
                    {
                        'id': external_id,
                        'kind': 'referenceLayer',
                        'type': 'geojson',
                        'name': 'Boundary',
                        'data': {
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Polygon',
                                'coordinates': [boundary_coords],
                            },
                        },
                    },
                ],
            }
            json.dump(geo_doc, f)
            f.write('\n')

    print(f'Found {len(address_to_building_matches)} address to building matches')
    print(f'Found {len(building_to_address_matches)} building to address matches')

    # Find one-to-one matches
    one_to_one_matches = {}
    for building_id, address_matches in building_to_address_matches.items():
        # Filter if there are more than one match, or if building was already
        # matched perfectly by address.
        if len(address_matches) != 1 or building_id in buildings_matched_to_address:
            continue

        street_addr, feature_ids = next(iter(address_matches.items()))
        if all(
            len(address_to_building_matches[feature_id]) == 1 and
            address_to_building_matches[feature_id][0] == building_id
            for feature_id in feature_ids
        ):
            one_to_one_matches[building_id] = feature_ids, street_addr

    print(f'Found {len(one_to_one_matches)} one-to-one matches')

    # Find cases where the building already has an address tagging
    mismatched_building_number_tags = {}
    mismatched_building_street_tags = {}
    mismatched_building_both_tags = {}
    for building_id, (feature_ids, street_addr) in one_to_one_matches.items():
        if not (osm_addr := buildings_address_tags.get(building_id)):
            continue

        # Derive city
        cities = set()
        for feature_id in feature_ids:
            _, (_, _, city), _ = addresses[feature_id]
            if city is not None:
                cities.add(city)

        osm_house_number, osm_street, _ = osm_addr
        if (osm_house_number, osm_street) != street_addr:
            if osm_street == street_addr[1]:
                mismatched_building_number_tags[building_id] = feature_ids, street_addr, cities, osm_house_number
            elif osm_house_number == street_addr[0]:
                mismatched_building_street_tags[building_id] = feature_ids, street_addr, cities, osm_street
            else:
                mismatched_building_both_tags[building_id] = feature_ids, street_addr, cities, (osm_street, osm_house_number)

    print(f'Found {len(mismatched_building_number_tags)} buildings with mismatched house number tags')
    print(f'Found {len(mismatched_building_street_tags)} buildings with mismatched street tags')
    print(f'Found {len(mismatched_building_both_tags)} buildings with both mismatched tags')

    print(f'Writing mismatched_building_street_tags issues... ({len(mismatched_building_street_tags)} issues)')
    with open('issues/mismatched_building_street_tags.geojson', 'w') as f:
        included_mismatches = set()
        for building_id, (feature_ids, street_addr, cities, osm_street) in sorted(mismatched_building_street_tags.items()):
            osm_addr = buildings_address_tags[building_id]

            osm_type, ref = buildings[building_id]
            if osm_type == 'way':
                nodes = global_osm_ways[building_id]
            elif osm_type == 'relation':
                nodes = global_osm_ways[ref]
            else:
                continue

            latlon_coords = np.array([
                global_osm_nodes[node_id]
                for node_id in nodes
            ])
            polygon_coords = np.column_stack((latlon_coords[:,1], latlon_coords[:,0])).tolist()

            properties = {
                'id': f'{osm_type}/{building_id}',
            }
            existing_tags = buildings_all_tags[building_id]
            for tag in ('addr:city', 'addr:street', 'addr:housenumber'):
                if tag in existing_tags:
                    properties[tag] = existing_tags[tag]

            _, street = street_addr
            properties['proposed_street'] = street

            # Only include each specific correction the first time, to avoid
            # a lot of duplicated tasks.
            if (street, osm_street) in included_mismatches:
                continue

            included_mismatches.add((street, osm_street))

            geo_doc = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [polygon_coords],
                        },
                        'properties': properties,
                    },
                ],
            }
            f.write('\x1e')
            json.dump(geo_doc, f)
            f.write('\n')

    print(f'Writing mismatched_building_number_tags issues... ({len(mismatched_building_number_tags)} issues)')
    with open('issues/mismatched_building_number_tags.geojson', 'w') as f:
        for building_id, (feature_ids, street_addr, cities, osm_house_number) in sorted(mismatched_building_number_tags.items()):
            osm_addr = buildings_address_tags[building_id]

            osm_type, ref = buildings[building_id]
            if osm_type == 'way':
                nodes = global_osm_ways[building_id]
            elif osm_type == 'relation':
                nodes = global_osm_ways[ref]
            else:
                continue

            latlon_coords = np.array([
                global_osm_nodes[node_id]
                for node_id in nodes
            ])
            polygon_coords = np.column_stack((latlon_coords[:,1], latlon_coords[:,0])).tolist()

            properties = {
                'id': f'{osm_type}/{building_id}',
            }
            existing_tags = buildings_all_tags[building_id]
            for tag in ('addr:city', 'addr:street', 'addr:housenumber'):
                if tag in existing_tags:
                    properties[tag] = existing_tags[tag]

            house_number, _ = street_addr
            properties['proposed_house_number'] = house_number

            geo_doc = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [polygon_coords],
                        },
                        'properties': properties,
                    },
                ],
            }
            f.write('\x1e')
            json.dump(geo_doc, f)
            f.write('\n')


if __name__ == '__main__':
    main()
