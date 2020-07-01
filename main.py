import geopandas as gpd
from shapely import wkt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import LineString
from timeit import default_timer as timer
from math import atan2, degrees
import argparse

pd.options.mode.chained_assignment = None  # default='warn'


def read_shp(path):
    """Reads a .shp file to a GeoPandas Data Frame

    Arguments:
        path {String} -- A string containing the path to the .shp file.

    Returns:
        [GeoPandas Data Frame] -- A GeoPandas Data Frame containing the data read from the .shp file.
    """
    df = gpd.read_file(path)
    df.set_index('id', inplace=True)
    df.sort_index(inplace=True)
    return df


def get_begin(points):
    return points[0].wkt


def get_end(points):
    return points[1].wkt


def get_lines(df):
    """This function determines which lines do not have a follow up line, 
    and which lines do not have a lead line.

    It does this by comparing the exterior points of a line to the exterior
    points of all other lines. If the exterior point of line a cannot be found
    in the exterior points of any other line, we conclude that line a has no
    lead/ follow-up line.


    Arguments:
        df {GeoPandas DataFrame} -- A GeoPandas DataFrame containing the line data.

    Returns:
        [Tuple] -- A tuple of the lines without a follow-up line, and lines without 
        a lead line, in this order.
    """
    df['b_point'] = df.geometry.boundary.apply(get_begin)
    df['e_point'] = df.geometry.boundary.apply(get_end)
    lines_no_end = df[~df.e_point.isin(df.b_point)]
    lines_no_begin = df[~df.b_point.isin(df.e_point)]
    return lines_no_end, lines_no_begin


def AngleBtw2Points(point_a, point_b):
    change_x = point_b[0] - point_a[0]
    change_y = point_b[1] - point_a[1]
    return degrees(atan2(change_y, change_x))


def calculate_neighbors(lines_no_end, lines_no_begin):
    """This function calculates the closest neighbors for each
    line in lines_no_end.

    Arguments:
        lines_no_end {GeoPandas DataFrame} -- A GeoPandas DataFrame containing the lines with no follow-up line.
        lines_no_begin {GeoPandas DataFrame} -- A GeoPandas DataFrame containing the lines with no lead line.

    Returns:
        [list] -- A list containing the indeces of the closest lines for each of the lines in lines_no_end.
    """
    end_points = gpd.GeoSeries(lines_no_end.e_point.apply(wkt.loads))
    begin_points = gpd.GeoSeries(lines_no_begin.b_point.apply(wkt.loads))

    gd_a, gd_b = end_points, begin_points

    n_a = np.array(list(zip(gd_a.geometry.x, gd_a.geometry.y)))
    n_b = np.array(list(zip(gd_b.geometry.x, gd_b.geometry.y)))
    neighbors = NearestNeighbors(n_neighbors=3).fit(n_a)
    distances, indices = neighbors.kneighbors(n_b)
    return indices


def convert_to_id(column, df):
    return column.apply(lambda x: df.index[x])


def generate_artificial_lines(lines_no_begin, lines_no_end):
    """This function generates the data for artificial lines.

    Arguments:
        lines_no_end {GeoPandas DataFrame} -- A GeoPandas DataFrame containing the lines with no follow-up line.
        lines_no_begin {GeoPandas DataFrame} -- A GeoPandas DataFrame containing the lines with no lead line.

    Returns:
        [GeoPandas DataFrame] -- A GeoPandas DataFrame containing the data for the artificial lines.
    """

    begin_points = lines_no_begin['b_point']
    end_points = lines_no_end.loc[lines_no_begin['first'], 'e_point']

    artificial_lines = pd.DataFrame({'b_idx': list(begin_points.index),
                                     'b_points': list(begin_points),
                                     'e_idx': list(end_points.index),
                                     'e_points': list(end_points)})
    end_points = gpd.GeoSeries(artificial_lines.e_points.apply(wkt.loads))
    begin_points = gpd.GeoSeries(artificial_lines.b_points.apply(wkt.loads))

    artificial_lines['geometry'] = gpd.GeoSeries([LineString(x) for x in list(zip(end_points, begin_points))])
    artificial_lines.drop(['b_points', 'e_points'], axis=1, inplace=True)
    artificial_lines = gpd.GeoDataFrame(artificial_lines)
    artificial_lines.set_index(['e_idx', 'b_idx'], inplace=True)
    artificial_lines.sort_index(inplace=True)
    return artificial_lines


def calculate_cos(lines_no_begin, artificial_lines, lines_no_end, angle_treshold):
    filtered_lines = []

    for name, group in artificial_lines.groupby(level=0):

        min_degree = np.inf
        min_degree_art = np.inf
        max_idx = np.inf

        for idx, line in enumerate(group.geometry.to_numpy(copy=True)):

            line1 = lines_no_end.loc[group.iloc[idx, :].name[0], :].geometry.coords[-2:]
            line2 = lines_no_begin.loc[group.iloc[idx, :].name[1], :].geometry.coords[:2]
            art_line = list(line.coords)

            angle1 = AngleBtw2Points(line1[0], line1[1])
            angle2 = AngleBtw2Points(line2[0], line2[1])
            art_angle = AngleBtw2Points(art_line[0], art_line[1])

            degree = abs(angle1 - angle2)
            degree_art = abs(art_angle - angle2)

            if degree < min_degree:
                min_degree = degree
                max_idx = idx
                min_degree_art = degree_art

        row = {'e_idx': group.iloc[max_idx, :].name[0],
               'b_idx': group.iloc[max_idx, :].name[1],
               'angle': min_degree,
               'angle_art': min_degree_art,
               'geometry': group.iloc[max_idx, 0]}

        filtered_lines.append(row)

    filtered_lines = gpd.GeoDataFrame(filtered_lines)
    filtered_lines.set_index(['e_idx', 'b_idx'], inplace=True)
    filtered_lines.sort_index(inplace=True)
    same_begin_end = filtered_lines.index.get_level_values('e_idx') == filtered_lines.index.get_level_values('b_idx')
    filtered_lines = filtered_lines[
        (filtered_lines.angle < angle_treshold) & ~same_begin_end & (filtered_lines.angle_art < angle_treshold)]
    return filtered_lines


def write_shapefile(df, path):
    df.crs = 'EPSG:4326'
    df.to_file(path)


def prepare_df_for_concatenation(df, df2):

    df['e_idx'] = df.index.get_level_values(0)
    df['b_idx'] = df.index.get_level_values(1)

    df['naam'] = "Artificial_" + df['e_idx'].astype(str) + "_" + df['b_idx'].astype(str)
    df['dgl_loc'] = "_"
    df['ref_loc'] = "_"
    df['lengte'] = 0
    df['wegtype'] = "_"
    df['meetgeg'] = "_"
    df['ref_begin'] = "_"
    df['ref_eind'] = "_"

    # df.drop(columns=['e_idx', 'b_idx', 'angle', 'angle_art'], inplace=True)
    df = df[df2.columns]

    return df


def print_options(args):
    print(f"Running script with the following options:")
    print(f" - File path: {args.p},")
    print(f" - Acceptable angle: {args.d}")
    print(f" - Output path: {args.o}")


def main(file_path, angle_threshold, output_path):

    # Read the shapefile into a GeoPandas DataFrame.
    df = read_shp(file_path)
    result_df = df.copy()
    result = gpd.GeoDataFrame(columns=df.columns)

    # Identify lines that do not connect to another line at their begin or end point.
    lines_no_end, lines_no_begin = get_lines(df)

    # Generate artificial lines
    result_lines = gpd.GeoDataFrame()
    previous_result_length = 0

    # loop stops when iteration finds less than 5 new artificial lines.
    while 1:

        # Calculate and store the id of the nearest line.
        lines_no_begin['first'] = calculate_neighbors(lines_no_end, lines_no_begin)[:, 0]
        lines_no_begin['first'] = convert_to_id(lines_no_begin['first'], lines_no_end)

        # Generates artificial lines between all lines without an end, with the nearest line.
        artificial_lines = generate_artificial_lines(lines_no_begin, lines_no_end)

        # filter out all artificial lines with a angle bigger than angle_threshold.
        filtered_lines = calculate_cos(lines_no_begin, artificial_lines, lines_no_end, angle_threshold)

        # Add the artificial lines calculated in this loop to the total artificial lines.
        result_lines = result_lines.append(filtered_lines)

        # Remove the lines that have got an artificial line assigned.
        lines_no_begin.drop(filtered_lines.index.get_level_values(1), inplace=True)
        lines_no_end.drop(filtered_lines.index.get_level_values(0), inplace=True)

        # If we can't find more than 5 new artificial lines, we stop searching.
        if abs(previous_result_length - len(result_lines)) < 5:
            break

        previous_result_length = len(result_lines)

    # Add artificial line to real line data.
    result_lines = prepare_df_for_concatenation(result_lines, result_df)
    result_df = result_df.append(result_lines, ignore_index=True)

    print(len(result_lines))

    # write artificial lines to a new shapefile.
    write_shapefile(result_df, output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Parsing command line arguments
    parser.add_argument('--p', '--PATH',
                        type=str,
                        default='Meetvak/Meetvakken_WGS84.shp',
                        help="The path to the .shp file. Absolute or relative (to main.py) are accepted. \n"
                             "Default value = /Meetvak/Meetvakken_WGS84.shp",
                        metavar='p')

    parser.add_argument('--d', '--DEGREE',
                        type=int,
                        default=5,
                        help="The acceptable angle that the artificial line makes with the other line. \n"
                             "Default value = 5",
                        metavar='d')

    parser.add_argument('--o', '--OUTPUT',
                        type=str,
                        default='filtered_lines.shp',
                        help="The the path where the output file is generated. "
                             "Absolute or relative (to main.py) are accepted. \n"
                             "Default value = /filtered_lines.shp",
                        metavar='p')

    args = parser.parse_args()

    # Printing script options.
    print_options(args)

    start = timer()
    main(args.p, args.d, args.o)
    end = timer()
    print(f"Script finished.")
    print(f"Total time elapsed: {end - start} Seconds.")
