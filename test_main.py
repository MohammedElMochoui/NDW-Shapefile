import pytest
import main
import numpy as np
from shapely.geometry import MultiPoint

# Variable creation needed for testing
df = main.read_shp('Meetvak/Meetvakken_WGS84.shp')
lines_without_end, lines_without_begin = main.get_lines(df)
indices = main.calculate_neighbors(lines_without_end, lines_without_begin)
lines_without_begin['first'] = main.calculate_neighbors(lines_without_end, lines_without_begin)[:, 0]
lines_without_begin_ids = main.convert_to_id(lines_without_begin['first'], lines_without_end).to_numpy(copy=True)
lines_without_begin['first'] = main.convert_to_id(lines_without_begin['first'], lines_without_end)
artificial_lines = main.generate_artificial_lines(lines_without_begin, lines_without_end)


# Function should return y coordinate
def test_get_end_returns_second_coordinate_when_given_two_coordinates():
    m = MultiPoint([(0, 0), (1, 1)])
    assert (main.get_end(m) == "POINT (1 1)"), \
        "The Coordinate returned by get_end() is not the the second coordinate."


# Function should return x coordinate
def test_get_begin_returns_first_coordinate_when_given_two_coordinates():
    m = MultiPoint([(0, 0), (1, 1)])
    assert (main.get_begin(m) == "POINT (0 0)"), \
        "The coordinate returned by get_begin() is not equal to the first coordinate."


# Function should not accept inputs with less than two coordinates
def test_get_end_input_has_less_than_two_coordinates():
    m = MultiPoint([(0, 0)])
    with pytest.raises(Exception):
        main.get_end(m)


# Function should set the id column as DataFrame
def test_read_shp_returns_dataframe_with_id_as_index():
    assert (df.index.name == 'id'), \
        "The dataframe has no index with name id."


# The amount of lines without end should be smaller than the total amount of lines in the DataFrame
def test_lines_no_end_has_less_elements_than_rows_in_df():
    assert (len(lines_without_end) < len(df)), \
        "Lines no end has more elements than the DataFrame!"


# The amount of lines without begin should be smaller than the total amount of lines in the DataFrame
def test_lines_no_begin_has_less_elements_than_rows_in_df():
    assert (len(lines_without_begin) < len(df)), \
        "Lines no begin has more elements than the DataFrame."


# The amount of neighbors should be equal to the amount of lines without begin.
def test_calculate_neighbors_neighbor_calculated_for_each_line_in_lines_no_begin():
    assert (len(indices) == len(lines_without_begin)), \
        "The amount of indices is not equal to the lines without begin."


# All the indices of neighbors calculated should have a corresponding line without end.
def test_calculate_neighbors_all_indices_in_range_of_len_lines_no_end():
    assert (np.all(np.apply_along_axis(lambda x: x < len(lines_without_end), 1, np.array(indices)))), \
        "Not all indices of neighbors can be found in lines without end."


# The amount of artificial lines should be equal to the amount of lines without begin.
def test_generate_artificial_lines_length_of_artificial_lines_is_equal_to_length_lines_no_begin():
    assert (len(artificial_lines) == len(lines_without_begin)), \
        "The amount of artificial lines is not equal to the amount of lines without begin."


# The begin points of the artificial lines should intersect with an end point of a line without end
def test_generate_artificial_lines_all_lines_in_artificial_lines_have_begin_point_in_lines_no_end():
    art_in_without_end = artificial_lines.index.get_level_values(0).to_numpy(copy=True)
    art_in_without_end = art_in_without_end.reshape(art_in_without_end.shape[0], 1)
    lines_without_end_array = lines_without_end.index.to_numpy(copy=True)
    art_in_without_end = np.apply_along_axis(lambda x: np.any(lines_without_end_array == x), 1, art_in_without_end)
    assert (np.all(art_in_without_end)), \
        "Not all artificial lines have their begin point in lines without end."


# The end points of the artificial lines should intersect with a begin point of a line without begin
def test_generate_artificial_lines_all_lines_in_artificial_lines_have_begin_point_in_lines_no_begin():
    art_in_without_begin = artificial_lines.index.get_level_values(1).to_numpy(copy=True)
    art_in_without_begin = art_in_without_begin.reshape(art_in_without_begin.shape[0], 1)
    lines_without_begin_array = lines_without_begin.index.to_numpy(copy=True)
    art_in_without_begin = np.apply_along_axis(lambda x: np.any(lines_without_begin_array == x), 1, art_in_without_begin)
    assert (np.all(art_in_without_begin)), \
        "Not all artificial lines have their end point in lines without begin."


# All id's calculated by the function should be in the set of id's in lines without begin
def test_convert_to_id():
    lines_without_end_array = lines_without_end.index.to_numpy(copy=True)
    lines_without_end_array = lines_without_end_array.reshape(lines_without_end_array.shape[0], 1)
    global lines_without_begin_ids
    lines_without_begin_ids = lines_without_begin_ids.reshape(lines_without_begin_ids.shape[0], 1)
    id_in_lines_without_end = np.apply_along_axis(lambda x: np.any(lines_without_end_array == x), 1, lines_without_begin_ids)
    assert (np.all(id_in_lines_without_end)), \
        "Not all id's can be found int lines without ends."
