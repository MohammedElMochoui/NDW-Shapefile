# NDW-Shapefile

## Install dependencies
Run pip install -r requirements.txt (Python 2), or pip3 install -r requirements.txt (Python 3)

## Run script
usage: main.py [-h] [--p p] [--d d] [--o p]

optional arguments:
  -h, --help         show this help message and exit
  --p p, --PATH p    The path to the .shp file. Absolute or relative (to main.py) are accepted. Default value = /Meetvak/Meetvakken_WGS84.shp
  --d d, --DEGREE d  The acceptable angle that the artificial line makes with the other line. Default value = 5
  --o p, --OUTPUT p  The the path where the output file is generated. Absolute or relative (to main.py) are accepted. Default value = /filtered_lines.shp
