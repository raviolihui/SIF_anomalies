import json

from pystac import Catalog, get_stac_version
from pystac.extensions.eo import EOExtension
from pystac.extensions.label import LabelExtension


root_catalog = Catalog.from_file('https://data-portal.s5p-pal.com/api/s5p-l3')

collection = root_catalog.get_child("Solar Induced Fluorescence")
if collection is None:
    print("Collection is Empty. Check your downloads and try again.")
else:
    print("Collection has a root child. You may proceed to the following steps.")


items = list(collection.get_all_items())

# Now you can iterate over them
for item in items:
    print(item.id, item.datetime)

# #2018 - 2024
# #lat lon lat_min, lat_max = -15, 10
# #lon_min, lon_max = -75, -50
# #daily