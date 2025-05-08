from pystac_client import Client
from pystac_client import ItemSearch
import requests
import os

# Use the STAC API root URL instead
STAC_API_ROOT = "https://data-portal.s5p-pal.com/api/"
COLLECTION_ID = "s5p-l3-sif-month"  # Verify exact ID if this fails

def get_most_recent_sif_monthly():
    # 1. Connect to the STAC API
    catalog = Client.open(STAC_API_ROOT)
    
    # 2. Access the SIF monthly collection
    collection = catalog.get_collection(COLLECTION_ID)
    
    # 3. Get the most recent item
    items = ItemSearch(
        url=collection.get_self_href(),  # Use collection's own endpoint
        sortby=[{"field": "properties.archive_date", "direction": "desc"}],
        max_items=1
    ).items()
    
    item = list(items)[0]
    
    # 4. Download the product
    product = item.assets["product"]
    download_url = product.href
    filename = os.path.basename(download_url)
    
    print(f"Downloading {filename}...")
    r = requests.get(download_url)
    with open(filename, "wb") as f:
        f.write(r.content)
    
    print(f"Downloaded {filename} ({len(r.content)} bytes)")

if __name__ == "__main__":
    get_most_recent_sif_monthly()
