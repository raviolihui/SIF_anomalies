
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
from skimage.transform import rescale, resize
from keras import regularizers
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D
from keras.models import Model
import numpy as np
from keras.models import Model
from matplotlib.patches import Patch
import matplotlib.image as mpimg
import cv2
import numpy as np
import glob
import matplotlib.image as mpimg
from keras.models import Model
from keras.models import load_model

#Upload NCDF SIF file from TROPOSIF dataset


lat_min, lat_max = -15, 5
lon_min, lon_max = -75, -50

data_dir = "/Users/carmenoliver/Desktop/SIF_anomalies/SIF_DATA_TROPOMI/"

files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])
print("Files found:", files)


sif_amazon = []
time_list = []

data_dir = data_dir = "/Users/carmenoliver/Desktop/SIF_anomalies/SIF_DATA_TROPOMI/"

# Get all available files
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])
print("Files found:", files)
for file in files:
    match = re.search(r"month-(\d{6})", file)
    if match:
        date_str = match.group(1)  
        year = int(date_str[:4])  
        month = int(date_str[4:6])  

        file_path = os.path.join(data_dir, file)
        
        try:
            ds = xr.open_dataset(file_path)
        except Exception as e:
            print(f"Error opening {file_path}: {e}")
            continue
        ds_amazon = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
        
        sif_amazon.append(ds_amazon["solar_induced_fluorescence"].values)  
        time_list.append(f"{year}-{month:02d}")
        # Ensure data was loaded
if not sif_amazon or not time_list:
    raise ValueError("No valid data found. Check the input files and filtering logic.")
sif_amazon = np.stack(sif_amazon, axis=0).squeeze()  # Shape: [N_months, lat, lon]

years = sorted(set(int(t.split("-")[0]) for t in time_list))
num_years = len(years)

sif_monthly = np.full((num_years, 12, *sif_amazon.shape[1:]), np.nan)
for i, month_data in enumerate(sif_amazon):
    year_idx = i // 12  
    month_idx = i % 12  
    sif_monthly[year_idx, month_idx] = month_data 


print("Fixed sif_monthly shape:", sif_monthly.shape)




mean_sif_monthly = np.nanmean(sif_monthly)
sif_monthly = np.nan_to_num(sif_monthly, nan=mean_sif_monthly)


sif_monthly = xr.DataArray(
    sif_monthly,
    dims=["year", "month", "latitude", "longitude"],
    coords={
        "year" : np.arange(2019, 2019 + num_years),
        "month" : np.arange(1, 13),
        "latitude" : ds_amazon["latitude"].values,
        "longitude" : ds_amazon["longitude"].values
    }
)



sif_monthly_climatology = sif_monthly.sel(year=slice(2019, 2023))

climatology = sif_monthly_climatology.mean(dim="year", skipna=True)

monthly_means = climatology.mean(dim=["latitude", "longitude"], skipna=True)

overall_mean_sif = monthly_means.mean(skipna=True)

months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']




####Make more datapoints from original dataset by strided downsampling


def create_strided_downsamples(data, modulo):
    years, months, h, w = data.shape

    # Crop to make h and w divisible by modulo
    new_h = (h // modulo) * modulo
    new_w = (w // modulo) * modulo
    cropped = data[..., :new_h, :new_w]

    versions = []
    for row_mod in range(modulo):
        for col_mod in range(modulo):
            version = cropped[..., row_mod::modulo, col_mod::modulo]
            versions.append(version)

    return np.stack(versions, axis=1)
sif_strided = create_strided_downsamples(sif_monthly[0:5],10) 
print("sif_strided shape:", sif_strided.shape)  
sif_08_2024_downsampled = sif_strided[0,2, 7, :, :] 
print("sif_08_2024_downsampled shape:", sif_08_2024_downsampled.shape) 
sif_patches = sif_strided.reshape(-1, 12, sif_strided.shape[-2], sif_strided.shape[-1])
sif_monthly_resized = sif_monthly.coarsen(latitude=10, longitude=10, boundary='trim').mean()

# Convert back to xarray DataArray for consistency
sif_patches = xr.DataArray(
    sif_patches,
    dims=["sample", "month", "latitude", "longitude"],
    coords={
        "sample": np.arange(sif_patches.shape[0]),
        "month": np.arange(1, 13),
        "latitude": sif_monthly_resized.latitude.values,
        "longitude": sif_monthly_resized.longitude.values,
    }
)


#### Computation and ploting of the zscore anomalies

climatology_mean = sif_patches.mean(dim="sample")
climatology_std = sif_patches.std(dim="sample")
print("Climatology mean shape:", climatology_mean.shape, "Climatology std shape:", climatology_std.shape, sif_monthly_resized.sel(year=years[5:]).squeeze("year").shape)

z_scores = (sif_monthly_resized.sel(year=years[5:]).squeeze("year") - climatology_mean) / climatology_std
anomaly_z = z_scores.where(np.abs(z_scores) > 2, other=0).astype(bool)

months_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec']

fig, axs = plt.subplots(3, 4, figsize=(18, 12), subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle("Z-score Anomaly Maps for All Months of 2024 (|z| > 2)", fontsize=18)

anomaly_2024 = anomaly_z
for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Background
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    
    # Plot anomalies directly from xarray
    anomaly_2024.isel(month=month_idx).plot.imshow(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        add_colorbar=False
    )
    
    ax.set_title(f"{months_labels[month_idx]} 2024")
    ax.axis("off")

# Legend
legend_elements = [
    Patch(facecolor="darkred", edgecolor="darkred", label="Z-score Anomaly"),
    Patch(facecolor="white", edgecolor="black", label="Not Anomaly"),
]
axs[2, 3].legend(handles=legend_elements, loc="lower right", frameon=True)

plt.tight_layout()
plt.savefig("pic/anomaly_maps_all_months_zscore.png", dpi=300)

