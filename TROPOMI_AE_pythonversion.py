# %% [markdown]
# This notebook is a complementary code of the study of anomalies of SIF using 3D Convolutional Autoencoders and z-scores
# 
# In this notebook one can find a step by step process on finding anomalies of SIF over the Amazonia from the TROPOSIF dataset both with CAE and with the z-score method. At the end we also perform a comparison with the droughts recorded by the Monitor de Secas (MS) and SPEI. 
# 
# Make sure to follow one cell at a time to understand the process, you will find the autoencoder architecture and the training in some cells bellow. You can avoid having to run this long process by skipping some cells (will be explained in detail further on the notebook) and use a file with the already train autoencoder. This will not only save time, but also ensure that the results you obtain are exactly the results that I used to write my master thesis. 
# 
# If any questions were to arrise, do not hesitate to contact me: carmen-oliver.huidobro@uv.es

# %%
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
from skimage.transform import resize
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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch
import time
from codecarbon import EmissionsTracker

# %% [markdown]
# We are using a rectangle to bound the Amazonia area with latmin, latmax, lonmin and lonmax. These can be modified if one wishes to find SIF anomalies in other areas of the world.

# %%
lat_min, lat_max = -15, 5
lon_min, lon_max = -75, -50

data_dir = "/Users/Carmen/Desktop/SIF_anomalies/SIF_DATA_TROPOMI/" #change according to your path

sif_amazon = []
time_list = []

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

# %%
has_nans = np.isnan(sif_monthly).any()
print("Does sif_monthly contain NaNs?", has_nans)
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

climatology_table = pd.DataFrame({
    "Month": months,
    "Avg SIF": monthly_means.values.round(3)
})

print("Monthly Climatology Summary:")
print(climatology_table)
print("\nOverall Mean SIF:", round(overall_mean_sif.item(), 3))

# %%
plt.figure(figsize=(6,3))
plt.plot(months, monthly_means, color='green', label ='Climatology (2019-2023)')
sif_monthly.sel(year=2024).mean(dim=["latitude", "longitude"]).plot(label='2024')
plt.ylabel("SIF - mW m⁻² nm⁻¹ sr⁻¹")
plt.xticks(rotation=45)
plt.legend()
plt.savefig("pic/sif_climatology.png")
plt.show()

# %%
# Choose a month (0-11)
sif_july = climatology.sel(month=7)  

fig = plt.figure(figsize=(5, 3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.RIVERS, alpha=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.3)

im = sif_july.plot.pcolormesh(
    ax=ax,
    cmap="YlGn",
    add_colorbar=False,  
    add_labels=False,
    transform=ccrs.PlateCarree()
)

cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.8)
cbar.set_label("SIF - mW m⁻² nm⁻¹ sr⁻¹")

#plt.title("Climatology Map of Amazonas - July")
plt.show()


sif_july = climatology.isel(month=6)   # 6 = July (if 0=Jan, 1=Feb,...)

fig = plt.figure(figsize=(5, 3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-90, -30, -35, 20], crs=ccrs.PlateCarree()) 

ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.RIVERS, alpha=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.3)

im = sif_july.plot.pcolormesh(
    ax=ax,
    cmap="YlGn",
    transform=ccrs.PlateCarree(),
    add_colorbar=False   
)

cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.8)
cbar.set_label("SIF - mW m⁻² nm⁻¹ sr⁻¹")

ax.set_xlabel("Longitude (°W to °E)")
ax.set_ylabel("Latitude (°S to °N)")
#plt.title("Climatology Map of Amazonas - July")
plt.tight_layout()
plt.show()

# %% [markdown]
# The following process creates more datapoints with less resolutions from the previously gathered data

# %%
def create_strided_downsamples(data, modulo):
    h, w = data.shape[-2], data.shape[-1]

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

print("sif_patches shape:", sif_patches.shape)
print( np.isnan(sif_patches).any())

fig = plt.figure(figsize=(5, 3))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.RIVERS, alpha=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.3)

xticks = np.linspace(lon_min, lon_max, 6)
yticks = np.linspace(lat_min, lat_max, 5)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

im = ax.pcolormesh(
    np.linspace(lon_min, lon_max, sif_patches.shape[-1]),
    np.linspace(lat_min, lat_max, sif_patches.shape[-2]),
    sif_08_2024_downsampled,
    cmap="YlGn",
    shading="auto",
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.03 , shrink = 1.2)
cbar.set_label("SIF -  mW m⁻² nm⁻¹ sr⁻¹")
plt.xlabel("Longitude (°W to °E)")
plt.ylabel("Latitude (°S to °N)")
plt.show()

sif_patches_flat = sif_patches.to_numpy().flatten()

plt.figure(figsize=(10, 6))
plt.hist(sif_patches_flat, bins=100, color='blue', alpha=0.7)

percentile_1 = np.percentile(sif_patches_flat, 1)
percentile_99 = np.percentile(sif_patches_flat, 99)

plt.axvline(percentile_1, color='red', linestyle='--', label='1th Percentile')
plt.axvline(percentile_99, color='green', linestyle='--', label='99th Percentile')
plt.legend()
plt.xlabel("SIF Value - mW m⁻² nm⁻¹ sr⁻¹")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



# %% [markdown]
# The following three cells define the autoencoder, train it and save it. Do not run them if you want to obtain the exact same results as I showed in my thesis. 
# 
# If you want to reproduce the results that I obtained go 4 cells bellow and run the following ones as it will then use the saved autoencoder model already trained by me before.  

# %%
input_sequences = sif_patches.values[..., np.newaxis]  
input_seq = Input(shape=(12, sif_patches.shape[2], sif_patches.shape[3], 1))
print(input_sequences.shape)

# # Encoder
# x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_seq)
# x = MaxPooling3D((2, 2, 2), padding='same')(x)
# x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling3D((2, 2, 2), padding='same',
#                        activity_regularizer=regularizers.l1(1e-5))(x)

# # Decoder (mirror of encoder)
# x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling3D((2, 2, 2))(x)
# x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
# x = UpSampling3D((2, 2, 2))(x)
# x = Cropping3D(((0, 0), (0,  x.shape[2] - input_seq.shape[2]), (0, x.shape[3] - input_seq.shape[3])))(x)
# decoded = Conv3D(1, (3, 3, 3), activation='relu', padding='same')(x)

# # # Autoencoder Model
# autoencoder2 = Model(input_seq, decoded)
# autoencoder2.compile(optimizer='adam', loss='mse')
# autoencoder2.summary()




# %%
# #np.random.seed(42)  #to get reproducible results
# autoencoder2.fit(input_sequences, input_sequences, 
#                 epochs=50,
#                 batch_size=30,
#                 validation_split=0.2,
#                 shuffle=True) 



# %%
# autoencoder2.save("autoencoder_sif_model.keras")
# print("Model saved to autoencoder_sif_model.keras")


# model_path = "autoencoder_sif_model.keras"
# if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
#     print(f"Model file '{model_path}' exists and is not empty.")
# else:
#     print(f"Model file '{model_path}' is missing or empty.")

# if 'autoencoder2' in locals() and hasattr(autoencoder2, 'history') and autoencoder2.history is not None:
#     history = autoencoder2.history
# else:
#     print("Training history not found. Please ensure you have the training history object.")
#     history = None

# if history is not None:
#     if hasattr(history, 'history'):
#         hist_dict = history.history
#     else:
#         hist_dict = history

#     plt.figure(figsize=(8, 5))
#     plt.plot(hist_dict['loss'], linewidth=0.7,label='Training Loss')
#     if 'val_loss' in hist_dict:
#         plt.plot(hist_dict['val_loss'], linewidth=0.7, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss (MSE)')
#     plt.title('Autoencoder Training and Validation Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()




# %% [markdown]
# With the line " autoencoder2 = load_model("autoencoder_sif_model.keras") " we are reproducing the results that I described in my thesis project. Run the cells from here to reproduce the same results

# %%
autoencoder2 = load_model("autoencoder_sif_model.keras")


# %%
encoded_layer = None
for layer in autoencoder2.layers:
    if layer.name == 'max_pooling3d_77':  
        encoded_layer = layer
        break
if encoded_layer is None:
    raise ValueError("Encoded layer not found in autoencoder2.")



reconstruct = autoencoder2.predict(input_sequences)
print("Reconstructed shape:", reconstruct.shape) 

sample_idx = 100  
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
time_steps = [0, 3, 6, 9, 11]

vmin = min(sif_patches[sample_idx, t, :, :].min() for t in time_steps)
vmax = max(sif_patches[sample_idx, t, :, :].max() for t in time_steps)

for i, t in enumerate(time_steps):
    im0 = axes[0, i].imshow(reconstruct[sample_idx, t, :, :], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[0, i].set_title(f'Reconstructed Month {t+1}')
    axes[0, i].axis('off')

    im1 = axes[1, i].imshow(sif_patches.values[sample_idx, t, :, :], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    axes[1, i].set_title(f'Input Month {t+1}')
    axes[1, i].axis('off')

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(im0, cax=cbar_ax, label='SIF Value')

plt.show()

# %%
input_means = np.mean(sif_patches.values, axis=(2, 3))  
recon_means = np.mean(reconstruct[..., 0], axis=(2, 3))  
sample = 100
plt.figure(figsize=(9, 4))
plt.plot(range(12), input_means[sample], marker='o', label='Input Sample 100')
plt.plot(range(12), recon_means[sample], marker='o', label='Reconstructed (CAE)')
plt.xticks(range(12), months, rotation=45)
plt.xlabel('Month')
plt.ylabel('Mean SIF Value - mW m⁻² nm⁻¹ sr⁻¹')
plt.legend()
plt.show()

# %%
sif_monthly_resized = sif_monthly.coarsen(latitude=10, longitude=10, boundary='trim').mean()
plt.figure(figsize=(5, 3))
plt.imshow(sif_monthly_resized[0, 0], cmap='YlGn', origin = 'lower',  extent=[lon_min, lon_max, lat_min, lat_max])
plt.show()

sif_mon = sif_monthly_resized.values
original = sif_mon[..., np.newaxis]  

# Start tracking time and energy
tracker = EmissionsTracker()
tracker.start()
start_time = time.time()

reconstructed = autoencoder2.predict(original)

# Stop tracking time and energy
end_time = time.time()
emissions = tracker.stop()

print(f"Time taken: {end_time - start_time:.10f} seconds")
print(f"Energy consumed: {emissions:.10f} kWh")

print("Reconstructed shape:", reconstructed.shape)  # Should be (6, 12, 91, 113, 1)

reconstructed = reconstructed[..., 0]
original = original[..., 0]
print("Original shape:", original.shape, "Reconstructed shape:", reconstructed.shape)

reconstructed = xr.DataArray(
    reconstructed,
    dims=["year", "month", "latitude", "longitude"],
    coords={
        "year": sif_monthly_resized.year.values,
        "month": sif_monthly_resized.month.values,
        "latitude": sif_monthly_resized.latitude.values,
        "longitude": sif_monthly_resized.longitude.values,
    }
)

original = xr.DataArray(
    original,
    dims=["year", "month", "latitude", "longitude"],
    coords={
        "year": sif_monthly_resized.year.values,
        "month": sif_monthly_resized.month.values,
        "latitude": sif_monthly_resized.latitude.values,
        "longitude": sif_monthly_resized.longitude.values,
    }
)

reconstruction_error_pixel = (original.sel(year=slice(2019, 2023)) -
                              reconstructed.sel(year=slice(2019, 2023))) ** 2  #MSE, leave 2024 out to compute threshold
print("Reconstruction error shape:", reconstruction_error_pixel.shape) 

#this threshold is for general areas, not pixel per pixel 
thresholds_per_month = reconstruction_error_pixel.quantile(0.99, dim=["year", "latitude", "longitude"])

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

mean_errors = reconstruction_error_pixel.mean(dim=["year", "latitude", "longitude"])

plt.figure(figsize=(6, 3))
plt.plot(mean_errors["month"], mean_errors, 'b-', label='Mean Error', marker='o')
plt.plot(thresholds_per_month["month"], thresholds_per_month, 'r--', label='99th Percentile Threshold')
plt.xticks(range(1, 13), months)
plt.xlabel('Month')
plt.ylabel('Reconstruction Error')
plt.title('Monthly Reconstruction Error and 99th Percentile Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


anomalies_month = (reconstruction_error_pixel > thresholds_per_month).astype(int)
print(anomalies_month.shape)  

extended_months = [f"{months[m-1]}-{y}" 
                   for y in reconstructed.year.values 
                   for m in reconstructed.month.values]


climatology = sif_monthly_resized.sel(year=years[:5])  # 2019–2023
climatology_mean = climatology.mean(dim="year")

climatology_24_months = xr.concat([climatology_mean.mean(dim=["latitude","longitude"])] * 6, dim="year").values.flatten()  
climatology_AE_mean = reconstructed.mean(dim=["year", "latitude", "longitude"])
climatology_AE = xr.concat([climatology_AE_mean] * 6, dim="year")
climatology_AE = climatology_AE.assign_coords(year=np.arange(2019, 2019+6)).values.flatten()
AE_reconstruction = reconstructed.mean(dim=["latitude", "longitude"]).values.flatten()
sif_measured_24_months = sif_monthly_resized.mean(dim=["latitude","longitude"]).values.flatten()


plt.figure(figsize=(14, 6))
plt.plot(extended_months, climatology_AE, label="Climatology (Expected by AE)", color="yellow", marker="o")
plt.plot(extended_months, climatology_24_months, label="Climatology (Mean of all data)", color="blue", marker="o")
plt.plot(extended_months, AE_reconstruction, label="Reconstructed SIF (Autoencoder)", color="red", marker="o")
plt.plot(extended_months, sif_measured_24_months, label="Measured SIF (Satellite)", color="green", marker="o")
plt.xlabel("Month (2023-2024)")
plt.ylabel("SIF Value")
plt.title("Climatology vs Measured SIF (2023-2024)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(months, climatology_AE_mean, label='Climatology CAE Mean (2019-2023)', color='blue')
plt.plot(months, np.nanmean(sif_monthly_resized[5], axis=(1, 2)), label='2024', color='red', marker='o')
plt.xlabel('Month')
plt.ylabel('SIF Value')
#plt.title('Monthly SIF Climatology (2019-2023) with 2024')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# %%
# Compute threshold for each pixel and each month without 2024
thresholds_per_pixel = reconstruction_error_pixel.sel(year=slice(2019, 2023)).quantile(0.99, dim="year")

# Including 2024
reconstruction_error_pixel_all = (original - reconstructed) ** 2
print("Reconstruction error shape:", reconstruction_error_pixel_all.shape)

anomalies_pixel = reconstruction_error_pixel_all > thresholds_per_pixel

anomalies_da = xr.DataArray(
    anomalies_pixel,
    dims=["year", "month", "latitude", "longitude"],
    coords={
        "year": original.year,
        "month": original.month,
        "latitude": original.latitude,
        "longitude": original.longitude,
    }
)


fig, axs = plt.subplots(3, 4,figsize=(9,6), subplot_kw={'projection': ccrs.PlateCarree()})
#fig.suptitle("CAE Anomaly Maps for All Months of 2024", fontsize=18)

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    anomalies_da.sel(year=2024, month=month_idx + 1).plot.imshow(
        ax=ax, transform=ccrs.PlateCarree(), cmap="Reds", add_colorbar=False
    )
    ax.set_title(f"{months[month_idx]} 2024")
    ax.axis('off')

legend_elements = [
    Patch(facecolor='darkred', edgecolor='darkred', label='Anomaly'),
    Patch(facecolor='white', edgecolor='black', label='Not Anomaly')
]
#ax.legend(handles=legend_elements, loc='lower right', frameon=True)
plt.show()

anomaly_list_ds = anomalies_da.where(anomalies_da, drop=True).to_dataframe(name="anomaly").reset_index()
anomaly_list_ds["error"] = reconstruction_error_pixel_all.where(anomalies_da).to_dataframe(name="error").reset_index()["error"]


anomaly_percentages_2024 = []
for month in range(1, 13):
    mask = anomalies_da.sel(year=2024, month=month)
    total = mask.notnull().sum().item()
    anomalies = mask.sum().item()
    percent = (anomalies / total) * 100 if total > 0 else np.nan
    anomaly_percentages_2024.append(percent)

anomaly_percentages_2024_cnn_df = pd.DataFrame({
    'Month': months,
    'Anomaly % (2024)': np.round(anomaly_percentages_2024, 2)
})

print("Anomaly Percentages Table for 2024:")
print(anomaly_percentages_2024_cnn_df)

plt.figure(figsize=(6,3))
plt.hist(reconstruction_error_pixel_all.sel(year=2024, month=t).values.flatten(), bins=100)
plt.axvline(thresholds_per_month.sel(month=t).item(), color='r', linestyle='--', label='99th percentile')
#plt.title(f"Reconstruction Errors for Month {t}")
plt.legend()
plt.show()


# %%
from matplotlib.colors import ListedColormap, BoundaryNorm

difference = original - reconstructed

masked_difference = difference.where(anomalies_pixel, other=0)

fig, axs = plt.subplots(3, 4, figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})
#fig.suptitle("Difference (Original - Reconstructed) for Each Month of 2024", fontsize=18)

vmax = np.nanmax(abs(masked_difference.values))
vmin = -vmax

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    im = masked_difference.isel(year=5, month=month_idx).plot.imshow(
        ax=ax,
        cmap="RdBu",
        add_colorbar=False,
        transform=ccrs.PlateCarree(),
        vmin=-1.5,
        vmax=1.5
    )
    
    ax.axis("off")
    ax.set_title(f"{months[month_idx]} 2024", fontsize=10)

#fig.subplots_adjust(right=0.88, wspace=0.2, hspace=0.25)
#cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
#cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
#cbar.set_label("Difference (Original - Reconstructed)", fontsize=12)
add_colorbar = True
if add_colorbar:
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="vertical", pad=0.02, fraction=0.02)
    #cbar.set_label("(SIF - Reconstructed) - mW m⁻² nm⁻¹ sr⁻¹", fontsize=12)
plt.savefig("pic/CAE_amazon_2024.png")
plt.show()


# create sign map: +1 for positive, -1 for negative, 0 for zero/no-anomaly
sign_map = xr.where(masked_difference > 0, 1,
                    xr.where(masked_difference < 0, -1, 0))


fig, axs = plt.subplots(3, 4, figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    im = sign_map.isel(year=5, month=month_idx).plot.imshow(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu",
        add_colorbar=False
    )

    ax.set_title(f"{months[month_idx]} 2024", fontsize=10)
    ax.axis("off")


plt.savefig("pic/CAE_amazon_2024_masked.png")
plt.show()

print(vmin, vmax)

# %%
quantiles = np.linspace(0.0, 1.0, 101)


thresholds_all = reconstruction_error_pixel.sel(year=slice(2019, 2023)).quantile(quantiles, dim="year")

pct_anomalies = []

for q in quantiles:

    thresh = thresholds_all.sel(quantile=q)
    
    is_anomaly = reconstruction_error_pixel_all > thresh

    pct = (int(is_anomaly.sum().item())/ 740376 )* 100
    pct_anomalies.append(pct)


plt.figure(figsize=(6, 4))
plt.bar(quantiles * 100, pct_anomalies, color='tab:blue', alpha=0.7, width=0.8)
plt.xlabel('Percentile Threshold (%)')
plt.ylabel('Percentage of Anomalous Pixels (%)')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('anomaly_percentile_plot.png')
plt.show()  






# %% [markdown]
# Following two cells compute anomalies with the z-score method 

# %%
climatology_mean = sif_patches.mean(dim="sample")
climatology_std = sif_patches.std(dim="sample")
print("Climatology mean shape:", climatology_mean.shape, "Climatology std shape:", climatology_std.shape, sif_monthly_resized.sel(year=years[5:]).squeeze("year").shape)

z_scores = (sif_monthly_resized.sel(year=years[5:]).squeeze("year") - climatology_mean) / climatology_std

anomaly_z = z_scores.where(np.abs(z_scores) > 2, other=0).astype(bool)

months_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec']

fig, axs = plt.subplots(3, 4,figsize=(9,6), subplot_kw={'projection': ccrs.PlateCarree()})
#fig.suptitle("Z-score Anomaly Maps for All Months of 2024 (|z| > 2)", fontsize=18)

anomaly_2024 = anomaly_z
for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    
    anomaly_2024.isel(month=month_idx).plot.imshow(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
        add_colorbar=False
    )
    
    ax.set_title(f"{months_labels[month_idx]} 2024")
    ax.axis("off")


#legend_elements = [
#    Patch(facecolor="darkred", edgecolor="darkred", label="Z-score Anomaly"),
#    Patch(facecolor="white", edgecolor="black", label="Not Anomaly"),
#]
#axs[2, 3].legend(handles=legend_elements, loc="lower right", frameon=True)

plt.tight_layout()
plt.show()

# %%
from matplotlib.cm import ScalarMappable



z_mapped = xr.where(z_scores > 2, 1,
             xr.where(z_scores < -2, -1, 0))

fig, axs = plt.subplots(3, 4,figsize=(9,6),
                        subplot_kw={'projection': ccrs.PlateCarree()})
#fig.suptitle("Z-score Anomaly Maps for All Months of 2024 (|z| > 2)", fontsize=18)

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    
    z_mapped.isel(month=month_idx).plot.imshow(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu",  # red for positive, blue for negative
        add_colorbar=False,
        vmin=-1, vmax=1
    )
    
    ax.set_title(f"{months_labels[month_idx]} 2024")
    ax.axis("off")

legend_elements = [
    Patch(facecolor="darkred", edgecolor="darkred", label="Z > 2 (Positive Anomaly)"),
    Patch(facecolor="darkblue", edgecolor="darkblue", label="Z < -2 (Negative Anomaly)"),
    Patch(facecolor="white", edgecolor="black", label="Normal (|Z| ≤ 2)"),
]
#axs[2, 3].legend(handles=legend_elements, loc="upper right", frameon=True)


plt.savefig("pic/zscore_amazon_2024_masked.png")
plt.show()



z_context = (sif_monthly_resized.sel(year=years[5:]).squeeze("year") - climatology_mean)
z_masked = z_context.where((z_scores > 2) | (z_scores < -2), other=0)

vmax = np.nanmax(abs(z_masked.values))
vmin = -vmax

# optional quick check
print("z_masked dtype:", z_context.dtype, "unique values sample:", np.unique(z_context.values.ravel())[:5])

fig, axs = plt.subplots(3, 4,figsize=(9,6),
                        subplot_kw={'projection': ccrs.PlateCarree()})
#fig.suptitle("Z-score Anomaly Maps for All Months of 2024 (|z| > 2)", fontsize=18)

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    
    z_masked.isel(month=month_idx).plot.imshow(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu",  # red for positive, blue for negative
        add_colorbar=False,
        vmin=-1.5, vmax=1.5
    )
    
    ax.set_title(f"{months_labels[month_idx]} 2024")
    ax.axis("off")

legend_elements = [
    Patch(facecolor="darkred", edgecolor="darkred", label="Z > 2 (Positive Anomaly)"),
    Patch(facecolor="darkblue", edgecolor="darkblue", label="Z < -2 (Negative Anomaly)"),
    Patch(facecolor="white", edgecolor="black", label="Normal (|Z| ≤ 2)"),
]
#axs[2, 3].legend(handles=legend_elements, loc="upper right", frameon=True)

add_colorbar = True
if add_colorbar:
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    sm = ScalarMappable(cmap="RdBu")
    sm.set_clim(-1.5, 1.5)
    sm.set_array(np.linspace(-1.5, 1.5, 256))
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", pad=0.02, fraction=0.02)
    #cb.set_label("SIF Original - Climatology)", fontsize=12)
plt.savefig("pic/zscore_amazon_2024.png")
plt.show()

print(vmin, vmax)

# %%
print("Anomaly mask shape:", anomaly_z.shape)
anomaly_percentages_z = []


year_percentages = []
for month_idx in range(12):
    mask = anomaly_z[month_idx]
    total = np.isfinite(mask).sum() 
    anomalies = np.nansum(mask)
    percent = (anomalies / total) * 100 if total > 0 else np.nan
    year_percentages.append(percent)
anomaly_percentages_z.append(year_percentages)


months_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
anomaly_z_df = pd.DataFrame(np.array(anomaly_percentages_z[0]).T, columns=[2024], index=months_short)

print("Percentage of anomalies in 2024 (z-score method):")
print(anomaly_z_df.round(3))



# %%
zvalues_2024 = z_scores.values.flatten()
zvalues_2024 = zvalues_2024[np.isfinite(zvalues_2024)]

# Define thresholds from 0.0 to 1.0
thresholds = np.linspace(0.0, 5.0, 101)


zvalues_2024_abs = np.abs(zvalues_2024)

percentages_abs = [(np.sum(zvalues_2024_abs > t) / len(zvalues_2024_abs)) * 100 for t in thresholds]

plt.figure(figsize=(6, 4))
plt.bar(thresholds, percentages_abs, color='tab:blue', alpha=0.7, width=0.04)
plt.xlabel("Z Score Threshold", fontsize=12)
plt.ylabel("Percentage of Anomalous Pixels (%)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("pic/sif_zscore_anomaly_percentage.png")
plt.show()


# %%
####FIX 
reconstruction_error_2024 = reconstruction_error_pixel_all.sel(year=2024).clip(max=1)

fig, axs = plt.subplots(3, 4, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle("Reconstruction Error Maps for All Months of 2024", fontsize=18)

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    im = reconstruction_error_2024.isel(month=month_idx).plot.imshow(
        ax=ax,
        cmap="Blues",
        origin="lower",
        add_colorbar=False,
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(f"{months[month_idx]} 2024")
    ax.axis("off")
    
fig.subplots_adjust(right=0.88, wspace=0.2, hspace=0.25)  # leave space for colorbar
cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
cbar.set_label("Reconstruction Error", fontsize=12)

plt.show()



fig, axs = plt.subplots(3, 4,figsize=(9,6), subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle("Clipped Z-scores for All Months of 2024", fontsize=18)

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    # Plot clipped z-scores for the month
    im = z_scores.isel(month=month_idx).plot.imshow(
        ax=ax,
        cmap="Blues",
        origin="lower",
        add_colorbar=False,
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(f"{months[month_idx]} 2024")

fig.subplots_adjust(right=0.88, wspace=0.2, hspace=0.25)  # leave space for colorbar
cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
cbar.set_label("Clipped Z-score [-6, 6]", fontsize=12)

plt.show()

# %% [markdown]
# Here starts the comparison of anomaly results between the two methods, as well as comparing the results with the droughts recorded by Monitor de Secas (MS) (see master thesis for more information). You will observe in the following cells both a visual and numerical comparison. 

# %%
mean_anomaly_zscore_2024 = np.array(anomaly_percentages_z[0]) 
mean_anomaly_cae_2024 = np.array(anomaly_percentages_2024)  

plt.figure(figsize=(6, 3))
bar_width = 0.4
x = np.arange(len(months))

plt.bar(x - bar_width/2, mean_anomaly_cae_2024, width=bar_width, label='CAE Anomalies (%)', color='red', alpha=0.7)
plt.bar(x + bar_width/2, mean_anomaly_zscore_2024, width=bar_width, label='Z-score Anomalies (%)', color='blue', alpha=0.7)

plt.xticks(x, months, rotation=45)
plt.ylabel('Anomaly Percentage (%)')
plt.title('Monthly Anomaly Percentage Comparison (2024)')
plt.legend()
plt.tight_layout()
plt.savefig("pic/sif_anomaly_percentage_comparison_2024.png")
plt.show()



# %%
cnn_anomalies_2024 = anomalies_pixel[5] 
zscore_anomalies_2024 = anomaly_z

month_idx = 9 #october

target_lat = sif_monthly_resized.latitude
target_lon = sif_monthly_resized.longitude

#secas image 
secas_oct_img = mpimg.imread('pic/secas_oct.png')

fig = plt.figure(figsize=(12, 3))

proj = ccrs.PlateCarree()
extent = [lon_min, lon_max, lat_min, lat_max]

ax0 = plt.subplot(1, 3, 1, projection=proj)
ax0.set_extent(extent, crs=proj)
ax0.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
ax0.add_feature(cfeature.BORDERS, linewidth=0.7)
ax0.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax0.add_feature(cfeature.RIVERS, alpha=0.5)
im0 = ax0.imshow(cnn_anomalies_2024[month_idx], cmap='Reds', origin='lower', extent=extent, transform=proj)
ax0.set_title("CNN (Autoencoder) Anomalies")
ax0.axis('off')

ax1 = plt.subplot(1, 3, 2, projection=proj)
ax1.set_extent(extent, crs=proj)
ax1.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
ax1.add_feature(cfeature.BORDERS, linewidth=0.7)
ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax1.add_feature(cfeature.RIVERS, alpha=0.5)
im1 = ax1.imshow(zscore_anomalies_2024[month_idx], cmap='Reds', origin='lower', extent=extent, transform=proj)
ax1.set_title("Z-score Anomalies")
ax1.axis('off')

ax2 = plt.subplot(1, 3, 3, projection=proj)
ax2.set_extent(extent, crs=proj)
im2 = ax2.imshow(secas_oct_img, extent=extent, transform=proj)
ax2.set_title("SECA's Drought")
ax2.axis('off')

plt.show()


# %%
secas_sept_path = os.path.join('pic', 'secas_sept.png')
secas_images = sorted(glob.glob('secas_*.png') + glob.glob(os.path.join('pic', 'secas_*.png')))

# idea is to create a drought mask for each image based on the white pixels
drought_masks = {}
for img_path in secas_images:
    imga = mpimg.imread(img_path)
    img = cv2.cvtColor(imga, cv2.COLOR_BGR2RGB)
    target_shape = [113, 91]
    img_rgb_resized = np.flipud(cv2.resize(img, target_shape, interpolation=cv2.INTER_NEAREST))
    white_threshold = 0.3
    white_pixels = np.all(img_rgb_resized > white_threshold, axis=-1)
    drought_mask = (~white_pixels).astype(int)
    drought_masks[img_path] = drought_mask
    print(f"{img_path}: unique values {np.unique(drought_mask)}, shape {drought_mask.shape}")
imga = mpimg.imread(secas_sept_path)
drought_mask = drought_masks[secas_sept_path]


# %%
selected_months = [5, 7, 8, 9, 10, 11]
fig, axs = plt.subplots(2, 6, figsize=(22, 7))

for i, m_idx in enumerate(selected_months):
    masked_cnn = np.where(drought_mask == 1, cnn_anomalies_2024[m_idx], 0)
    axs[0, i].imshow(masked_cnn, cmap='Reds', origin='lower', extent=[lon_min, lon_max, lat_min, lat_max])
    axs[0, i].set_title(f'CNN Anomaly - {months[m_idx]}')
    axs[0, i].axis('off')
    
    masked_zscore = np.where(drought_mask == 1, zscore_anomalies_2024[m_idx], 0)
    axs[1, i].imshow(masked_zscore, cmap='Reds', origin='lower', extent=[lon_min, lon_max, lat_min, lat_max])
    axs[1, i].set_title(f'Z-score Anomaly - {months[m_idx]}')
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()


# %%
selected_months = [0, 4, 5, 7, 8, 9, 10, 11]
dice_scores = []
iou_scores = []

for m_idx in selected_months:
    masked_anomaly_cnn = np.where(drought_mask == 1, cnn_anomalies_2024[m_idx], 0)
    intersection = np.sum(masked_anomaly_cnn * drought_mask)
    total_pixels = np.sum(masked_anomaly_cnn) + np.sum(drought_mask)
    dice = 1.0 if total_pixels == 0 else 2 * intersection / total_pixels
    dice_scores.append(dice)

    intersection_iou = np.sum(cnn_anomalies_2024[m_idx] * drought_mask)
    union = np.sum((cnn_anomalies_2024[m_idx] + drought_mask) > 0)
    iou = 1.0 if union == 0 else intersection_iou / union
    iou_scores.append(iou)

for i, m_idx in enumerate(selected_months):
    print(f"{months[m_idx]}: Dice = {dice_scores[i]:.4f}, IoU = {iou_scores[i]:.4f}")



# %%
selected_months = [0, 4,5, 7, 8, 9, 10, 11]
dice_scores_z = []
iou_scores_z = []

for m_idx in selected_months:
    masked_anomaly_z = np.where(drought_mask == 1, zscore_anomalies_2024[m_idx], 0)
    intersection = np.sum(masked_anomaly_z * drought_mask)
    total_pixels = np.sum(masked_anomaly_z) + np.sum(drought_mask)
    dice = 1.0 if total_pixels == 0 else 2 * intersection / total_pixels
    dice_scores_z.append(dice)

    intersection_iou = np.sum(zscore_anomalies_2024[m_idx] * drought_mask)
    union = np.sum((zscore_anomalies_2024[m_idx] + drought_mask) > 0)
    iou = 1.0 if union == 0 else intersection_iou / union
    iou_scores_z.append(iou)

for i, m_idx in enumerate(selected_months):
    print(f"{months[m_idx]}: Dice = {dice_scores_z[i]:.4f}, IoU = {iou_scores_z[i]:.4f}")



# %% [markdown]
# The following cell makes a plot with all the droughts recorded per month of 2024 out of the images provided my the Monitor de Secas (MS). 

# %%
month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec']
secas_pngs = [os.path.join('pic', f'secas_{m}.png') for m in month_names]

extent = [lon_min, lon_max, lat_min, lat_max]
fig, axs = plt.subplots(3, 4, figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    path = secas_pngs[month_idx]
    if os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img, origin='upper', extent=extent, transform=ccrs.PlateCarree())
    else:
        ax.text(0.5, 0.5, "Missing", transform=ax.transAxes, ha='center', va='center', fontsize=12)

    ax.set_title(f"{month_names[month_idx].capitalize()} 2024")
    ax.axis('off')

plt.tight_layout()
plt.savefig("pic/secas_amazon_2024.png")
plt.show()



# %% [markdown]
# Here starts the same study but comparing it with the drought dataset: SPEI
# 

# %%
ds = xr.open_dataset("spei01.nc")
print(ds)

# %%
lat_min, lat_max = -15, 5
lon_min, lon_max = -75, -50


ds_2024 = ds.sel(time=slice("2024-01-01", "2024-12-01"))

ds_amazon_2024 = ds_2024.sel(
    lat=slice(lat_min, lat_max),
    lon=slice(lon_min, lon_max),
)

print(ds_amazon_2024)

# %%
# keep only absolute SPEI values > 1 (store the absolute value, else 0)
spei_month_anomaly = (ds_amazon_2024['spei']).where(np.abs(ds_amazon_2024['spei']) > 1, other=0)

def _hide_axis_outline(ax):
    try:
        ax.outline_patch.set_visible(False)
    except Exception:
        pass
    try:
        ax.patch.set_alpha(0.0)
    except Exception:
        pass

vmax = np.nanmax(abs(ds_amazon_2024['spei'].values))
vmin = -vmax

fig, axs = plt.subplots(3, 4,figsize=(9,6), subplot_kw={'projection': ccrs.PlateCarree()})

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    im = ax.imshow(spei_month_anomaly.isel(time=month_idx), cmap='RdBu', vmin=vmin, vmax=vmax, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], transform=ccrs.PlateCarree())
    ax.set_title(f"{months[month_idx]} 2024")
    add_colorbar=False

add_colorbar=True
if add_colorbar:
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    #cbar.set_label("SPEI", fontsize=12)

plt.savefig("pic/spei_amazon_2024.png")
plt.show()


# Create a binary mask for values below -0.8
spei_month_masked = (ds_amazon_2024['spei']).where(np.abs(ds_amazon_2024['spei']) > 1, other=0).astype(bool)

fig, axs = plt.subplots(3, 4,figsize=(9,6), subplot_kw={'projection': ccrs.PlateCarree()})

for month_idx in range(12):
    ax = axs[month_idx // 4, month_idx % 4]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    im = ax.imshow(spei_month_masked.isel(time=month_idx), cmap='Reds', origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], transform=ccrs.PlateCarree())
    ax.set_title(f"{months[month_idx]} 2024")
    add_colorbar=False

plt.tight_layout()
plt.savefig("pic/spei_amazon_2024_masked.png")
plt.show()


anomaly_spei_percentage = []
for month_idx in range(12):
    mask = spei_month_masked.isel(time=month_idx)
    total = np.isfinite(mask).sum() 
    anomalies = np.nansum(mask)
    percent = (anomalies / total) * 100 if total > 0 else np.nan
    anomaly_spei_percentage.append(percent)


months_short = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
anomaly_spei_df = pd.DataFrame(np.array(anomaly_spei_percentage).T, columns=[2024], index=months_short)

print("Percentage of anomalies in 2024 (SPEI method):")
print(anomaly_spei_df.round(3))


# %%
candidates = {
    "zscore": ["pic/zscore_amazon.png", "pic/zscore_amazon_2024.png", "zscore_amazon.png", "zscore_amazon_2024.png"],
    "spei":   ["pic/spei_amazon_2024.png", "spei_amazon_2024.png", "pic/spei_amazon.png", "spei_amazon.png"],
    "secas":  ["pic/secas_amazon_2024.png", "pic/secas_amazon.png", "secas_amazon_2024.png", "secas_amazon.png"],
    "cae":    ["pic/CAE_amazon_2024.png", "pic/cae_amazon_2024.png", "CAE_amazon_2024.png", "cae_amazon_2024.png"],
}

found = {}
for key, paths in candidates.items():
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Could not find any file for '{key}'. Tried: {paths}")
    found[key] = path

imgs = {k: mpimg.imread(p) for k, p in found.items()}

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

order = ["cae", "zscore", "spei", "secas"]  # top-left, top-right, bottom-left, bottom-right


for ax, key in zip(axs, order):
    ax.imshow(imgs[key])
    ax.axis("off")

plt.tight_layout()
out_path = "pic/combined_anomalies_panel.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved combined image to: {out_path}")

# %%

candidates = {
    "cae":    ["pic/CAE_amazon_2024.png", "pic/cae_amazon_2024.png", "CAE_amazon_2024.png", "cae_amazon_2024.png"],
    "zscore": ["pic/zscore_amazon.png", "pic/zscore_amazon_2024.png", "zscore_amazon.png", "zscore_amazon_2024.png"]
}

found = {}
for key, paths in candidates.items():
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Could not find any file for '{key}'. Tried: {paths}")
    found[key] = path

imgs = {k: mpimg.imread(p) for k, p in found.items()}

fig, axs = plt.subplots(1, 2, figsize=(14, 10))
axs = axs.flatten()

order = ["cae", "zscore"]  # top-left, top-right, bottom-left, bottom-right


for ax, key in zip(axs, order):
    ax.imshow(imgs[key])
    ax.axis("off")


out_path = "pic/two_anomalies_panel.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved combined image to: {out_path}")

candidates = {
    "secas":  ["pic/secas_amazon_2024.png", "pic/secas_amazon.png", "secas_amazon_2024.png", "secas_amazon.png"],
    "spei":   ["pic/spei_amazon_2024.png", "spei_amazon_2024.png", "pic/spei_amazon.png", "spei_amazon.png"]

}

found = {}
for key, paths in candidates.items():
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Could not find any file for '{key}'. Tried: {paths}")
    found[key] = path

imgs = {k: mpimg.imread(p) for k, p in found.items()}

fig, axs = plt.subplots(1, 2, figsize=(14, 10))
axs = axs.flatten()

order = ["spei", "secas"]  # top-left, top-right, bottom-left, bottom-right


for ax, key in zip(axs, order):
    ax.imshow(imgs[key])
    ax.axis("off")

plt.tight_layout()
out_path = "pic/two_droughts_panel.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved combined image to: {out_path}")

# %%
cnn_anomalies_2024 = anomalies_pixel[5] 
zscore_anomalies_2024 = anomaly_z

month_idx = 9 #october

target_lat = sif_monthly_resized.latitude
target_lon = sif_monthly_resized.longitude

secas_oct_img = mpimg.imread('pic/secas_oct.png')

fig = plt.figure(figsize=(12, 3))

proj = ccrs.PlateCarree()
extent = [lon_min, lon_max, lat_min, lat_max]

ax0 = plt.subplot(1, 4, 1, projection=proj)
ax0.set_extent(extent, crs=proj)
ax0.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
ax0.add_feature(cfeature.BORDERS, linewidth=0.7)
ax0.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax0.add_feature(cfeature.RIVERS, alpha=0.5)
im0 = ax0.imshow(cnn_anomalies_2024[month_idx], cmap='Reds', origin='lower', extent=extent, transform=proj)
ax0.set_title("CNN (Autoencoder) Anomalies")
ax0.axis('off')

ax1 = plt.subplot(1, 4, 2, projection=proj)
ax1.set_extent(extent, crs=proj)
ax1.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
ax1.add_feature(cfeature.BORDERS, linewidth=0.7)
ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax1.add_feature(cfeature.RIVERS, alpha=0.5)
im1 = ax1.imshow(zscore_anomalies_2024[month_idx], cmap='Reds', origin='lower', extent=extent, transform=proj)
ax1.set_title("Z-score Anomalies")
ax1.axis('off')

ax2 = plt.subplot(1, 4, 3, projection=proj)
ax2.set_extent(extent, crs=proj)
im2 = ax2.imshow(secas_oct_img, extent=extent, transform=proj)
ax2.set_title("SECA's Drought")
ax2.axis('off')

ax3 = plt.subplot(1, 4, 4, projection=proj)
ax3.set_extent(extent, crs=proj)
ax3.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", alpha=0.3)
ax3.add_feature(cfeature.BORDERS, linewidth=0.7)
ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax3.add_feature(cfeature.RIVERS, alpha=0.5)
im3 = ax3.imshow(spei_month_masked.isel(time=month_idx), cmap='Reds', origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], transform=ccrs.PlateCarree())
ax3.set_title("SPEI Anomalies")
ax3.axis('off')

plt.show()

# %%
target_lat = np.linspace(lat_min, lat_max, 20)
target_lon = np.linspace(lon_min, lon_max, 25)

anomalies_cnn_2024_f = anomalies_da.sel(year=2024).astype(float)
print(anomalies_cnn_2024_f.shape)
cnn_anomalies_2024_resized = anomalies_cnn_2024_f.interp(latitude=target_lat, longitude=target_lon)
cnn_anomalies_2024_resized = cnn_anomalies_2024_resized = cnn_anomalies_2024_resized.round().astype(int).values

selected_months = [0, 4, 5, 7, 8, 9, 10, 11]
dice_scores = []
iou_scores = []

for m_idx in selected_months:
    intersection = np.sum(cnn_anomalies_2024_resized[m_idx] * spei_month_masked.isel(time=m_idx))
    total_pixels = np.sum(cnn_anomalies_2024_resized[m_idx]) + np.sum(spei_month_masked.isel(time=m_idx))
    dice = 1.0 if total_pixels == 0 else 2 * intersection / total_pixels
    dice_scores.append(dice)

    intersection_iou = np.sum(cnn_anomalies_2024_resized[m_idx] * spei_month_masked.isel(time=m_idx))
    union = np.sum((cnn_anomalies_2024_resized[m_idx] + spei_month_masked.isel(time=m_idx)) > 0)
    iou = 1.0 if union == 0 else intersection_iou / union
    iou_scores.append(iou)

for i, m_idx in enumerate(selected_months):
    print(f"{months[m_idx]}: Dice = {dice_scores[i]:.4f}, IoU = {iou_scores[i]:.4f}")




# %%
anomaly_2024_f = anomaly_2024.astype(float)

target_lat = np.linspace(lat_min, lat_max, 20)
target_lon = np.linspace(lon_min, lon_max, 25)

zscore_anomalies_2024_resized2 = anomaly_2024_f.interp(
    latitude=target_lat,
    longitude=target_lon)

zscore_anomalies_2024_resized2 = (zscore_anomalies_2024_resized2 >= 0.5).astype(int).values

selected_months = [0, 4, 5, 7, 8, 9, 10, 11]
dice_scores = []

iou_scores = []

for m_idx in selected_months:
    intersection = np.sum(zscore_anomalies_2024_resized2[m_idx] * spei_month_masked.isel(time=m_idx))
    total_pixels = np.sum(zscore_anomalies_2024_resized2[m_idx]) + np.sum(spei_month_masked.isel(time=m_idx))
    dice = 1.0 if total_pixels == 0 else 2 * intersection / total_pixels
    dice_scores.append(dice)

    intersection_iou = np.sum(zscore_anomalies_2024_resized2[m_idx] *spei_month_masked.isel(time=m_idx))
    union = np.sum((zscore_anomalies_2024_resized2[m_idx] + spei_month_masked.isel(time=m_idx)) > 0)
    iou = 1.0 if union == 0 else intersection_iou / union
    iou_scores.append(iou)

for i, m_idx in enumerate(selected_months):
    print(f"{months[m_idx]}: Dice = {dice_scores[i]:.4f}, IoU = {iou_scores[i]:.4f}")


