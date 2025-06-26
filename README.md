## SIF_anomalies

### Satellite-Based Anomaly Detection in Amazonian Solar-Induced Fluorescence (SIF)

#### Overview

**SIF_anomalies** is a research-oriented Python repository for detecting and analyzing spatiotemporal anomalies in satellite-derived Solar-Induced Fluorescence (SIF) over the Amazon rainforest. Leveraging advanced machine learning—specifically Convolutional Autoencoders (CAE)—this project aims to enhance the monitoring of vegetation health and stress responses to climate extremes, such as droughts and wildfires, using remote sensing data[1].

#### Features

- **Convolutional Autoencoder (CAE) for Anomaly Detection:**  
  Identifies contiguous regions of anomalous SIF activity, outperforming traditional statistical methods in spatial coherence and interpretability.

- **Data Preprocessing and Visualization:**  
  Includes scripts for preparing SIF datasets, normalizing inputs, and visualizing detected anomalies.

- **Modular and Extensible:**  
  Designed for easy adaptation to other geospatial anomaly detection tasks or integration of additional remote sensing indices.

#### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/raviolihui/SIF_anomalies.git
   cd SIF_anomalies
   ```

2. **Install dependencies:**  
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

#### Usage

- The main file TROPOMI_AE shows the main steps taken for the detection of anomalies both with CAE and z-score method using the dataset TROPOSIF provided by ESSD Copernicus

- TROPOMI_AE is a Jupyter Notebook in ypinb format, each cell is a step by step guide on how to obtain the results outlined by the master thesis

#### Data

- **Input:**  
  Satellite SIF datasets (e.g., TROPOMI, TROPOSIF) in NetCDF format.
  
- **Output:**  
  Anomaly maps highlighting regions and periods of significant SIF deviations.

#### Acknowledgements

This repository was developed as part of a research project on remote sensing-based vegetation monitoring in the Amazon. For questions, contact the repository maintainer via GitHub Issues[1].
