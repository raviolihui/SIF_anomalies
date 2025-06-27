## SIF_anomalies

### Satellite-Based Anomaly Detection in Amazonian Solar-Induced Fluorescence (SIF)

#### Overview

**SIF_anomalies** is a research-oriented Python repository for detecting and analyzing spatiotemporal anomalies in satellite-derived Solar-Induced Fluorescence (SIF) over Amazonia. We will be using two distinct methods. The first one uses unsupervised machine learning, in particular, Convolutional Autoencoders (CAE). The second one uses the classical method which is z-scores. This project aims to compare both methods and understand how much do they explain with respect to extreme weather events, in particular we focus in droughts. 

#### Features

- **Convolutional Autoencoder (CAE) for Anomaly Detection:**  
  Identifies contiguous regions of anomalous SIF activity, outperforming traditional statistical methods in spatial coherence and interpretability.

- **Data Preprocessing and augmenting:**  
  Includes scripts for preparing SIF datasets and augmenting the data samples of SIF by outputing more images but with lower resolution from the initial sample.


- **Modular and Extensible:**  
  Designed for easy adaptation to other areas of the world with the potential of comparing SIF anomalies with other extreme weather events. 

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

- For a faster use, make sure to skip the cells that define the CAE and trains it as one can use the already trained model. Cells after these explicitly use the model saved as "autoencoder_sif_model", therefore there is no need to write any extra code. 

#### Data

- **Input:**  
  Satellite SIF dataset TROPOSIF in NetCDF format. To obtain this files go to "https://eo4society.esa.int/projects/sentinel-5p-innovation-solar-induced-chlorophyll-fluorescence-sif/" or contact the owner of this repository. 
  
- **Output:**  
  Anomaly maps highlighting regions and periods of significant SIF deviations.

#### Acknowledgements

This repository was developed as part of a research project on remote sensing-based vegetation monitoring in the Amazon. For questions, contact the repository maintainer via GitHub Issues[1].
