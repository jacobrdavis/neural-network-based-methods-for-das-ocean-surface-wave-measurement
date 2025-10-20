# Data for: Neural network-based methods for ocean surface wave measurement using submarine distributed acoustic sensing (DAS)

Dataset DOI: [10.5061/dryad.brv15dvnz](https://doi.org/10.5061/dryad.brv15dvnz)

## Description of the data and file structure

This dataset accompanies the article "Neural network-based methods for ocean surface wave measurement using submarine distributed acoustic sensing (DAS)".   DAS and seafloor mooring data were collected offshore of Oliktok Point, Alaska, over 10-day period between 22 August 2023 and 22 September 2023. DAS data   were recorded over a 27.1 km segment of submarine fiber-optic cable extending northward across the Beaufort shelf roughly along the 150W meridian in water depths  ranging from 3 m to 19 m.  Mooring data were collected at three sites spaced approximately every 10-km along the cable.

The netCDF files contain hourly and half-hourly DAS and mooring data used for model development. Hourly DAS and mooring data are processed into hourly strain rate spectra at channels within 1 km of each the three mooring sites (61 DAS channels per site).  Half-hourly DAS data are processed into 30-min strain rate frequency-wavenumber (f-k) spectra from all channels within 2 km of each mooring site (123 DAS channels per site).  Hourly and half-hourly along-cable DAS datasets, which contain frequency spectra and f-k spectra computed in 2-km segments along the cable with 50% overlap, are also provided.

### Files and variables

DAS and mooring data are provided in netCDF (.nc) format and contain self-describing global attributes, variable descriptions, and units. See the Methods section of the Dryad repository for additional details on data processing.

#### File: oliktok_das_mooring_hourly_dataset.nc

**Description:** Hourly DAS features (depth, cosine square wave direction, strain rate spectra) and target mooring seafloor pressure spectra required to train the spectral neural network model.  Strain rate spectra from all channels within 1-km of each mooring site are included (61 channels per site). 

#### File: oliktok_das_mooring_half-hourly_dataset.nc

**Description:** Half-hourly DAS strain rate f-k spectra and target mooring seafloor pressure spectra required to train the f-k convolutional neural network model. Strain rate f-k spectra were estimated using two-dimensional arrays of raw strain rate from 2 km of DAS channels collected over 30 min.

#### File: oliktok_das_hourly_along_cable_dataset.nc

**Description:** Hourly DAS features (depth, cosine square wave direction, strain rate spectra) in 2-km along-cable segments with 50% overlap (for a total of 25 segments, or artificial "sites"). Data were processed using the same methods as `oliktok_das_mooring_hourly_dataset.nc`, except DAS strain rate spectra, depth, and cosine squared wave direction were averaged over all 123 channels within each segment.

#### File: oliktok_das_half-hourly_along_cable_dataset.nc

**Description:** Half-hourly DAS strain rate f-k spectra in 2-km along-cable segments with 50% overlap (for a total of 25 segments, or artificial "sites"). Data were processed using the same methods as `oliktok_das_mooring_half-hourly_dataset.nc`.

#### File: model_states.zip

**Description:** Trained spectral neural network (specral NN) and f-k convolutional neural network (f-k CNN) model states and fitted min-max normalizations required to apply the models. States are provided in PyTorch state dict format (.pth). Example code for loading the states and reconstructing the model and normalizations are provided in a separate repository hosted on Zenodo.

The .zip folder includes the following files:

* `fk_convolutional_neural_network_hyperparameters.pth`: Hyperparameters required to reconstruct the f-k CNN model.
* `fk_convolutional_neural_network_model.pth`: Trained f-k CNN model state.
* `fk_convolutional_neural_network_spectral_feature_norm.pth`: Fitted f-k CNN feature (f-k spectra) min-max normalization.
* `fk_convolutional_neural_network_target_norm.pt`: Fitted f-k CNN target (seafloor pressure spectra) min-max normalization.
* `spectral_neural_network_hyperparameters.pth`: Hyperparameters required to reconstruct the spectral NN model.
* `spectral_neural_network_model.pth`: Trained spectral NN  model state.
* `spectral_neural_network_scalar_feature_norm.pth`: Fitted spectral NN scalar feature (depth, cosine squared wave direction) min-max normalization.
* `spectral_neural_network_spectral_feature_norm.pth`: Fitted spectral NN spectral feature (strain rate spectra) min-max normalization.
* `spectral_neural_network_target_norm.pth`: Fitted spectral NN target (seafloor pressure spectra) min-max normalization.

## Code/software

Example code for training and applying the models is organized into Python notebooks and can be accessed via GitHub at [https://github.com/jacobrdavis/neural_network-based_methods_for_das_ocean_surface_wave_measurement](https://github.com/jacobrdavis/neural_network-based_methods_for_das_ocean_surface_wave_measurement) or via the Zenodo archive at [https://doi.org/10.5281/zenodo.17381091](https://doi.org/10.5281/zenodo.17381091).

NetCDF files can be opened using any software that supports the netCDF file format (e.g., the Xarray and netCDF4 Python packages or MATLAB's ncread function).  See [https://www.unidata.ucar.edu/software/netcdf/](https://www.unidata.ucar.edu/software/netcdf/) for more information.

Model state  files (.pth) can be loaded using PyTorch. PyTorch v2.9.0 state dict (.pth) files are serialized Python OrderedDicts and can be opened using `torch.load`.

## Access information

Other publicly accessible locations of the data:

* The complete April to September 2023 mooring dataset is archived on the Arctic Data center at [https://doi.org/10.18739/A24J0B01J](https://doi.org/10.18739/A24J0B01J) (Thomson and Smith, 2024)

#### References:

Thomson, J., & Smith, M. (2024). Moorings along the seafloor cable route extending offshore from Oliktok Point, Alaska, from April to September of 2023 [Dataset]. Arctic Data Center. [https://doi.org/doi:10.18739/A24J0B01J](https://doi.org/doi:10.18739/A24J0B01J)
