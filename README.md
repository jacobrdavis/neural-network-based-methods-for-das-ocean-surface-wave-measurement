[![DOI](https://zenodo.org/badge/1077904280.svg)](https://doi.org/10.5281/zenodo.17381090)

# Code for:

Davis, J. R., Thomson, J., Smith, M., Stanciu, A. C. (Submitted 2025) Neural network-based methods for ocean surface wave measurement using submarine distributed acoustic sensing (DAS)

## Abstract
Two new data-driven models for estimating ocean surface waves from  distributed acoustic sensing (DAS) submarine cable strain rate are developed using supervised machine learning on a 10-day dataset collected offshore of Oliktok Point, Alaska. The new models were trained on target data from seafloor pressure moorings at three sites spaced evenly along 27.1 km of cable and were benchmarked against an empirical transfer function method previously used to estimate waves from DAS. A model which uses convolutional neural networks to transform 2-km frequency-wavenumber strain spectra to seafloor pressure spectra outperforms the benchmark in wave height prediction (RMSE of 0.15 m versus 0.41 m) and period prediction (0.29 s versus 0.37 s). A two-hidden-layer, fully-connected neural network which transforms 1-D strain spectra to seafloor pressure spectra has less substantial improvement in skill than the convolutional neural network, yet outperforms the benchmark in wave height prediction (RMSE of 0.19 m versus 0.41 m) while maintaining the spatial resolution of individual DAS channels (16 m spacing). Regression-based machine learning is useful for estimating waves from DAS data when the pressure-strain relationship varies temporally and spatially across different wave conditions. Models can be applied to DAS data to measure waves with higher spatial resolution and longer temporal coverage than traditional methods, which often measure waves only at a single point.

### Plain Language Summary
Seafloor distributed acoustic sensing (DAS) uses fiber-optic cables to sense the surrounding ocean environment through measurements of cable deformation, or strain. In coastal environments, this strain can be used to infer the height of ocean surface waves above the cable. Here, we introduce two statistical models that learn to transform cable strain to wave height using data, and then we compare them to a previous model. The new models are developed using a dataset collected offshore of Oliktok Point, Alaska, on a fiber-optic cable typically used to provide phone and internet service to the area. The best-performing model uses DAS strain measurements collected over both time and space to estimate waves every few kilometers along the cable. When evaluated on the Oliktok Point dataset, errors in wave height estimates from this model were reduced by 60\% relative to the benchmark model. A second model, which uses DAS strain measurements collected only over time, also outperforms the benchmark (50\% reduction in error) and can be applied every several meters along the cable. These models are useful for wave height estimation from DAS in environments and conditions where cable strain cannot be directly related to the pressure induced by waves. The models can be applied to DAS data to measure waves at many locations simultaneously and over long periods, improving on traditional methods like buoys that record data at only one location.

<!-- <figure>
   <img src="./publication_figures/fig-alignment_categories_and_mss.png" width="576" alt="Wind-wave alignment categories in the storm-following reference frame and buoy mean square slope versus COAMPS-TC 10-m wind speed, classified by wind-wave alignment.">
   <figcaption><em>Left</em>: Wind-wave alignment categories in a storm-following reference frame; <em>Right</em>: Buoy mean square slope versus COAMPS-TC 10-m wind speed, classified by wind-wave alignment using an energy-weighted wave direction. </figcaption>
</figure> -->

## Data

Data and model states are archived on Dryad [(https://doi.org/10.5061/dryad.brv15dvnz)](https://doi.org/10.5061/dryad.brv15dvnz) and should be saved to the [input_data/](input_data/) directory. See [input_data/README.md](input_data/README.md) for more information.

## Structure

## Installation

1. Clone this repository.  In the terminal, run:
   ```sh
   git clone https://github.com/jacobrdavis/ocean-surface-wave-slopes-and-wind-wave-alignment-observed-in-hurricane-idalia.git
   ```
3. Download the data and move it to [input_data/](input_data/). (See the **Data** section above.)
4. Create a Python environment.  If using conda, run:
   ```sh
   conda env create -f environment.yml
   ```
5. Run any of the .ipynb notebooks.
