# Disaggregating Dengue Case Counts in DKI Jakarta and Their Implications for Public Health Policy Planning

This project is the Final Year Project (FYP) for the Data Science and Economics (DSE) program at the National University of Singapore. It focuses on disaggregating province-level dengue case counts to the district level in DKI Jakarta using a Bayesian Hierarchical Model with deep generative priors — specifically Aggregated Variational Autoencoders (aggVAE) and Aggregated Gaussian Processes (aggGP).

## Objectives
- To generate high-resolution dengue prevalence estimates from aggregated data.
- To evaluate the efficacy of deep generative and machine learning models in disaggregation under data-sparse settings.
- To assess how different disaggregation methods influence downstream cost-effectiveness analysis and healthcare policy decisions.

## Methodology
- Implemented aggVAE and aggGP models based on Semenova et al. (2022, 2023) within a Bayesian Hierarchical framework.
- Compared these models against traditional baselines: random forests, population-proportional splitting, and equal splitting.
- Tuned model hyperparameters, evaluated convergence (e.g., improved \(\hat{R}\)), and conducted robustness checks.
- Incorporated geospatial covariates including built-up area, road density, nightlights, population density, and HDI.
- Assessed predictive accuracy of case counts and prevalence, as well as preservation of district-to-province prevalence ratios.

## Key Findings
- Population-proportional splitting consistently achieved the highest accuracy in disaggregated case count and prevalence predictions.
- The aggVAE model outperformed others in preserving district-to-province prevalence ratios, which are key for scaling costs in cost-effectiveness analysis.
- While population-proportional splitting remains the most practical method for policymakers, aggVAE offers a promising, flexible, and interpretable approach for future work in disease mapping.

## Repository Structure
- `models/`: Implementations of aggVAE, aggGP, and baseline models (random forest, heuristics).
- `data/`: Dengue case data and geospatial covariates (e.g., shapefiles, infrastructure, urban features). Datasets containing the dengue case counts at the district- and province-level are confidential and are therefore, not included in this repository.
- `src/`: Scripts for data preprocessing, model training, diagnostics, and evaluation.

## Acknowledgements
Special thanks to Dr Swapnil Mishra and Dr Huang Ta-Cheng for their guidance and support throughout this project.
