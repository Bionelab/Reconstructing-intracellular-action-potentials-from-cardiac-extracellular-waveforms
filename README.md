# Reconstructing Intracellular Action Potentials from Cardiac Extracellular Waveforms

## Overview
This repository houses the Intelligent In-Cell Electrophysiology project, which focuses on the reconstruction of intracellular action potentials using a cutting-edge, physics-informed deep learning model. This model is meticulously trained on data obtained from nanoelectrode array recordings, pushing the boundaries of cardiac electrophysiology research.

## Directory Structure

### `fig2_part1_patch_vs_NEA`
This directory contains data and scripts for a comparative study that simultaneously records intracellular action potentials (iAP) using both nanoelectrode arrays (NEA) and traditional patch clamp techniques. It evaluates key metrics such as Mean Absolute Error (MAE), R² (Coefficient of Determination), and Action Potential Duration (APD). The Signal to Noise Ratio (SNR) is analyzed to determine the NEA configurations that minimize measurement errors.

### `fig2_p2_iap_neighboring_NEAs`
Here, we explore the variations in Mean Absolute Error (MAE), R², and APD when recording from neighboring sites on an NEA. This section focuses on high-quality waveforms with a signal-to-noise ratio exceeding 90 dB, offering insights into the spatial consistency of NEA recordings.

### `fig3_xgboost_eap_iap`
This section delves into the exploratory data analysis and feature engineering of extracellular action potential signals (eAPS). Utilizing the XGBoost algorithm, we develop predictive models that estimate APD values ranging from 10 to 100 milliseconds based on eAP components. This approach illustrates the potential of machine learning in enhancing the interpretation of extracellular signals.

### `fig4_Deep_Learning`
In the final section, we apply a physics-informed loss function to reconstruct iAPs from observed eAPs. This innovative approach incorporates both L1 regression and quantile regression techniques to provide robust estimates and assess uncertainty. This methodology represents a significant advancement in the predictive accuracy and reliability of iAP reconstructions from eAPs.

## Getting Started
To dive into this research, clone the repository and explore the structured directories, each equipped with necessary scripts and data sets to replicate the studies and extend the analyses as per your scientific curiosity.

