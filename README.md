Reconstructing Intracellular Action Potentials from Cardiac Extracellular Waveforms

## Overview
This repository houses the Intelligent In-Cell Electrophysiology project, which focuses on the reconstruction of intracellular action potentials using a cutting-edge, physics-informed deep learning model. This model is meticulously trained on data obtained from nanoelectrode array recordings, pushing the boundaries of cardiac electrophysiology research.

## Directory Structure

### NEA_Patch
This directory contains data and scripts for a comparative study that simultaneously records intracellular action potentials (iAP) using both nanoelectrode arrays (NEA) and traditional patch clamp techniques. It evaluates key metrics such as Mean Absolute Error (MAE), R² (Coefficient of Determination), and Action Potential Duration (APD). The Signal to Noise Ratio (SNR) is analyzed to determine the NEA configurations that minimize measurement errors.

### NEA_Neighboring
Here, we explore the variations in Mean Absolute Error (MAE), R², and APD when recording from neighboring sites on an NEA. This section focuses on high-quality waveforms with a signal-to-noise ratio exceeding 90 dB, offering insights into the spatial consistency of NEA recordings.

### `EDA_XGBOOST`
This section delves into the exploratory data analysis and feature engineering of extracellular action potential signals (eAPS). Utilizing the XGBoost algorithm, we develop predictive models that estimate APD values ranging from 10 to 100 milliseconds based on eAP components. This approach illustrates the potential of machine learning in enhancing the interpretation of extracellular signals.

### `PIA-UNET`
In this section, we apply a physics-informed loss function to reconstruct iAPs from observed eAPs. This innovative approach incorporates both L1 regression and quantile regression techniques to provide robust estimates and assess uncertainty. This methodology represents a significant advancement in the predictive accuracy and reliability of iAP reconstructions from eAPs.

### `QPIA-UNET`
In this final section, we apply a quantile based physics-informed loss function to reconstruct iAPs from observed eAPs with 90- percent confidence interval. We furhter apply than on multi channel eAP recording and show the multi-channels predicted iAPs and their APDs change over time

## Getting Started
- Make sure to run Requirments.text it and install the needed libraries!
- in each folder run the "download_files.py" to downnload the neceessary data
- first start with folder general

To dive into this research, clone the repository and explore the structured directories, each equipped with necessary scripts and data sets to replicate the studies and extend the analyses as per your scientific curiosity.
## LICENSE
MIT License

Copyright (c) [2024] [BIONELAB]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

**Commons Clause**

The Software is provided to you by the Licensor under the MIT License, as defined above, with the following added condition:

**Non-Commercial**: You may not use the Software for any commercial purposes. For the purposes of this License, "Commercial" means primarily intended for or directed towards commercial advantage or monetary compensation.

Any license notice or attribution required by the MIT License must also include this Commons Clause condition.
![License: MIT with Commons Clause](https://img.shields.io/badge/License-MIT%20%2B%20Commons%20Clause-blue.svg)



