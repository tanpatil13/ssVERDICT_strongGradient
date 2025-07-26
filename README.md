# ssVERDICT: Self-Supervised VERDICT-MRI for Enhanced Prostate Tumour Characterisation

This code extends the work discussed in the publication ssVERDICT: Self-Supervised VERDICT-MRI for Enhanced Prostate Tumour Characterisation by Snigdha Sen et al.

A preprint is available at: https://arxiv.org/abs/2309.06268

This code fits the Vascular, Extracellular and Restricted DIffusion for Cytometry in Tumours (VERDICT)-MRI model for prostate using self-supervised deep learning, adapted from https://github.com/sebbarb/deep_ivim.

There are five directories each pertaining to a fitting strategy, i.e. combinations of fitting algorithms and biophysical models. 
1) baseline: VERDICT model fitted using a baseline autoencoder architecture discussed in the above-mentioned publication
2) dense-MLP: VERDICT model fitted using a more complex self-supervised MLP architecture than the baseline
3) cnn-UNet: VERDICT model fitted using the CNN-based self-supervised UNet architecture
4) 1D-DKI: 1-dimentional DKI fitted using a self-supervised MLP architecture
5) 3D-DKI: 3-dimensonal DKI fitted using a self-supervised MLP architecture

To run the training and inference pipeline, and generate the biomarker parameter maps:
1) Execute the **run_model_fitting.sh** script with an argument specifying the directory name for the fitting method to be executed
2) For e.g., $_**./run_model_fitting.sh "dense-MLP"**_
