# ssVERDICT: Self-Supervised VERDICT-MRI for Enhanced Prostate Tumour Characterisation

This code extends the work discussed in the publication ssVERDICT: Self-Supervised VERDICT-MRI for Enhanced Prostate Tumour Characterisation by Snigdha Sen et al.

A preprint is available at: https://arxiv.org/abs/2309.06268

This code fits the Vascular, Extracellular and Restricted DIffusion for Cytometry in Tumours (VERDICT)-MRI model for prostate using self-supervised deep learning, adapted from https://github.com/sebbarb/deep_ivim.

There are six directories each pertaining to a fitting strategy, i.e. different combinations of fitting algorithms and biophysical models. 
1) baseline: VERDICT model fitted using the baseline autoencoder architecture discussed in the above-mentioned publication
2) dense-MLP: VERDICT model fitted using a more complex self-supervised MLP architecture than the baseline model
3) cnn-UNet: VERDICT model fitted using a CNN-based self-supervised UNet architecture
4) dki-1D: 1-dimensional DKI fitted using a self-supervised MLP architecture
5) dki-3D: 3-dimensional DKI fitted using a self-supervised MLP architecture
6) NLLS: VERDICT model fitted using the non-linear least square based Levenberg-Marquardt method

To run the training and inference pipeline, and generate the biomarker parameter maps:
1) Execute the **run_model_fitting.sh** script with an argument specifying the directory name for the fitting method to be executed
2) For e.g., $ _**./run_model_fitting.sh "dense-MLP"**_
