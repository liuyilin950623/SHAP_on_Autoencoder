# SHAP_on_Autoencoder
Explaining Anomalies Detected by Autoencoders Using SHAP

Dataset: Boston Housing Dataset

Machine Learning Methods: Autoencoder, Kernel SHAP

Paper: Explaining Anomalies Detected by Autoencoders Using SHAP
https://arxiv.org/pdf/1903.02407.pdf

The implementation has 3 steps.
1. Select the top features with largest reconstruction errors.
2. For each feature in the list of top features:
   - We want to explain what features (other than itself) have led to the reconstruction error
   - Set the weights in the autoencoder that is specific to multiply the feature and keep all other weights
   - Use model agnostic Kernal SHAP to calculate the Shapley values
3. We then decide whether the feature is a contributing feature or an offsetting feature (depending on the sign of the reconstruction error)
Here, I made some minor adjustments to the original paper for the ease of interpretatbility. Contributing factors are marked as postive Shapley values.

* [SHAP_on_Autoencoder](./)
* [src](./src)
   * [train_autoencoder.py](./src/train_autoencoder.py)
* [config](./config)
   * [autoencoder_specifications.py](./config/autoencoder_specifications.py)
   * [environment.yml](./config/environment.yml)
* [log](./log)
   * [model.h5](./log/model.h5)
   * [model.json](./log/model.json)
* [sandbox](./sandbox)
   * [Demonstration.ipynb](./sandbox/Demonstration.ipynb)
* [README.md](./README.md)



