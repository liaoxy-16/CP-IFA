# CP-IFA: Generative Causal Discovery via Counterfactual Perturbation and Information Flow Asymmetry
This repository contains the official implementation of **CP-IFA**, a novel unsupervised framework for discovering causal networks from observational data. By combining a generative Variational Autoencoder (VAE) architecture with counterfactual perturbation and information flow asymmetry, this method can accurately infer both the causal skeleton and the explicit causal directions.

## Framework Overview
<img src="fig 1.png" width="600">

CP-IFA performs causal discovery in two progressive stages:

1.  **Stage I: Causal Skeleton Learning (CP-VAE)**
    We introduce a Counterfactual Perturbation Module built upon a VAE framework. The model maps observation data to a latent manifold. By iteratively masking input variables and applying counterfactual intervention, we calculate the causal sensitivity score of each variable to construct an undirected causal skeleton.
2.  **Stage II: Direction Inference (IFAE)**
    Using the principle of Information Flow Asymmetry, we perform bidirectional prediction modeling for connected nodes. The Maximal Information Coefficient (MIC) is used to evaluate the independence between residuals and inputs to break Markov equivalence classes and determine edge directions.

## Repository Structure

The workflow is divided into four main Python scripts, designed to be executed sequentially:

-   `CPM_train.py`: Trains the CP-VAE models.
-   `CPM_evaluate.py`: Evaluates the trained models and extracts counterfactual perturbation losses.
-   `causal_sensitivity.py`: Computes the Causal Sensitivity Scores to form the causal skeleton.
-   `IFAE.py`: Infers the directed causal graph using Information Flow Asymmetry.

## Important Note on Final Graph Construction

Please note that `IFAE.py` calculates the causal directions for **all possible pairs** of variables in the dataset. To construct the final, accurate directed causal graph, you must combine these directional results with the **causal skeleton** obtained from the CPM stage (`causal_sensitivity.py`). 

Specifically, use the CPM sensitivity scores to threshold and filter out edges between irrelevant variables, and then apply the IFAE directions only to the remaining valid edges in the skeleton.
