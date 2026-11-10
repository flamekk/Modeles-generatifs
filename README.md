# 🧠 GENERATIVE DEEP LEARNING — PyTorch & MNIST

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Torchvision-0.25.0-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/MNIST-Computer%20Vision-blueviolet" />
  <img src="https://img.shields.io/badge/UV-Environment%20Management-6B4FBB" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

<p align="center">
  <strong>TP N°2 — Modèles génératifs avec PyTorch</strong>
  <br/>
  <i>From Image Classification to Latent Space Exploration and Image Generation</i>
</p>

---

## ✨ Project Overview

How can a neural network recognize handwritten digits, reconstruct images and even generate entirely new ones?

This project explores three complementary Deep Learning architectures using **PyTorch** and the **MNIST handwritten digit dataset**.

Starting with a discriminative neural network, the project progressively introduces generative learning through a classical Autoencoder (AE) and a Variational Autoencoder (VAE).

Beyond model training, the work focuses on understanding the mathematical foundations, evaluating reconstruction quality, exploring latent representations and generating new handwritten digits.

**The objective is not only to train models, but also to understand how they learn, represent and generate visual information.**

## 🎯 Project Highlights

| Component | Implementation |
|---|---|
| Dataset | MNIST – 60,000 training and 10,000 test images |
| Discriminative model | Multiclass digit classifier |
| Generative model 01 | Fully connected Autoencoder |
| Generative model 02 | Variational Autoencoder |
| Latent representation | 2-dimensional latent space |
| Training framework | PyTorch |
| Environment management | UV |
| Visualization | Matplotlib |
| Experimental outputs | 16 figures and quantitative metrics |

---

# 🏗️ 1. Project Architecture

The project follows three main learning approaches:

```text
                 MNIST DATASET
                 28 × 28 Images
                       |
          +------------+------------+
          |            |            |
          v            v            v
      CLASSIFIER       AE          VAE
          |            |            |
          v            v            v
      Digit Class   Encoder      Encoder
       (0–9)           |          μ, log σ²
                       |            |
                       v            v
                  Latent Code   Reparameterization
                    (2D)            |
                       |            v
                       v       Latent Sampling
                    Decoder         |
                       |            v
                       v         Decoder
                  Reconstruction   |
                                    v
                              Generated Images
```

### Three approaches, three different objectives

- **Classifier:** Learns to recognize and predict digit categories.
- **Autoencoder:** Learns a compressed representation and reconstructs the original image.
- **Variational Autoencoder:** Learns a probabilistic latent space that enables reconstruction and generation of new images.

---

# 📊 2. Experimental Results

The following results come from the executed notebook and the saved experimental metrics.

<details>
<summary><strong>🏆 Model Performance — Click to expand</strong></summary>

| Model | Evaluation Metric | Test Result |
|---|---|---:|
| Classifier | Accuracy | **96.15%** |
| Classifier | Macro F1-score | **0.9610** |
| Autoencoder | Reconstruction MSE | **0.04126** |
| Variational Autoencoder | Reconstruction MSE | **0.03945** |
| Mean training image | Reference MSE | 0.06747 |

The VAE also achieved:

| Metric | Result |
|---|---:|
| Binary Cross Entropy (BCE) | 141.30 |
| KL Divergence | 6.05 |
| Total BCE + KL objective | 147.34 |

*VAE loss values are estimated using five samples per test image. BCE, MSE and classification losses have different definitions and should not be directly compared.*

</details>

### 🔍 What do these results tell us?

- The classifier achieves 96.15% accuracy on the MNIST test set.
- Both generative models reconstruct images more accurately than the mean-image baseline according to MSE.
- The VAE combines reconstruction with probabilistic regularization.
- The two-dimensional latent space allows visual exploration of learned representations.

---

# 🟦 3. Discriminative Learning — MNIST Classifier

**Objective:** Recognize handwritten digits from 0 to 9.

The classifier learns the relationship between image pixels and their corresponding labels.

### Architecture

```text
Input Image (28 × 28)
         |
       Flatten
         |
    Linear (784 → 128)
         |
        ReLU
         |
    Linear (128 → 10)
         |
         v
    Digit Prediction
```

### Implementation

- MNIST dataset loading.
- Data preparation and normalization.
- Training, validation and test split.
- Fully connected neural network.
- Cross Entropy Loss.
- Adam optimizer.
- Accuracy and Macro F1-score.
- Confusion matrix.
- Misclassified image analysis.

### Experimental Performance

**Test Accuracy: 96.15%**

![Classifier confusion matrix](resultats/figures/03_matrice_confusion.png)

*Confusion matrix illustrating correct predictions and classification errors across the ten digit classes.*

Additional visualizations:

- [Training and validation curves](resultats/figures/02_courbes_classifieur.png)
- [Misclassified test images](resultats/figures/04_erreurs_classifieur.png)

---

# 🟩 4. Generative Learning — Autoencoder (AE)

**Objective:** Learn a compressed representation of handwritten digits and reconstruct the original images.

Unlike the classifier, the Autoencoder does not require class labels during training.

### Architecture

```text
       Original Image
          784 px
             |
             v
          Encoder
       784 → 128 → 32
             |
             v
       Latent Space
          2D (z)
             |
             v
          Decoder
        2 → 32 → 128
             |
             v
      Reconstructed Image
           784 px
```

### Implementation

- Encoder and decoder construction.
- Two-dimensional latent representation.
- Compression and reconstruction.
- Mean Squared Error (MSE).
- Adam optimizer.
- Latent space visualization.
- Class prototypes.
- Latent interpolation.

### Reconstruction Results

**Test Reconstruction MSE: 0.04126**

![AE and VAE reconstruction comparison](resultats/figures/11_comparaison_reconstructions.png)

*Original images and their corresponding AE and VAE reconstructions.*

### Exploring the latent space

The two-dimensional representation makes it possible to visualize how different handwritten digits are organized in the latent space.

![Autoencoder latent space](resultats/figures/07_espace_latent_ae.png)

Additional experiments:

- [Latent space visualization](resultats/figures/07_espace_latent_ae.png)
- [Class prototypes](resultats/figures/08_prototypes_ae.png)
- [Latent interpolation](resultats/figures/09_interpolation_ae.png)

---

# 🟪 5. Variational Autoencoder (VAE)

**Objective:** Learn a probabilistic latent representation capable of reconstructing existing images and generating new handwritten digits.

The VAE introduces a probabilistic encoder that predicts two parameters:

- μ (mean)
- log σ² (log variance)

### Architecture

```text
       Input Image
          784 px
             |
             v
          Encoder
       784 → 256 → 64
             |
       +-----+-----+
       |           |
       v           v
      Mean       Log Variance
       μ          log σ²
       |           |
       +-----+-----+
             |
             v
    Reparameterization
     z = μ + σ × ε
             |
             v
       Latent Vector
             |
             v
          Decoder
        2 → 64 → 256
             |
             v
     Reconstructed Image
```

### Mathematical Foundations

The reparameterization trick:

\[
z = \mu + \exp\left(\frac{\log \sigma^2}{2}\right)\epsilon
\]

where:

\[
\epsilon \sim \mathcal{N}(0,I)
\]

The loss function combines two objectives:

\[
\mathcal{L}_{VAE}
=
\mathcal{L}_{BCE}
+
\beta D_{KL}
\]

With:

\[
\beta = 1
\]

- **BCE:** Measures reconstruction quality.
- **KL Divergence:** Regularizes the latent distribution toward a standard normal distribution.

### Reconstruction Performance

**Test Reconstruction MSE: 0.03945**

![VAE reconstruction comparison](resultats/figures/11_comparaison_reconstructions.png)

### Generating New Handwritten Digits

One of the most interesting capabilities of a VAE is generating new images without requiring an original input image.

New latent vectors are sampled from:

\[
z \sim \mathcal{N}(0,I)
\]

The decoder transforms these vectors into new digit images.

![Generated MNIST digits](resultats/figures/13_generation_vae.png)

*Twenty generated samples obtained directly from randomly sampled latent vectors.*

### Exploring the 2D Latent Space

The two-dimensional latent space allows us to explore how the decoder transforms different latent coordinates into handwritten digits.

![VAE latent space grid](resultats/figures/14_grille_latente_vae.png)

Each image corresponds to a different position in the latent space.

The gradual transitions between digits illustrate how the VAE learns a structured probabilistic representation.

Additional visualizations:

- [VAE latent space](resultats/figures/12_espace_latent_vae.png)
- [Latent interpolation](resultats/figures/15_interpolation_vae.png)
- [Reconstruction comparison](resultats/figures/11_comparaison_reconstructions.png)
- [MSE comparison](resultats/figures/16_comparaison_mse.png)

---

# ⚖️ 6. Autoencoder vs Variational Autoencoder

| Aspect | Autoencoder (AE) | Variational Autoencoder (VAE) |
|---|---|---|
| Latent representation | Deterministic | Probabilistic |
| Encoder output | Latent vector | Mean and log variance |
| Sampling | Not intrinsic | Reparameterization |
| Reconstruction | Yes | Yes |
| New image generation | Not directly supported by a regular prior | Yes |
| Latent regularization | No KL term | KL divergence |
| Main objective | Compression and reconstruction | Probabilistic representation and generation |

### Experimental comparison

| Model | Test MSE |
|---|---:|
| Mean-image baseline | 0.06747 |
| Autoencoder | 0.04126 |
| VAE | **0.03945** |

The VAE achieves the lowest reconstruction MSE in this particular experiment. However, reconstruction error alone does not measure image realism or sample diversity.

---

# ⚙️ 7. Dataset & Experimental Protocol

### MNIST Dataset

The MNIST dataset contains grayscale handwritten digit images with dimensions of 28 × 28 pixels.

| Dataset partition | Number of images | Purpose |
|---|---:|---|
| Training | 55,000 | Model optimization |
| Validation | 5,000 | Best epoch selection |
| Testing | 10,000 | Final evaluation |
| **Total** | **70,000** | |

### Experimental configuration

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Latent dimension | 2 |
| VAE β | 1 |
| Hardware | CPU |

### Training configuration

| Model | Epochs | Best epoch |
|---|---:|---:|
| Classifier | 5 | 5 |
| Autoencoder | 15 | 15 |
| VAE | 20 | 20 |

*Reference execution: approximately five minutes on CPU. Execution time may vary depending on hardware and software versions.*

---

# 🛠️ 8. Technology Stack

| Technology | Role |
|---|---|
| Python 3.11 | Main programming language |
| PyTorch 2.10.0 | Deep Learning framework |
| Torchvision 0.25.0 | MNIST dataset and transformations |
| NumPy | Numerical computations |
| Matplotlib | Visualization and analysis |
| Torcheval | Evaluation dependencies |
| UV | Python environment and dependency management |
| Jupyter Notebook | Interactive experimentation |
| VS Code | Development environment |
| Git & GitHub | Version control |

---

# 📂 9. Repository Structure

```text
Modeles-generatifs/
│
├── README.md
├── TP2_Modeles_generatifs_MNIST.ipynb
│
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
│
├── docs/
│   └── images/
│       ├── notebook_apercu.png
│       └── notebook_conclusion.png
│
├── scripts/
│   ├── tp2_source.py
│   ├── construire_notebook.py
│   └── executer_notebook.py
│
└── resultats/
    ├── metriques.json
    ├── matrice_confusion.json
    ├── historique_classifieur.csv
    ├── historique_autoencodeur.csv
    ├── historique_vae.csv
    │
    ├── figures/
    │   ├── 01_exemples_mnist.png
    │   ├── 02_courbes_classifieur.png
    │   ├── 03_matrice_confusion.png
    │   ├── 04_erreurs_classifieur.png
    │   ├── ...
    │   └── 16_comparaison_mse.png
    │
    └── modeles/
        └── Generated locally (Git ignored)
```

---

# 🚀 10. Installation & Execution

### Prerequisites

- Python 3.11
- UV package manager
- Visual Studio Code
- Python and Jupyter extensions

### Step 1 — Clone the repository

```bash
git clone https://github.com/flamekk/Modeles-generatifs.git
cd Modeles-generatifs
```

### Step 2 — Install dependencies

```bash
uv sync --locked
```

### Step 3 — Open the notebook

Open:

`TP2_Modeles_generatifs_MNIST.ipynb`

Select the Python kernel corresponding to `.venv`.

### Step 4 — Execute the notebook

Run all cells sequentially.

Alternatively, execute the complete notebook from the terminal:

```bash
uv run python scripts/executer_notebook.py
```

The script executes the notebook, recalculates the metrics and regenerates the figures.

### Rebuild the notebook from its source

```bash
uv run python scripts/construire_notebook.py
```

**Important:** Rebuilding the notebook clears its previous outputs. It is not necessary when simply running the delivered notebook.

---

# 📈 11. Experimental Reproducibility

The repository includes:

- An executed Jupyter notebook.
- A fixed random seed.
- A reproducible train/validation split.
- A UV dependency lockfile.
- Saved evaluation metrics.
- Training history CSV files.
- Sixteen generated visualizations.
- Scripts to rebuild and execute the notebook.

Reference environment:

- Python 3.11.18
- PyTorch 2.10.0
- Torchvision 0.25.0
- CPU execution

This makes it possible to inspect the experiments and reproduce the workflow.

---

# 🧠 12. Skills & Learning Outcomes

Through this project, the following skills were developed:

**Deep Learning**
- Neural network architecture design.
- Forward propagation and model optimization.
- Training, validation and testing workflows.
- Loss functions and gradient-based optimization.

**Generative Modeling**
- Autoencoder architecture.
- Variational Autoencoder architecture.
- Latent representations.
- Reparameterization trick.
- KL divergence.
- Image reconstruction and generation.

**Computer Vision**
- MNIST dataset processing.
- Image classification.
- Reconstruction quality analysis.
- Visualization of generated images.

**Experimental Analysis**
- Accuracy and F1-score.
- MSE reconstruction evaluation.
- Confusion matrix interpretation.
- Latent space exploration.
- Comparison of neural network architectures.

**Development & Reproducibility**
- Python environment management with UV.
- Jupyter Notebook development.
- Git and GitHub.
- Experiment tracking and result persistence.

---

# 🔬 13. Limitations & Future Improvements

This practical project focuses on the first part of the generative models activity.

Possible extensions include:

- Exploring different latent dimensions.
- Testing different β values in the VAE.
- Comparing reconstruction losses.
- Improving generated image sharpness.
- Experimenting with Convolutional Autoencoders.
- Implementing Generative Adversarial Networks (GANs).
- Exploring diffusion-based generative models.

The current VAE is not class-conditional, so the generated digit class cannot be directly specified.

---

# 👩‍💻 Author

<p align="center">
  <strong>HIBA ZBARI</strong>
  <br/>
  Engineering Student — Big Data & Artificial Intelligence
  <br/>
  <br/>
  <a href="https://github.com/flamekk">
    <img src="https://img.shields.io/badge/GitHub-flamekk-181717?logo=github&logoColor=white" />
  </a>
  <br/>
  <br/>
  <strong>Academic Activity N°2 — Generative Models with PyTorch</strong>
  <br/>
  Mohamed YOUSSFI · 23 October 2026
</p>

---
