# Algorithmic Prediction and Intelligent Graph Analysis of Github Network Dynamics Using Machine Learning

This is my **Course Project**, aligned with the course - **Social Networks & Informatics**, and associated with my **Master's Degree in Computer Science** at the **University of Colorado Denver**.

## Project Description

### What?

In my project, the objective was to predict future links/relationships between users in a large **GitHub** Network and analyze the network dynamics within this complex social network using a number of Machine Learning algorithms, metrics, and plots. Various ML algorithms such as **Logistic Regression**, **Random Forest Classifier**, **SVM**, and **XGBoost** facilitated the analysis of user interactions and relationships in the GitHub Network, which were represented by a Directed Graph. **NetworkX** will help explore various aspects of network analysis such as node centrality and node degree, and **SHAP** values were used to find feature dependence, feature importance, and so on. These tools helped gain rich insights about the GitHub Network and how its structure influences relationships between users.

### Why?

Human societies operate as a collective in the form of social systems and structures. As such, there are endless numbers of social networks all around us, and they inevitably affect how we connect with other people and how information travel between us. Prediction of human behavior and understanding of the dynamics of these networks can be greatly helpful in many ways. A good example is a Social Media Platform, where content or friends suggested to a user can be made more relevant if the user and the network they operate in are analyzed and interpreted. Another example is Cybersecurity, where network insights can help identify malicious users or suspicious behavior. This project was a similar endeavor in which machine learning and graph analysis were used in tandem to peek under the veneer of a GitHub Social Network, revealing crucial patterns and facilitating the prediction of user interactions.

### How?

A [GitHub Social Network Dataset](./data/musae_git_edges.csv), with **37,700 nodes** corresponding to developers and **289,003 edges** corresponding to follower relationships, curated by the [Multi-Scale Attributed Node Embedding (MUSAE) Project](https://arxiv.org/abs/1909.13021) was obtained from the [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/). The initial step involved pre-processing the raw GitHub Network data to build a directed graph. This graph was then used to generate missing links/edges, which were then used alongside the existing edges to introduce a binary classification problem. Various NetworkX functions were used on the graph to calculate centrality measures such as **Page-Rank** and **Edge Betweenness**, find **shortest path lengths**, and extract **followship features** for all the source and target nodes. Using these features, training of the aforementioned ML models were carried out to predict the missing links/edges, thereby predicting user relationships. An in-depth analysis on the Social Network was performed using **Network plots**, **Heatmaps**, **Scatterplots**, **Histograms**, etc., to visualize insightful hidden details such as **Edge/Node Betweenness**, **Degree Distribution**, **Feature Correlation**, **Feature Distribution**, **Feature Relationships**, and so on. Additionally, **SHAP (SHapley Additive exPlanations)** values were computed, which facilitated an even more in-depth analysis of the graph features by way of **Feature importance**, **influence**, and **dependence**, found using **Force Plots**, **Summary Plots**, **Dependence Plots**, etc. This approach gave me insights into the most influential network features and how different models perform on social network data.

To summarize, this project explored the dynamics of social connections in a meaningful way, and by applying machine learning to graph-based data, it lead to some influential insights and information.

_For a comprehensive report of the Project, please refer to the [Project Report](./Suryakumar-Selvakumar-SNI-Final-Project-Report.pdf) document._

## Instructions to Run the Project

**IMPORTANT: Read all the instructions once before following them**

**1. Fork & Clone the repo:** <br>
`git clone https://github.com/your-username/your-forked-repo.git`

**2. Create a virtual env with Conda and Install all required dependencies:**

_Approach #1: Use provided `environment.yml`_

- `conda env create -f environment.yml`
- Activate conda env with `conda activate social-env`
- It is highly recommended to run this project on a Linux system. If that is not an option, please follow Approach #2.

_Approach #2: Use provided `requirements.txt`_

- `conda create -n your-env-name python=3.9`
- `conda activate your-env-name`
- `pip install -r requirements.txt`
- OS-specific Conda libraries that may be needed for the project may have to be installed manually.

**3. Steps to run the project files:**

- Create an ipykernel with your conda env using `python -m ipykernel install --user --name=social-env`
- Now, use `jupyter notebook` command to launch the jupyter notebook server, from which you can run the notebooks. Make sure to switch to `social-env` kernel before running the notebooks.

**4. The normalized dataset needed to train the ML models was slightly bigger than the maximum file size restriction of GitHub, thus it has been compressed to facilitate upload to this repo. Upon cloning the repo, extract the `github-normalized-dataset.zip` in the same directory, i.e., `/data` and now the notebooks 4 & 5 are ready to be run.**

**5. This project uses libraries such as cuML, CuPy, and cuDF which provide GPU-accelerated versions of ML algorithms, NumPy arrays, and Pandas DataFrames all of which take full advantage of an Nvidia GPU's computing power facilitated by the CUDA API. Hence, run times for the notebooks may vary depending on your GPU. Below is the GPU config that was used to build the project:**

- GPU: _NVIDIA GeForce RTX 4070 Ti_
- Driver Version: _560.35.03_
- CUDA Version: _12.6_

**6. It is to be noted that the latest versions of the aforementioned libraries are utilized, thus the latest CUDA version as given above is a must-have for them to work. Unfortunately, the latest version of CUDA is not yet available to be downloaded via Conda for a conda environment, thus it has to be manually downloaded and set to work with the user's conda environment:**

- Download the CUDA Toolkit 12.6.2 from this [Nvidia Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- Install the tool-kit and ensure your package is located in the following path: `/usr/local/cuda-12.6/`
- Activate your Conda env and find the environment activation script: <br>
  `mkdir -p $CONDA_PREFIX/etc/conda/activate.d`
- Create an activation script to set CUDA 12.6 paths:<br>
  `nano $CONDA_PREFIX/etc/conda/activate.d/cuda_env_vars.sh`
- Add the following lines to this file: <br>
  `export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}`<br>
  `export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`
- Save and exit the file. Now, everytime the user activates the conda env, it should automatically point to the installed cuda-toolkit.

**7. Alternatively, if the user wishes to downgrade CUDA or the libraries that enable GPU-accelerated computing, please refer to the [RAPIDS Installation Matrix](https://docs.rapids.ai/install/) to determine the compatible versions of CUDA & relevant libraries. For CuPy, any version you choose to install, find the compatible CUDA version online from their [official documentation](https://docs.cupy.dev/).**

- Make sure to find one common compatible CUDA version that works with all these libraries.
- Once finalized, run `conda install -c nvidia cudatoolkit=<cuda-version>`
- Inside any of the notebooks, import `cupy` and run `cupy.show_config()` and ensure that in the returned output, the same CUDA version that the user has installed is shown across the following sections:
  - `CUDA Root`
  - `nvcc PATH`
  - `CUDA Build Version`
  - `CUDA Driver Version`
  - `CUDA Runtime Version`

**8. The code in the notebooks - 4 & 5 are setup in such a way that new models won't be trained and the SHAP values won't get generated if those models or the values already exist locally. If the user wishes to train any of the models themselves or regenerate the SHAP values, ensure to delete or rename or move the respective pre-existing files.**

**9. File-Specific Information:**

- _Notebook - 1 - Pre-Processing-Data:_ Should run relatively fast with the generation of missing edges taking some time
- _Notebook - 2 - Extracting-Features:_ Computing Page-Rank, shortest paths, and followships for the whole network may take some time
- _Notebook - 3 - Normalizing-Dataset:_ Fastest notebook to run out of all.
- _Notebook - 4 - Training-and-Evaluating-Models:_ Heavily dependent on user's GPU, stronger the GPU, faster the run time. It took around 4 minutes for my GPU.
- _Notebook - 5 - Analyzing-Social-Network:_ Longest run time out of all notebooks, also dependent on user's GPU. Time taken for me - 20 minutes.

**10. For Clarifications regarding any of the NetworkX algorithms or functions used in the project, please refer to the [NetworkX Documentation](https://networkx.org/documentation/stable/reference/index.html)**
