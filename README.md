# JON - Justifiable Oncology Nemesis

JON is a tool that utilizes U-Net architecture and Bayesian probabilistic layers in order to detect areas with high probability of pathology changes on computer tomography scans. 

## Motivation

Medical image analysis is an area of great interest in both medicine and computer science. Pathological changes classificators, which are used in this research, have to fulfill following requirements:

- explainability - a doctor has to know the algorithm’s motivation to make decision, 
- speed - time to classify pathological changes has to be as short as possible.

One of the most popular explainable solutions is the Sliding Window algorithm. It processes every single pixel within its context. The main disadvantage of this approach is the fact that it is time-consuming, because medical images are usually three-dimensional and consist of millions of pixels.

The presentation describes a tool to reduce the number of pixels being processed. This operation highly optimizes the classification process.

Implementation of this tool uses a probabilistic Bayesian neural network for lung segmentation. The model is trained on scans from the computer tomography of healthy patients. Bayesian neural networks have layers providing non-deterministic output of the model which enable the program to calculate variance of predictions. The main assumption of the project is treating these high variance areas as anomalies.


## Inspirations
- [Fully automated algorithm for the detection of bone marrow oedema lesions in patients with axial spondyloarthritis – Feasibility study](https://www.sciencedirect.com/science/article/abs/pii/S0208521621000589)
- [We Know Where We Don't Know: 3D Bayesian CNNs for Credible Geometric Uncertainty](https://arxiv.org/abs/1910.10793)
