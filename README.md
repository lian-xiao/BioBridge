# Multi-Domain Aware Meta-Learning Framework for Drug-Target Interactions

This repository contains the implementation of the meta-learning framework described in the paper "Bridging Data and Domain Gaps: A Multi-Domain Aware Meta-Learning Framework for Interpretable Drug-Target Interactions". The framework is designed to enhance the interpretability of drug-target interactions by leveraging a multi-domain aware approach.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Code](#running-the-code)
  - [Regular Learning Mode](#regular-learning-mode)
  - [Meta-Learning Mode](#meta-learning-mode)
- [Citation](#citation)
- [License](#license)

## Requirements

- Python 3.8 or higher
- PyTorch (compatible with your CUDA version, if using GPU)
- Learn2Learn
- PyTorch Lightning
- And other dependencies listed in `requirements.txt`

## Installation

To set up the environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Multi-Domain-Meta-Learning.git
   cd Multi-Domain-Meta-Learning
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirement.txt
   ```

## Running the Code

### Regular Learning Mode

To run the regular learning mode, execute the following command:

```bash
python Dti_cnn_main.py
```

This script will train the model using the regular learning approach as described in the paper.

### Meta-Learning Mode

To run the meta-learning mode, execute the following command:

```bash
python Dti_cnn_meta.py
```

This script will train the model using the multi-domain aware meta-learning framework as described in the paper.
