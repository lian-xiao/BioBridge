# Transferable Binding Principles Meta-Learnig for Cross-Domain Drug-Target Interaction Prediction

This repository contains the implementation of the meta-learning framework described in the paper "Transferable Binding Principles Meta-Learnig for Cross-Domain Drug-Target Interaction Prediction".
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
   git clone https://github.com/lian-xiao/BioBridge.git
   cd BioBridge
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

This script will train the model using the meta-learning strategy as described in the paper.
