# Attention Mechanism on GAM-RHN

This repository provides the implementation of attention mechanisms integrated into Grouped Auxiliary Memory Recurrent Highway Networks (GAM-RHN). It includes modules for data preprocessing, model training, evaluation, and implementation of various attention mechanisms.

This work is based on a yet-to-be-published paper titled [Integrating Attention Mechanisms into Recurrent Highway Networks with Grouped Auxiliary Memory](https://shorturl.at/tU8X7), developed collaboratively by:

- **Shaurya**: [GitHub Profile](https://github.com/ladsad)
- **Devika**: [GitHub Profile](https://github.com/DevikaIyer23)

---

## Project Structure

```
Attention_Mechanism_on_GAM-RHN/
├── README.md                   # Project overview, installation instructions, and usage.
├── requirements.txt            # Python dependencies.
├── config.py                   # Configuration for hyperparameters, paths, etc.
├── data/
│   ├── data_loader.py          # Data loading and preprocessing scripts.
│   └── preprocess.py           # Functions for data preprocessing (e.g., tokenization).
├── models/
│   ├── rhn_cell.py             # Core RHN model architecture.
│   ├── gam_rhn.py              # Implementation of Grouped Auxiliary Memory logic.
│   ├── gam_rhn_attention.py    # RHN+GAM model with attention mechanisms.
│   └── attention.py            # Attention mechanism implementations.
├── train.py                    # Script for model training.
├── evaluate.py                 # Script for model evaluation.
└── notebooks/
    └── colab_notebook.ipynb    # Google Colab notebook for demonstration.
```

---

## Installation

Follow these steps to set up the project:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/Attention_Mechanism_on_GAM-RHN.git
   cd Attention_Mechanism_on_GAM-RHN
   ```

2. **Set up a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Download save-state files:**
   Download the pretrained model files from the following Google Drive link:
   [Download Save-State Files](https://drive.google.com/drive/folders/1cR_AYvDu26eNPHIlp91wycJ5-UBMh4lD?usp=drive_link)

   After downloading, place the files in the `models/save_state/` directory.

---

## Usage

### Data Preprocessing

Prepare the dataset by running the preprocessing script:
```sh
python data/preprocess.py
```

### Training

Train the model using the training script:
```sh
python train.py
```

### Evaluation

Evaluate the trained model using the evaluation script:
```sh
python evaluate.py
```

---

## Configuration

All configurable settings such as hyperparameters, data paths, and model parameters are defined in the `config.py` file. Adjust the configurations as required to tailor the model for your specific use case.

---

## Models

The `models/` directory contains key components of the project:

- **`rhn_cell.py`**: Core implementation of the Recurrent Highway Network (RHN).
- **`gam_rhn.py`**: Grouped Auxiliary Memory (GAM) logic.
- **`gam_rhn_attention.py`**: GAM-RHN model with integrated attention mechanisms.
- **`attention.py`**: Implementation of various attention mechanisms.

---

## Contributing

We welcome contributions! If you have ideas for improvements or fixes, please open an issue or submit a pull request. 

---

## References

This project builds upon concepts from the following resources:

- Paper: [Recurrent Highway Networks with Grouped Auxiliary Memory](https://ieeexplore.ieee.org/document/8932404).
- Repository: [Original GAM-RHN Implementation](https://github.com/WilliamRo/gam_rhn).

---