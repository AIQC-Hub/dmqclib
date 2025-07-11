# dmqclib

[![PyPI - Version](https://img.shields.io/pypi/v/dmqclib)](https://pypi.org/project/dmqclib/)
[![Conda - Version](https://img.shields.io/conda/vn/conda-forge/dmqclib)](https://anaconda.org/conda-forge/dmqclib)
[![Check Package](https://github.com/AIQC-Hub/dmqclib/actions/workflows/check_package.yml/badge.svg)](https://github.com/AIQC-Hub/dmqclib/actions/workflows/check_package.yml)
[![Codecov](https://codecov.io/gh/AIQC-Hub/dmqclib/graph/badge.svg?token=N6P5V9KBNJ)](https://codecov.io/gh/AIQC-Hub/dmqclib)
[![CodeFactor](https://www.codefactor.io/repository/github/aiqc-hub/dmqclib/badge)](https://www.codefactor.io/repository/github/aiqc-hub/dmqclib)

**dmqclib** is a Python library that provides a configuration-driven workflow for machine learning, simplifying dataset preparation, model training, and data classification. It is a core component of the AIQC project.

## Installation

The package is available on PyPI and conda-forge.

**Using pip:**
```bash
pip install dmqclib
```

**Using conda:**
```bash
conda install -c conda-forge dmqclib
```

## Core Concepts

The library is designed around a three-stage workflow:

1.  **Dataset Preparation:** Ingest raw data and transform it into a feature-rich dataset ready for training.
2.  **Training & Evaluation:** Train machine learning models and evaluate their performance using cross-validation.
3.  **Classification:** Apply a trained model to classify new, unseen data.

Each stage is controlled by a YAML configuration file, allowing you to define and reproduce your entire workflow with ease.

## Usage

The general workflow for any task in `dmqclib` follows these steps:

1.  **Generate a Configuration Template:** Create a starter YAML file for the task (e.g., `prepare`, `train`, `classify`).
2.  **Customize the Configuration:** Edit the YAML file to specify paths, dataset names, and other parameters.
3.  **Run the Task:** Load the configuration and execute the main function for the task.

### 1. Dataset Preparation

This workflow processes your input data and creates balanced training, validation, and test sets.

**Step 1: Generate a configuration template.**
```python
import dmqclib as dm

# This creates 'prepare_config.yaml' with predefined sections
dm.write_config_template(
    file_name="/path/to/prepare_config.yaml", 
    module="prepare"
)
```

**Step 2: Customize `prepare_config.yaml`.**
You must edit the file to set the correct input/output paths and define your dataset. See the [Configuration](#configuration) section for details.

**Step 3: Run the preparation process.**
```python
import dmqclib as dm

config_file = "/path/to/prepare_config.yaml"
dataset_name = "NRT_BO_001" # This must match a name in your config file

config = dm.read_config(config_file, module="prepare")
config.select(dataset_name)
dm.create_training_dataset(config)
```

This generates the following output folders:
- **summary**: Statistics of input data used for normalization.
- **select**: Profiles with bad observation flags (positive samples) and good profiles (negative samples).
- **locate**: Observation records for both positive and negative profiles.
- **extract**: Features extracted from the observation records.
- **training**: The final training, validation, and test datasets.

### 2. Model Training and Evaluation

This workflow uses the prepared dataset to train a model and evaluate its performance.

**Step 1: Generate a training configuration template.**
```python
import dmqclib as dm

dm.write_config_template(
    file_name="/path/to/train_config.yaml", 
    module="train"
)
```

**Step 2: Customize `train_config.yaml`.**
Edit the file to point to your prepared dataset and define training parameters.

**Step 3: Train and evaluate the model.**
```python
import dmqclib as dm

config_file = "/path/to/train_config.yaml"
training_set_name = "NRT_BO_001" # This must match a name in your config

config = dm.read_config(config_file, module="train")
config.select(training_set_name)
dm.train_and_evaluate(config)
```

This generates the following output folders:
- **validate**: Results from the cross-validation process.
- **build**: The final trained models and their evaluation results on the test dataset.

### 3. Data Classification

This workflow applies a trained model to classify all observations in a dataset.

**Step 1: Generate a classification configuration template.**
```python
import dmqclib as dm

dm.write_config_template(
    file_name="/path/to/classify_config.yaml", 
    module="classify"
)
```

**Step 2: Customize `classify_config.yaml`.**
Edit the file to point to the input data and the trained model.

**Step 3: Run classification.**
```python
import dmqclib as dm

config_file = "/path/to/classify_config.yaml"
dataset_name = "NRT_BO_001" # This must match a name in your config

config = dm.read_config(config_file, module="classify")
config.select(dataset_name)
dm.classify_dataset(config)
```

This workflow processes a dataset using a trained model and generates:
- **classify**: The final classification results and a summary report.

## Configuration

Configuration is managed via YAML files. The `write_config_template` function provides a starting point that you must customize for each module.

### 1. Dataset Preparation (`module="prepare"`)

The preparation config requires you to modify two key sections:

- **`path_info_sets`**: Defines the location of input and output data.
  ```yaml
  path_info_sets:
    - name: data_set_1
      common:
        base_path: /path/to/data # EDIT: Root output directory
      input:
        base_path: /path/to/input # EDIT: Directory with input files
        step_folder_name: ""
      split:
        step_folder_name: training
  ```

- **`data_sets`**: Defines a specific dataset to be processed.
  ```yaml
  data_sets:
    - name: NRT_BO_001
      dataset_folder_name: nrt_bo_001
      input_file_name: nrt_cora_bo_test.parquet # EDIT: The name of your input file.
  ```

### 2. Training and Evaluation (`module="train"`)

The training config links the prepared data to the model training process.

- **`path_info_sets`**: Defines where to find the prepared dataset and where to save model artifacts.
  ```yaml
  path_info_sets:
    - name: data_set_1
      common:
        base_path: /path/to/output_data # EDIT: Must match the `common.base_path` from the preparation step.
      input:
        step_folder_name: training # Locates the split data (e.g., /path/to/output_data/training).
      model:
        base_path: /path/to/models # EDIT: Directory to save model files.
        step_folder_name: model
  ```

- **`training_sets`**: Links to a dataset prepared in the previous workflow.
  ```yaml
  training_sets:
    - name: NRT_BO_001
      dataset_folder_name: nrt_bo_001 # Must match the `dataset_folder_name` from the preparation step.
  ```

### 3. Classification (`module="classify"`)

The classification config uses a trained model to classify new data.

- **`path_info_sets`**: Defines paths for raw data, models, and classification results.
  ```yaml
  path_info_sets:
    - name: data_set_1
      common:
        base_path: /path/to/output_data # EDIT: The root directory for classification outputs.
      input:
        step_folder_name: training
      model:
        base_path: /path/to/models # EDIT: Directory where model files are located.
        step_folder_name: model
      concat:
        step_folder_name: classify
  ```

- **`classification_sets`**: Defines a specific dataset to be classified.
  ```yaml
  classification_sets:
    - name: NRT_BO_001
      dataset_folder_name: nrt_bo_001　# Must match the `dataset_folder_name` from the preparation step
      input_file_name: nrt_cora_bo_test.parquet # EDIT: The raw data file to classify.
  ```

## Contributing & Development

We welcome contributions! Please use the following guidelines for development.

### Environment Setup

We recommend using **uv** for managing the development environment.

1.  Install `python`, `uv`, and `ruff` (e.g., via conda or mamba):
    ```bash
    # Using mamba (recommended)
    mamba create -n dmqclib-dev -c conda-forge python=3.12 uv ruff
    mamba activate dmqclib-dev
    ```

2.  Navigate to the project root and create the virtual environment:
    ```bash
    cd /path/to/dmqclib
    uv sync
    ```

3.  (Optional) Install the library in editable mode. This is sometimes needed before running tests.
    ```bash
    uv pip install -e .
    ```

### Running Tests

Unit tests are run with `pytest`.

```bash
uv run pytest -v
```

### Code Style (Linting & Formatting)

We use **Ruff** for linting and formatting.

**Linting:**
```bash
# Lint the library source code
uvx ruff check src

# Lint the test code
uvx ruff check tests
```

**Formatting:**
```bash
# Format the library source code
uvx ruff format src

# Format the test code
uvx ruff format tests
```

## Documentation (for Maintainers)

Project documentation is hosted on [Read the Docs](https://dmqclib.readthedocs.io/en/latest/index.html).

### Building Docs Locally

1.  **Update Docstrings (Requires Google Gemini API Key):**
    ```bash
    # Update docstrings for source files
    python ./docs/scripts/update_docstrings.py src docs/scripts/prompt_main.txt

    # Update docstrings for test files
    python ./docs/scripts/update_docstrings.py tests docs/scripts/prompt_unittest.txt
    ```

2.  **Review Docstrings:**
    Manually review all modified files. Remove generated headers/footers and correct any sections marked with "Issues:".

3.  **Update API Documents:**
    From the project root, run:
    ```bash
    uv run sphinx-apidoc -f --remove-old --module-first -o docs/source/api src/dmqclib
    ```

4.  **Build HTML:**
    From the project root, run:
    ```bash
    cd docs
    uv run make html
    cd ..
    ```
    You can view the generated site by opening `docs/build/html/index.html` in a browser.

## Deployment (for Maintainers)

### PyPI

The package is published to [PyPI](https://pypi.org/project/dmqclib/) automatically via a GitHub Action whenever a new release is created on GitHub.

### Anaconda.org (Manual)

Publishing to the `takayasaito` channel on [Anaconda.org](https://anaconda.org/takayasaito/dmqclib) is a manual process.

1.  **Install build tools:**
    ```bash
    mamba install -c conda-forge conda-build anaconda-client grayskull
    ```

2.  **Generate Recipe:**
    From the project root, run `grayskull pypi dmqclib`. This creates `dmqclib/meta.yaml`.

3.  **Build Package:**
    `conda build dmqclib`

4.  **Upload Package:**
    ```bash
    anaconda login
    anaconda upload /path/to/your/conda-bld/noarch/dmqclib-*.conda
    ```

5.  **Cleanup:**
    Copy `dmqclib/meta.yaml` to `conda/meta.yaml` for version control and remove the temporary `dmqclib` directory.

### conda-forge (Manual)

Submitting or updating the package on `conda-forge` involves creating a pull request to the `conda-forge/staged-recipes` repository.

1.  **Fork and clone** the `staged-recipes` repository.
2.  **Create a new branch** (e.g., `git checkout -b dmqclib-recipe`).
3.  **Generate a strict recipe:** `grayskull pypi dmqclib --strict-conda-forge`.
4.  **Review `meta.yaml`** and ensure it meets `conda-forge` standards.
5.  **Commit, push, and open a pull request** to the `staged-recipes` repository.
