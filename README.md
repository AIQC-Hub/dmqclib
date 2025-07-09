# dmqclib

[![PyPI - Version](https://img.shields.io/pypi/v/dmqclib)](https://pypi.org/project/dmqclib/)
[![Anaconda - Version](https://anaconda.org/takayasaito/dmqclib/badges/version.svg)](https://anaconda.org/takayasaito/dmqclib)
[![Check Package](https://github.com/AIQC-Hub/dmqclib/actions/workflows/check_package.yml/badge.svg)](https://github.com/AIQC-Hub/dmqclib/actions/workflows/check_package.yml)
[![Codecov](https://codecov.io/gh/AIQC-Hub/dmqclib/graph/badge.svg?token=N6P5V9KBNJ)](https://codecov.io/gh/AIQC-Hub/dmqclib)
[![CodeFactor](https://www.codefactor.io/repository/github/aiqc-hub/dmqclib/badge)](https://www.codefactor.io/repository/github/aiqc-hub/dmqclib)

The *dmqclib* package offers helper functions and classes that simplify model building and evaluation for the *AIQC* project.

## Installation
The package is indexed on [PyPI](https://pypi.org/project/dmqclib/) and [Anaconda.org](https://anaconda.org/takayasaito/dmqclib), allowing you to install it using either *pip* or *conda*.

Using *pip*:
```bash
pip install dmqclib
```

Using *conda*:
```bash
conda install -c conda-forge dmqclib
```

## Usage

### 1. Dataset Preparation

#### 1.1 Create a Configuration File
First, create a configuration file that will serve as a template for preparing your dataset.

```python
import dmqclib as dm

config_file = "/path/to/config_file.yaml"
dm.write_config_template(config_file, module="prepare")
```

The function `write_config_template` generates a template configuration file at the specified location. You will need to edit this file to include entries relevant to the dataset you want to prepare for training. For detailed instructions, refer to the [Configuration](#configuration) section.

#### 1.2 Create a Training Dataset
Next, use the configuration file to create the training dataset.

```python
import dmqclib as dm

config_file = "/path/to/config_file.yaml"
dataset_name = "NRT_BO_001"

config = dm.read_config(config_file, module="prepare")
config.select(dataset_name)
dm.create_training_dataset(config)
```

The configuration file must contain the appropriate entries for the `dataset_name` variable to successfully execute the above command. The function `create_training_data_set` generates several folders and datasets, including:

- **summary**: Summary statistics of input data to estimate normalization values.
- **select**: Selected profiles with bad observation flags (positive) and associated profiles with good data (negative).
- **locate**: Observation records for both positive and negative profiles.
- **extract**: Extracted features for positive and negative observation records.
- **split**: Division of extracted feature records into training, validation, and test datasets.

### 2. Training and Evaluation

#### 2.1 Create a Training Configuration File
Before training your model, create a separate configuration file specifically for training purposes.

```python
import dmqclib as dm

training_config_file = "/path/to/training_config_file.yaml"
dm.write_config_template(training_config_file, module="train")
```

The function `write_config_template` will produce a template configuration file at the specified location. You will need to edit this file to include entries related to your model training and evaluation. For details, please refer to the [Configuration](#configuration) section.

#### 2.2 Train a Model and Evaluate Performance
After editing the configuration file, you are ready to train your model and evaluate its performance.

```python
import dmqclib as dm

training_config_file = "/path/to/training_config_file.yaml"
training_set_name = "NRT_BO_001"

training_config = dm.read_config(training_config_file, module="train")
training_config.select(training_set_name)
dm.train_and_evaluate(training_config)
```

Similar to the previous steps, ensure that the configuration file contains the necessary entries for the `training_set_name` variable. The function `train_and_evaluate` generates several folders and datasets, including:

- **validate**: Results from cross-validation processes.
- **build**: Developed models and evaluation results on the test dataset.

### 3. Classification

#### 3.1 Create a Configuration File
First, create a configuration file that will serve as a template for preparing your dataset.

```python
import dmqclib as dm

classification_config_file = "/path/to/classification_config_file.yaml"
dm.write_config_template(classification_config_file, module="classify")
```

The function `write_config_template` generates a template configuration file at the specified location. You will need to edit this file to include entries relevant to the dataset you want to prepare for training. For detailed instructions, refer to the [Configuration](#configuration) section.

#### 3.2 Create a Training Dataset
Next, use the configuration file to perform classification on all observations.

```python
import dmqclib as dm

classification_config_file = "/path/to/classification_config_file.yaml"
dataset_name = "NRT_BO_001"

classification_config = dm.read_config(classification_config_file, module="classify")
classification_config.select(dataset_name)
dm.classify_dataset(classification_config)
```

The configuration file must contain the appropriate entries for the `dataset_name` variable to successfully execute the above command. The function `classify_dataset` generates several folders and datasets, including:

- **summary**: Summary statistics of input data to estimate normalization values.
- **select**: Selected profiles with bad observation flags (positive) and associated profiles with good data (negative).
- **locate**: Observation records for both positive and negative profiles.
- **extract**: Extracted features for positive and negative observation records.
- **classify**: Classification results and report

## Configuration

### 1. Dataset Preparation
A configuration file for dataset preparation must include the following seven sections:

- **path_info_sets**: Information about paths and folders.
- **target_sets**: Names of target variables that include NRT/DM flags.
- **feature_sets**: Set of features utilised for training models.
- **feature_param_sets**: Parameters associated with the features.
- **step_class_sets**: Process steps necessary for creating training datasets.
- **step_param_sets**: Parameters corresponding to the process steps.
- **data_sets**: A list of datasets.

Among these sections, **path_info_sets** and **data_sets** require modification before running the data generation function.

#### Example of `path_info_sets`
```yaml
path_info_sets:
  - name: data_set_1
    common:
      base_path: /path/to/data # Modify this
    input:
      base_path: /path/to/input # Modify this
      step_folder_name: ""
```

In the *path_info_sets* section:
- `common:base_path` indicates the default output data location.
- `input:base_path` specifies the input data location.
- The entry `input:step_folder_name` can remain as an empty string (`""`).

#### Example of `data_sets`
```yaml
data_sets:
  - name: NRT_BO_001
    dataset_folder_name: nrt_bo_001
    input_file_name: nrt_cora_bo_test.parquet  # Modify this
```

In the *data_sets* section, you can edit all three entries above or add a new dataset entry as needed.

### 2. Training and Evaluation
A configuration file for training and evaluation must include the following five sections:

- **path_info_sets**: Information about paths and folders.
- **target_sets**: Names of target variables that include NRT/DM flags.
- **step_class_sets**: Process steps necessary for creating training datasets.
- **step_param_sets**: Parameters corresponding to the process steps.
- **training_sets**: A list of training sets.

Among these sections, **path_info_sets** and **training_sets** need to be modified before running the training function.

#### Example of `path_info_sets`
```yaml
path_info_sets:
  - name: data_set_1
    common:
      base_path: /path/to/data # Modify this
    input:
      base_path: /path/to/data # Modify this
      step_folder_name: "training"
```

In the *path_info_sets* section:
- `common:base_path` indicates the default output data location.
- `input:base_path` specifies the location for the input data.
- The entry `input:step_folder_name` can remain as "training".

#### Example of `training_sets`
```yaml
training_sets:
  - name: NRT_BO_001
    dataset_folder_name: nrt_bo_001
```

In the *training_sets* section, you may edit the existing entries or add a new training set entry as needed.

## Development Environment

### Package Manager
Using *[uv](https://docs.astral.sh/uv/)* is recommended when contributing modifications to the package. 
After the installation of *uv*, running `uv sync` inside the project will create the environment.

#### Example of Environment Setup
For example, the following commands create a new *conda* environment and set up the library environment with *uv*:

Using *conda*:
```bash
conda create --name aiqc -c conda-forge python=3.12 uv ruff
conda activate aiqc
```
Or using *mamba* (an alternative to conda for faster dependency resolution):
```bash
mamba create -n aiqc -c conda-forge python=3.12 uv ruff
mamba activate aiqc
```

To update the local environment with *uv*, navigate to your project directory:
```bash
cd /your/path/to/dmqclib
uv sync
```

### Unit Test

You can run unit tests using *pytest*.

```bash
uv run pytest -v
```

(Optional) You may need to install the library in editable mode at least once before running unit tests.

```bash
uv pip install -e .
```


### Python Linter
To lint the code under the *src* folder with [ruff](https://astral.sh/ruff), use the following command:

```bash
uvx ruff check src
```

and the unit test code under the *tests* folder:

```bash
uvx ruff check tests
```

### Code Formatter
To format the code under the *src* folder with [ruff](https://astral.sh/ruff), use the following command:

```bash
uvx ruff format src
```

and the unit test code under the *tests* folder:

```bash
uvx ruff format tests
```

## Documentation

### Read the Docs
The documentation of the package is available on the [Read the Docs web site](https://dmqclib.readthedocs.io/en/latest/index.html). The following steps are required to prepare the documents for the Read the Docs. The updated documents are automaticlly picked up by Read the Docs though GitHub web hook when merged to the main branch.

#### 1. Update docstrings
The docstings of all source files and unit test files are updated by Google Gemini. You need to an API key to run the following commands.

```bash
python ./docs/scripts/update_docstrings.py src scripts/prompt_main.txt
python ./docs/scripts/update_docstrings.py tests docs/scripts/prompt_unittest.txt
```

#### 2. Review docstrings
All files with the updated docstings need to be manually reviewd one by one. You can simply remove a few lines from the top and bottom in most files. Some file contain ckunks with "Issues:" that need be reviewd and corrected if necessary.   

#### 3. Update API Documents

Run the following command in the project root to generate API documents.

```bash
uv run sphinx-apidoc -f --remove-old --module-first -o docs/source/api src/dmqclib
```

#### 4. Build the documents

Run the following commands in the project root to build the documents.

```bash
cd docs
uv run make html
cd ..
```

You can review the produced documents opening docs/build/html/index.html by browser.

## Deployment

### Release to PyPI
The GitHub Action ([publish_to_pypi.yaml](https://github.com/AIQC-Hub/dmqclib/blob/main/.github/workflows/publish_to_pypi.yml)) automatically publishes the package to [PyPI](https://pypi.org/project/dmqclib/) whenever a GitHub release is created.

Alternatively, you can manually publish the package to PyPI:

```bash
uv build
uv publish --token pypi-xxxx-xxxx-xxxx-xxxx
```

### Release to Anaconda.org

Unlike using a GitHub Action for PyPI, publishing to [Anaconda.org](https://anaconda.org/takayasaito/dmqclib) is a manual process.

You’ll need the following tools:

  - conda-build
  - anaconda-client
  - grayskull

Install them with *conda* or *mamba* (preferably in a dedicated environment):
```bash
mamba install -c conda-forge conda-build anaconda-client grayskull
```

#### 1. Generate the Conda Recipe with Grayskull

From the project root, run:
```bash
grayskull pypi dmqclib
```

This creates a *meta.yaml* file in the *dmqclib/* directory.

> [!NOTE]
> Make sure to review the *meta.yaml* file before building the package.

#### 2. Build the Package
```bash
conda build dmqclib
```

This creates a *.conda* package in your local conda-bld directory (e.g., ~/miniconda3/conda-bld/noarch/).

#### 3. Upload to Anaconda.org

```bash
anaconda login
anaconda upload /full/path/to/conda-bld/noarch/dmqclib-<version>-<build>.conda
```

#### 4. Keep the Recipe Under Version Control

```bash
cp dmqclib/meta.yaml conda/meta.yaml
rm -r dmqclib
```

### Release to conda-forge

#### 1. Fork and Clone the Staged-Recipes Repository  

First, fork the conda-forge/staged-recipes repository on GitHub. Then clone your fork locally:

```bash
git clone https://github.com/<your_github>/staged-recipes.git    
```

#### 2. Create a New Branch from Main  

From inside your cloned staged-recipes folder:  

```bash
git checkout -b dmqclib-recipe  
```

#### 3. Generate the Conda Recipe  

Use Grayskull to generate a recipe for conda-forge:

```bash
cd staged-recipes
grayskull pypi dmqclib --strict-conda-forge  
```

*Grayskull* creates a folder named *dmqclib* directly under *staged-recipes*.

#### 4. Review the Generated meta.yaml  

Compare the generated file *dmqclib/meta.yaml* with the existing *meta_conda_forge.yaml* from the *dmqclib* repository. Adjust as needed to meet conda-forge guidelines.

#### 5. Commit and Push Your Changes  

```bash
git add dmqclib/meta.yaml  
git commit -m "Adding dmqclib"  
git push --set-upstream origin dmqclib-recipe  
```

#### 6. Open a Pull Request and Request a Review  

On GitHub, open a Pull Request from your dmqclib-recipe branch to the main branch of conda-forge/staged-recipes.  
Once the automated checks pass, leave a comment in your PR to request a review, for example:  
*@conda-forge/help-python, ready for review!*

#### 7. Keep the Recipe Under Version Control

```bash
cp dmqclib/meta.yaml /path/to/dmqclib/conda/meta_conda_forge.yaml
```
