Quick Start
=============================================

This guide demonstrates how to run the whole machine learning process with the minimum configurations.

Installation
-----------------------------

(Optional) Create a ``mamba``/``conda`` environment before installing ``dmqclib``.

.. code-block:: bash

   # conda
   conda create --name dmqclib -c conda-forge python=3.12 pip uv
   conda activate dmqclib

   # mamba
   mamba create -n dmqclib -c conda-forge python=3.12 pip uv
   mamba activate dmqclib


Use ``pip`` or ``conda``/``mamba`` to install ``dmqclib``.

.. code-block:: bash

   # pip
   pip install dmqclib

   # conda
   conda install -c conda-forge dmqclib

   # mamba
   mamba install -c conda-forge dmqclib


Download Raw Input Data
-----------------------------

You can get an input data set (``nrt_cora_bo_4.parquet``) from `Kaggle <https://www.kaggle.com/api/v1/datasets/download/takaya88/copernicus-marine-nrt-ctd-data-for-aiqc>`_.

Prepare Directory Structure
-----------------------------

The following Python commands create the directory structure for the input and output files.

.. code-block:: python

    import os
    import polars as pl
    import dmqclib as dm

    print(f"dmqclib version: {dm.__version__}")

    # !! IMPORTANT: Update these paths to your actual data and desired output locations !!
    input_file = "/path/to/input/nrt_cora_bo_4.parquet"
    data_path = "/path/to/data"

    config_path = os.path.join(data_path, "config")
    os.makedirs(config_path, exist_ok=True)

Stage 1: Data Preparation Stage
---------------------------------------------

The `prepare` workflow (`stage="prepare"`) is central to setting up your data for machine learning tasks within this library. It provides comprehensive control over the entire data processing pipeline, from preparing feature data sets from your raw data and creating the training, validation, and test data sets.

Template Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following Python commands create a configuration template for the data preparation stage.

.. code-block:: python

    config_file_prepare = os.path.join(config_path, "data_preparation_config.yaml")
    dm.write_config_template(file_name=config_file_prepare, stage="prepare")

Update the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** ``/path/to/data/config/data_preparation_config.yaml``

1.  **Update Data and Input Paths:**
    Adjust the ``base_path`` values in the ``path_info_sets`` section.

    .. code-block:: yaml
       :caption: data_preparation_config.yaml: path_info_sets
       :emphasize-lines: 4, 6

       path_info_sets:
         - name: data_set_1
           common:
             base_path: /path/to/data  # <--- Update this to your common data root
           input:
             base_path: /path/to/input # <--- Update this to where your input data is located
             step_folder_name: ""

2.  **Configure Test Data Year(s):**
    Specify the year(s) for an independent test dataset (unseen data) by changing the ``remove_years`` or ``keep_years`` list.

    .. code-block:: yaml
       :caption: data_preparation_config.yaml: step_param_sets
       :emphasize-lines: 7, 8

       step_param_sets:
         - name: data_set_param_set_1
           steps:
             input: { sub_steps: { rename_columns: false,
                                   filter_rows: true },
                      rename_dict: { },
                      filter_method_dict: { remove_years: [ 2023 ], # <--- Specify years to exclude from training/validation
                                            keep_years: [ ] } }

3.  **Specify Input File Name:**
    Ensure ``input_file_name`` matches the base name of your input data file.

    .. code-block:: yaml
       :caption: data_preparation_config.yaml: data_sets
       :emphasize-lines: 4

       data_sets:
         - name: dataset_0001
           dataset_folder_name: dataset_0001
           input_file_name: nrt_cora_bo_4.parquet # <--- Your input file's base name

Run the Data Preparation Stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the configuration file is updated, the following Python command will create input data for machine learning training and validation.

.. code-block:: python

    config_prepare = dm.read_config(os.path.join(config_path, "data_preparation_config.yaml"))
    dm.create_training_dataset(config_prepare)

Stage 2: Training & Evaluation
-----------------------------

The `train` workflow (`stage="train"`) is responsible for orchestrating the machine learning model building process. It takes the prepared dataset (the output from the `prepare` workflow) and handles critical steps such as cross-validation, actual model training, and final evaluation on a held-out test set.

Template Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following Python commands create a configuration template for the training & evaluation stage.

.. code-block:: python

    config_file_train = os.path.join(config_path, "training_config.yaml")
    dm.write_config_template(file_name=config_file_train, stage="train")

Update the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** ``/path/to/data/config/training_config.yaml``

1.  **Update Data Path:**
    Adjust the ``base_path`` in the ``path_info_sets`` section. This should be the same as the ``common.base_path`` you set in ``data_preparation_config.yaml``.

    .. code-block:: yaml
       :caption: training_config.yaml: path_info_sets
       :emphasize-lines: 4

       path_info_sets:
         - name: data_set_1
           common:
             base_path: /path/to/data # <--- Update this to your common data root

Run the Training & Evaluation Stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the configuration file is updated, the following Python command will run the machine learning processes to generate the training and validation results.

.. code-block:: python

    config_train = dm.read_config(os.path.join(config_path, "training_config.yaml"))
    dm.train_and_evaluate(config_train)

Stage 3: Classification
-----------------------------

The `classify` workflow (`stage="classify"`) is designed to apply a pre-trained machine learning model to new, unseen datasets to generate predictions. It leverages the same modular "building blocks" concept found in the `prepare` and `train` workflows, but its configuration is streamlined.

Template Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following Python commands create a configuration template for the classification stage.

.. code-block:: python

    config_file_classify = os.path.join(config_path, "classification_config.yaml")
    dm.write_config_template(file_name=config_file_classify, stage="classify")

Update the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File:** ``/path/to/data/config/classification_config.yaml``

1.  **Update Paths:**
    Adjust the ``base_path`` values for ``common``, ``input``, and ``model``.
    *   ``common.base_path``: Your common data root.
    *   ``input.base_path``: Where your input data for classification is located.
    *   ``model.base_path``: Where the trained model will be located (usually within your ``data_path``).

    .. code-block:: yaml
       :caption: classification_config.yaml: path_info_sets
       :emphasize-lines: 4, 6, 9

       path_info_sets:
         - name: data_set_1
           common:
             base_path: /path/to/data  # <--- Update to your common data root
           input:
             base_path: /path/to/input # <--- Update to your classification input data location
             step_folder_name: ""
           model:
             base_path: /path/to/data/dataset_0001 # <--- Update to where your trained model is
             step_folder_name: "model"

2.  **Configure Classification Data Year(s):**
    Specify the year(s) for the classification dataset. This is typically the test dataset year(s) you *removed* during data preparation.

    .. code-block:: yaml
       :caption: classification_config.yaml: step_param_sets
       :emphasize-lines: 8

       step_param_sets:
         - name: data_set_param_set_1
           steps:
             input: { sub_steps: { rename_columns: false,
                                   filter_rows: true },
                      rename_dict: { },
                      filter_method_dict: { remove_years: [],
                                            keep_years: [ 2023 ] } } # <--- Specify years to *keep* for classification

3.  **Specify Input File Name:**
    Ensure ``input_file_name`` matches the base name of your input data file for classification.

    .. code-block:: yaml
       :caption: classification_config.yaml: data_sets
       :emphasize-lines: 4

       data_sets:
         - name: classification_0001
           dataset_folder_name: dataset_0001
           input_file_name: nrt_cora_bo_4.parquet # <--- Your input file's base name

Run the Classification Stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the configuration file is updated, the following Python commands will run the machine learning processes to generate the classification results.

.. code-block:: python

    config_classify = dm.read_config(os.path.join(config_path, "classification_config.yaml"))
    dm.classify_dataset(config_classify)
