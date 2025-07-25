Quick start with the minimum configurations
=============================================

This guide demonstrates how to run the whole machine learning process with the minimum configurations.

.. important::

    Before starting, ensure you have your raw input data (e.g., ``input_data.parquet``) available at the specified ``/path/to/input``.

Generate Configuration Files
-----------------------------

The following Python commands create three configuration files under ``/path/to/data/config`` and show summary statistics from your input data.

.. important::

    Make sure to save or note down the output from ``dm.get_summary_stats()`` as you will need it to update the configuration files in the next step. Specifically, pay attention to the ``min_max`` values for ``longitude``, ``latitude``, ``temp``, ``psal``, and ``pres``.

.. code-block:: python

    import os
    import polars as pl
    import dmqclib as dm

    print(f"dmqclib version: {dm.__version__}")

    # --- User-defined paths ---
    # !! IMPORTANT: Update these paths to your actual data and desired output locations !!
    input_file = "/path/to/input/input_data.parquet"
    data_path = "/path/to/data" # This will be the base path for generated configs, models, and results

    # --- Derived paths (do not change) ---
    config_path = os.path.join(data_path, "config")
    os.makedirs(config_path, exist_ok=True) # Ensure config directory exists

    config_file_prepare = os.path.join(config_path, "data_preparation_config.yaml")
    config_file_train = os.path.join(config_path, "training_config.yaml")
    config_file_classify = os.path.join(config_path, "classification_config.yaml")

    # Generate template configuration files
    print(f"Generating config templates in: {config_path}")
    dm.write_config_template(file_name=config_file_prepare, stage="prepare")
    dm.write_config_template(file_name=config_file_train, stage="train")
    dm.write_config_template(file_name=config_file_classify, stage="classify")
    print("Config templates generated.")

    # Get and print summary statistics for configuration
    print("\n--- Summary Statistics (All Variables) ---")
    stats_all = dm.get_summary_stats(input_file, "all")
    print(dm.format_summary_stats(stats_all))

    print("\n--- Summary Statistics (Profiles - key variables) ---")
    stats_profiles = dm.get_summary_stats(input_file, "profiles")
    print("Temperature (temp):")
    print(stats_profiles.filter(pl.col("variable")=="temp"))
    print("Salinity (psal):")
    print(stats_profiles.filter(pl.col("variable")=="psal"))
    print("Pressure (pres):")
    print(stats_profiles.filter(pl.col("variable")=="pres"))
    print("\nFormatted Profile Statistics:")
    print(dm.format_summary_stats(stats_profiles))


Update Configuration Files
-----------------------------
You need to update the three configuration files created by the commands above before running the main processes.

Configuration for the data preparation stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

2.  **Fill in Summary Statistics:**
    Use the summary statistics output from the `Generate Configuration Files`_ step to fill in the ``min_max`` values for ``longitude``, ``latitude``, ``temp``, ``psal``, and ``pres``. Replace all "..." placeholders.

    .. code-block:: yaml
       :caption: data_preparation_config.yaml: summary_stats_sets
       :emphasize-lines: 5, 6, 8, 9, 10, 12, 13, 14

       summary_stats_sets:
         - name: summary_stats_set_1
           stats:
             - name: location
               min_max: { longitude: { ... },  # <--- From stats_all output
                          latitude: { ... } }   # <--- From stats_all output
             - name: profile_summary_stats5
               min_max: { temp: { ... },       # <--- From stats_profiles output
                          psal: { ... },       # <--- From stats_profiles output
                          pres: { ... } }       # <--- From stats_profiles output
             - name: basic_values3
               min_max: { temp: { ... },       # <--- From stats_all output
                          psal: { ... },       # <--- From stats_all output
                          pres: { ... } }       # <--- From stats_all output

3.  **Configure Test Data Year(s):**
    Specify the year(s) for an independent test dataset (unseen data) by changing the ``remove_years`` or ``keep_years`` list.

    .. code-block:: yaml
       :caption: data_preparation_config.yaml: step_param_sets
       :emphasize-lines: 7

       step_param_sets:
         - name: data_set_param_set_1
           steps:
             input: { sub_steps: { rename_columns: false,
                                   filter_rows: true },
                      rename_dict: { },
                      filter_method_dict: { remove_years: [2023], # <--- Specify years to exclude from training/validation
                                            keep_years: [] } }

4.  **Specify Input File Name:**
    Ensure ``input_file_name`` matches the base name of your input data file.

    .. code-block:: yaml
       :caption: data_preparation_config.yaml: data_sets
       :emphasize-lines: 4

       data_sets:
         - name: dataset_0001
           dataset_folder_name: dataset_0001
           input_file_name: input_data.parquet # <--- Your input file's base name


Configuration for the training and validation stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Configuration for the classification stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

2.  **Copy Summary Statistics:**
    Copy the *entire* ``stats`` block from ``summary_stats_sets`` in your ``data_preparation_config.yaml`` and paste it here, replacing the "..." placeholder. The content should be identical.

    .. code-block:: yaml
       :caption: classification_config.yaml: summary_stats_sets
       :emphasize-lines: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

       summary_stats_sets:
         - name: summary_stats_set_1
           stats:
             # <--- Paste the full 'stats' block from data_preparation_config.yaml here
             - name: location
               min_max: { longitude: { ... },
                          latitude: { ... } }
             - name: profile_summary_stats5
               min_max: { temp: { ... },
                          psal: { ... },
                          pres: { ... } }
             - name: basic_values3
               min_max: { temp: { ... },
                          psal: { ... },
                          pres: { ... } }

3.  **Configure Classification Data Year(s):**
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
                                            keep_years: [2023] } } # <--- Specify years to *keep* for classification

4.  **Specify Input File Name:**
    Ensure ``input_file_name`` matches the base name of your input data file for classification.

    .. code-block:: yaml
       :caption: classification_config.yaml: data_sets
       :emphasize-lines: 4

       data_sets:
         - name: classification_0001
           dataset_folder_name: dataset_0001
           input_file_name: input_data.parquet # <--- Your input file's base name


Run the processes in all stages
----------------------------------

Once all configuration files are updated, the following Python commands will run the full machine learning process to generate the training, validation, and classification results.

The final classification results will be found under ``/path/to/data/classify``.

.. code-block:: python

    # Ensure config_path is defined from the "Generate Configuration Files" step
    # Example (if running this script separately):
    # import os
    # import dmqclib as dm
    # data_path = "/path/to/data"
    # config_path = os.path.join(data_path, "config")

    config_prepare = dm.read_config(os.path.join(config_path, "data_preparation_config.yaml"))
    dm.create_training_dataset(config_prepare)
    print("\nData preparation complete.")

    config_train = dm.read_config(os.path.join(config_path, "training_config.yaml"))
    dm.train_and_evaluate(config_train)
    print("\nTraining and evaluation complete.")

    config_classify = dm.read_config(os.path.join(config_path, "classification_config.yaml"))
    dm.classify_dataset(config_classify)
    print("\nClassification complete. Check results in /path/to/data/classify")
