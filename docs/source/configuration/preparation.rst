Dataset Preparation
=================================

The dataset preparation configuration (`module="prepare"`) is the most detailed of all the workflows. It controls every aspect of the data processing pipeline, from reading raw files to feature engineering and creating the final training, validation, and test sets.

The 'Building Blocks' Concept
-----------------------------

The configuration is designed around a "building blocks" concept. You define various sets of configurations once, give them a name, and then combine them to create a complete data processing pipeline.

The primary sections are:
- **`path_info_sets`**: Define reusable path structures.
- **`target_sets`**: Define which variables are the prediction targets.
- **`feature_sets`**: Define which feature engineering methods to apply.
- **`feature_param_sets`**: Provide parameters for the feature methods.
- **`step_class_sets`**: Define the specific Python classes for each processing step (Advanced).
- **`step_param_sets`**: Provide parameters for the processing steps.
- **`data_sets`**: The main section that assembles a pipeline by referencing the named blocks from the sets above.

Key Configuration Sections
--------------------------

`path_info_sets`
^^^^^^^^^^^^^^^^^^^^^^

Defines the locations for input data and output artifacts. You can define multiple path configurations if you work with different storage locations.

- **`common.base_path`**: The root directory where all processed data will be saved.
- **`input.base_path`**: The directory where your raw input files are located.
- **`split.step_folder_name`**: The name of the subdirectory for the final training/validation/test splits (e.g., `training`).

.. code-block:: yaml

   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data
       input:
         base_path: /path/to/input
         step_folder_name: ""
       split:
         step_folder_name: training

`target_sets`
^^^^^^^^^^^^^^^^^^^^^^

Specifies the target variables for the machine learning model. For each variable, you also define its corresponding quality control (QC) flag column, which is used to identify good vs. bad data points.

.. code-block:: yaml

   target_sets:
     - name: target_set_1_3
       variables:
         - name: temp
           flag: temp_qc
         - name: psal
           flag: psal_qc

`feature_sets` & `feature_param_sets`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These two sections work together to control feature engineering.

- **`feature_sets`**: Lists the names of the feature engineering methods to apply.
- **`feature_param_sets`**: Provides detailed parameters for each of the features listed in `feature_sets`.

.. code-block:: yaml

   # A list of features to apply
   feature_sets:
     - name: feature_set_1
       features:
         - location
         - day_of_year
         - basic_values3_plus_flanks

   # Parameters for the features listed above
   feature_param_sets:
     - name: feature_set_1_param_set_3
       params:
         - feature: location
           stats: { longitude: { min: 14.5, max: 23.5 },
                    latitude: { min: 55, max: 66 } }
         - feature: day_of_year
           convert: sine
         - feature: basic_values3_plus_flanks
           flank_up: 5
           stats: { temp: { min: 0, max: 20 } }

`data_sets`
^^^^^^^^^^^^^^^^^^^^^^

This is the main "pipeline assembly" section. Each entry in this list defines a complete dataset to be processed by linking together the building blocks defined in the other sections.

- **`name`**: A unique name for this dataset job (e.g., `NRT_BO_001`).
- **`input_file_name`**: The specific raw data file to process.
- **`path_info`**: The name of the path configuration to use from `path_info_sets`.
- **`target_set`**: The name of the target configuration to use from `target_sets`.
- **`feature_set`**: The name of the feature list to use from `feature_sets`.
- ...and so on for all other sets.

.. code-block:: yaml

   data_sets:
     - name: NRT_BO_001
       dataset_folder_name: nrt_bo_001
       input_file_name: nrt_cora_bo_test.parquet
       path_info: data_set_1
       target_set: target_set_1_3
       feature_set: feature_set_1
       feature_param_set: feature_set_1_param_set_3
       step_class_set: data_set_step_set_1
       step_param_set: data_set_param_set_1


Full Example
------------

Below is a complete example of a `prepare_config.yaml` file. The lines you will most commonly edit are highlighted.

.. code-block:: yaml
   :caption: Full prepare_config.yaml example
   :emphasize-lines: 4, 7, 72

   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data # EDIT: Root output directory
       input:
         base_path: /path/to/input # EDIT: Directory with input files
         step_folder_name: ""
       split:
         step_folder_name: training

   target_sets:
     - name: target_set_1_3
       variables:
         - {name: temp, flag: temp_qc}
         - {name: psal, flag: psal_qc}
         - {name: pres, flag: pres_qc}

   feature_sets:
     - name: feature_set_1
       features:
         - location
         - day_of_year
         - profile_summary_stats5
         - basic_values3_plus_flanks

   feature_param_sets:
     - name: feature_set_1_param_set_3
       params:
         - feature: location
           stats: { longitude: { min: 14.5, max: 23.5 },
                    latitude: { min: 55, max: 66 } }
         - feature: day_of_year
           convert: sine
         - feature: profile_summary_stats5
           stats: { temp: { mean: { min: 0, max: 12.5 } },
                    psal: { mean: { min: 2.9, max: 12 } },
                    pres: { mean: { min: 24, max: 105 } } }
         - feature: basic_values3_plus_flanks
           flank_up: 5
           stats: { temp: { min: 0, max: 20 },
                    psal: { min: 0, max: 20 },
                    pres: { min: 0, max: 200 } }

   step_class_sets:
     - name: data_set_step_set_1
       steps:
         input: InputDataSetA
         summary: SummaryDataSetA
         select: SelectDataSetA
         locate: LocateDataSetA
         extract: ExtractDataSetA
         split: SplitDataSetA

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input: { sub_steps: { rename_columns: false, filter_rows: false } }
         summary: { }
         select: { }
         locate: { }
         extract: { }
         split: { test_set_fraction: 0.1, k_fold: 10 }

   data_sets:
     - name: NRT_BO_001
       dataset_folder_name: nrt_bo_001
       input_file_name: nrt_cora_bo_test.parquet  # EDIT: Your input filename
       path_info: data_set_1
       target_set: target_set_1_3
       feature_set: feature_set_1
       feature_param_set: feature_set_1_param_set_3
       step_class_set: data_set_step_set_1
       step_param_set: data_set_param_set_1
