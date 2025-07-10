Classification
============================

The classification configuration file (`module="classify"`) directs ``dmqclib`` on how to use a pre-trained model to make predictions on a dataset. Its primary role is to link a model, an input data file, and an output location.

While it shares a structure with other configuration files, the classification config is simpler. You typically only need to edit two main sections: ``path_info_sets`` and ``classification_sets``. Other parameters related to features and targets are usually inherited from the model's training context.

Key Configuration Sections
--------------------------

`path_info_sets`
^^^^^^^^^^^^^^^^^^^^^^

This section defines all the necessary directory paths for the classification workflow. It tells ``dmqclib`` where to find the trained model, where the raw data is (if not specified elsewhere), and where to save the final output.

- **`common.base_path`**: The root directory where all outputs for this task will be saved.
- **`model.base_path`**: The path to the directory containing your trained model files from the training step.
- **`concat.step_folder_name`**: The name of the subdirectory within `common.base_path` where the final classified file will be stored.

.. code-block:: yaml
   :caption: Example path_info_sets
   :name: classify-path-info-sets

   path_info_sets:
     - name: data_set_1
       # The root directory for classification outputs
       common:
         base_path: /path/to/output_data
       # Points to the directory containing your trained models
       model:
         base_path: /path/to/models
         step_folder_name: model
       # Points to the directory containing your input files
       input:
         base_path: /path/to/input
         step_folder_name: ""
       # Defines the output sub-folder for results (e.g., /path/to/output_data/classify)
       concat:
         step_folder_name: classify

`classification_sets`
^^^^^^^^^^^^^^^^^^^^^^

This section defines a specific classification job. You can define multiple jobs in this list, each with a unique name.

- **`name`**: A unique identifier for this classification task (e.g., "NRT_BO_001").
- **`dataset_folder_name`**: The name of the folder where intermediate files will be stored. This should typically match the name used during preparation and training to maintain consistency.
- **`input_file_name`**: The name of the raw data file (e.g., a ``.parquet`` file) that you want to classify.
- **`path_info`**, **`step_class_set`**, etc.: These keys link this job to the corresponding configurations defined elsewhere in the file.

.. code-block:: yaml
   :caption: Example classification_sets
   :name: classify-sets

   classification_sets:
     - name: NRT_BO_001
       # A folder for intermediate files, should match training
       dataset_folder_name: nrt_bo_001
       # The raw data file to be classified
       input_file_name: nrt_cora_bo_test.parquet
       # Links to the 'path_info_sets' entry named 'data_set_1'
       path_info: data_set_1
       # Links to the 'step_class_sets' entry
       step_class_set: data_set_step_set_1
       # ... and so on

.. _full-classify-config-example:

Full Example
------------

Here is a complete example of a ``classify_config.yaml`` file. Notice that sections like `feature_sets` and `target_sets` are still present, as the framework expects them, but they are often just boilerplate inherited from the template because the feature logic is encapsulated within the trained model.

.. note::
   The most important sections to edit for a new classification task are highlighted with comments.

.. code-block:: yaml
   :caption: Full classify_config.yaml example
   :emphasize-lines: 4, 8, 70

   path_info_sets:
     - name: data_set_1
       common:
         base_path: /home/user/aiqc_project/data  # EDIT: Your root output directory
       input:
         base_path: /path/to/input # EDIT: Directory with input files
         step_folder_name: ""
       model:
         base_path: /home/user/aiqc_project/models # EDIT: Directory with your trained models
         step_folder_name: model
       concat:
         step_folder_name: classify

   # The following sections are often left as default
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
         - {feature: location, stats: {longitude: {min: 14.5, max: 23.5}}}
         - {feature: day_of_year, convert: sine}
         - {feature: profile_summary_stats5, stats: {temp: {mean: {min: 0, max: 12.5}}}}
         - {feature: basic_values3_plus_flanks, flank_up: 5}

   step_class_sets:
     - name: data_set_step_set_1
       steps:
         # These define the classes to use for each step of the process
         input: InputDataSetAll
         summary: SummaryDataSetAll
         select: SelectDataSetAll
         locate: LocateDataSetAll
         extract: ExtractDataSetAll
         model: XGBoost
         classify: ClassifyDataSetAll
         concat: ConcatDataSetAll

   step_param_sets:
     - name: data_set_param_set_1
       steps:
         input: {sub_steps: {rename_columns: false, filter_rows: false}}
         summary: { }
         select: { }
         locate: { }
         extract: { }
         model: { }
         classify: { }
         concat: { }

   # This is the main section defining the classification job
   classification_sets:
     - name: NRT_BO_001
       dataset_folder_name: nrt_bo_001
       input_file_name: nrt_cora_bo_test.parquet # EDIT: The data file to classify
       path_info: data_set_1
       target_set: target_set_1_3
       feature_set: feature_set_1
       feature_param_set: feature_set_1_param_set_3
       step_class_set: data_set_step_set_1
       step_param_set: data_set_param_set_1
