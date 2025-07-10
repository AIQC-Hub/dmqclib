Training & Evaluation
===================================

The training and evaluation configuration (`module="train"`) takes the prepared dataset from the previous stage and orchestrates the model building process. This includes cross-validation, model training, and final evaluation on a held-out test set.

This configuration is generally simpler than the preparation config because the complex feature engineering steps are already complete. Its main role is to specify the model, the validation strategy, and the locations of the input data and output models.

Key Configuration Sections
--------------------------

`path_info_sets`
^^^^^^^^^^^^^^^^^^^^^^

This section is crucial for linking the training workflow to the outputs of the preparation step and defining where to save the final model.

- **`common.base_path`**: The root directory containing the prepared dataset. **This must match the `common.base_path` used in the preparation step.**
- **`input.step_folder_name`**: The name of the subdirectory where the split data is located (e.g., `training`).
- **`model.base_path`**: The directory where the final trained model files will be saved.

.. code-block:: yaml

   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data
       input:
         step_folder_name: training
       model:
         base_path: /path/to/models
         step_folder_name: model

`step_class_sets`
^^^^^^^^^^^^^^^^^^^^^^

This powerful section allows you to define the core components of your training pipeline by specifying the Python classes to use for each step. This is where you choose your machine learning model and validation method.

- **`validate`**: The class for the cross-validation strategy (e.g., `KFoldValidation`).
- **`model`**: The class for the machine learning algorithm to be trained (e.g., `XGBoost`).

.. code-block:: yaml

   step_class_sets:
     - name: training_step_set_1
       steps:
         input: InputTrainingSetA
         validate: KFoldValidation
         model: XGBoost
         build: BuildModel

`step_param_sets`
^^^^^^^^^^^^^^^^^^^^^^

This section provides the parameters for the classes defined in `step_class_sets`. For example, you can specify the number of folds for k-fold cross-validation or provide hyperparameters for your model.

.. code-block:: yaml

   step_param_sets:
     - name: training_param_set_1
       steps:
         input: { }
         validate: { k_fold: 10 } # Set k to 10 for KFoldValidation
         model: { } # Model hyperparameters would go here
         build: { }

`training_sets`
^^^^^^^^^^^^^^^^^^^^^^

This is the main "assembly" section that defines a complete training job. It links together the prepared dataset with the path, target, and step configurations.

- **`name`**: A unique name for this training job.
- **`dataset_folder_name`**: The name of the folder containing the prepared data. **This must match the `dataset_folder_name` used in the preparation step.**
- **`path_info`**: The name of the path configuration to use from `path_info_sets`.
- **`target_set`**: The name of the target variable set to use.
- **`step_class_set`** & **`step_param_set`**: The names of the step configurations to use.

.. code-block:: yaml

   training_sets:
     - name: NRT_BO_001
       dataset_folder_name: nrt_bo_001
       path_info: data_set_1
       target_set: target_set_1_3
       step_class_set: training_step_set_1
       step_param_set: training_param_set_1

Full Example
------------

Below is a complete example of a `train_config.yaml` file. The lines you will most commonly edit are highlighted.

.. code-block:: yaml
   :caption: Full train_config.yaml example
   :emphasize-lines: 4, 8, 38

   path_info_sets:
     - name: data_set_1
       common:
         base_path: /path/to/data # EDIT: Must match the output path from preparation
       input:
         step_folder_name: training
       model:
         base_path: /path/to/models # EDIT: Directory to save model files
         step_folder_name: model

   target_sets:
     - name: target_set_1_3
       variables:
         - {name: temp, flag: temp_qc}
         - {name: psal, flag: psal_qc}
         - {name: pres, flag: pres_qc}

   step_class_sets:
     - name: training_step_set_1
       steps:
         input: InputTrainingSetA
         validate: KFoldValidation
         model: XGBoost
         build: BuildModel

   step_param_sets:
     - name: training_param_set_1
       steps:
         input: { }
         validate: { k_fold: 10 }
         model: { }
         build: { }

   training_sets:
     - name: NRT_BO_001
       dataset_folder_name: nrt_bo_001 # EDIT: Must match the prepared dataset folder
       path_info: data_set_1
       target_set: target_set_1_3
       step_class_set: training_step_set_1
       step_param_set: training_param_set_1
