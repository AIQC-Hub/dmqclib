Step 3: Training & Evaluation
=============================

With a properly prepared dataset, you are now ready to train and evaluate a machine learning model. This workflow uses the training, validation, and test sets created in the previous step to build a model, assess its performance using cross-validation, and generate final evaluation metrics.

Like all workflows in ``dmqclib``, this process is controlled by a dedicated YAML configuration file.

.. admonition:: Prerequisites

   This tutorial assumes you have successfully completed :doc:`./preparation`. The training process directly uses the output files generated in that step.

The Training Workflow
---------------------

The workflow mirrors the preparation step: you will generate a new configuration template, customize it to point to your data and model directories, and then run the training script.

Step 3.1: Generate the Configuration Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new Python script for this task (e.g., ``run_train.py``). Use ``dmqclib`` to generate a configuration template specifically for training.

.. code-block:: python

   import dmqclib as dm

   # This creates 'train_config.yaml' in the current directory
   dm.write_config_template(
       file_name="train_config.yaml",
       module="train"
   )
   print("Configuration template 'train_config.yaml' has been created.")

Step 3.2: Customize the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the new ``train_config.yaml`` file. Your goal is to tell ``dmqclib``:
1. Where to find the prepared dataset.
2. Where to save the trained model and its artifacts.

You will need to edit the ``path_info_sets`` and ``training_sets`` sections. We will also create a new directory to store our models.

1. **Create a directory for your models:**

   .. code-block:: bash

      mkdir -p ~/aiqc_project/models

2. **Update your ``train_config.yaml`` file:**
   Modify the file to match the following structure, ensuring the paths align with your project setup.

   .. code-block:: yaml

      path_info_sets:
        - name: data_set_1
          # EDIT: This must match the output path from the preparation step.
          common:
            base_path: /home/user/aiqc_project/data
          # This tells dmqclib to look for input inside the `training` sub-folder.
          input:
            step_folder_name: training
          # EDIT: This is where your final model files will be saved.
          model:
            base_path: /home/user/aiqc_project/models
            step_folder_name: model

  .. code-block:: yaml

      training_sets:
        - name: NRT_BO_001
          # This must match the `dataset_folder_name` from the preparation step.
          dataset_folder_name: nrt_bo_001

.. note::
   The training configuration file includes many other options for model selection, hyperparameter tuning, and cross-validation strategies. For a complete reference, see the :doc:`../../configuration/training` page.

Step 3.3: Run the Training Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, update your ``run_train.py`` script to load the training configuration and execute the ``train_and_evaluate`` function.

.. code-block:: python

   import dmqclib as dm

   # Path to your customized training configuration file
   config_file = "train_config.yaml"
   # This name must match the 'name' in the 'training_sets' section of your YAML
   training_set_name = "NRT_BO_001"

   print(f"Loading configuration for '{training_set_name}' from '{config_file}'...")
   config = dm.read_config(config_file, module="train")
   config.select(training_set_name)

   print("Starting model training and evaluation...")
   dm.train_and_evaluate(config)
   print("Training and evaluation complete!")

Run the script from your terminal:

.. code-block:: bash

   python run_train.py

Understanding the Output
------------------------

After the script finishes, ``dmqclib`` generates new folders inside your output directory (``~/aiqc_project/data/nrt_bo_001/``). The primary outputs are:

- **validate**: Contains detailed results from the cross-validation process, allowing you to inspect model performance across different data folds.
- **build**: Holds the final, trained model object(s) and a comprehensive report of their evaluation performance on the held-out test dataset. The model saved here is ready for classification.

Next Steps
----------

You have now successfully trained and evaluated a model! The final step is to use this model to classify new, unseen data.

Proceed to the next tutorial: :doc:`./classification`.
