Step 4: Classification
======================

You have prepared a dataset and trained a model. The final step is to put that model to work by classifying a full dataset. This workflow applies your trained model to every observation in an input file, adding predictions and probability scores as new columns.

This is the culmination of the ``dmqclib`` pipeline, turning your machine learning model into a practical tool for data analysis.

.. admonition:: Prerequisites

   This tutorial assumes you have successfully completed :doc:`./training`. You will need:

   - The trained model saved in your ``models`` directory.
   - The original raw data file (``nrt_cora_bo_4.parquet``) to classify.

The Classification Workflow
---------------------------

The process is consistent with the previous stages: generate a configuration template, customize it to point to your data and model, and then run the classification script.

Step 4.1: Generate the Configuration Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a final Python script for this task (e.g., ``run_classify.py``). Use ``dmqclib`` to generate the configuration template for the ``classify`` module.

.. code-block:: python

   import dmqclib as dm

   # This creates 'classify_config.yaml' in the current directory
   dm.write_config_template(
       file_name="classify_config.yaml",
       module="classify"
   )
   print("Configuration template 'classify_config.yaml' has been created.")

Step 4.2: Customize the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the new ``classify_config.yaml`` file. You need to configure the paths for the input data, the trained model, and the final output.

Update your ``classify_config.yaml`` file to match the following, ensuring the paths are correct for your project setup.

.. code-block:: yaml

   path_info_sets:
     - name: data_set_1
       # EDIT: The root directory where classification outputs will be saved.
       common:
         base_path: /home/user/aiqc_project/data
       # This section is used internally and can often be left as is.
       input:
         step_folder_name: training
       # EDIT: Point this to the directory where your trained models are stored.
       model:
         base_path: /home/user/aiqc_project/models
         step_folder_name: model
       # This defines the sub-folder for the final output.
       concat:
         step_folder_name: classify

.. code-block:: yaml

   classification_sets:
     - name: NRT_BO_001
       # This must match the folder name used during preparation and training.
       dataset_folder_name: nrt_bo_001
       # EDIT: This is the raw data file you want to classify.
       input_file_name: nrt_cora_bo_4.parquet

.. note::
   The classification configuration has fewer options than preparation or training, as its primary role is execution. For a complete reference, see the :doc:`../../configuration/classification` page.

Step 4.3: Run the Classification Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, update your ``run_classify.py`` script to load the configuration and execute the ``classify_dataset`` function.

.. code-block:: python

   import dmqclib as dm

   # Path to your customized classification configuration file
   config_file = "classify_config.yaml"
   # This name must match the 'name' in the 'classification_sets' of your YAML
   dataset_name = "NRT_BO_001"

   print(f"Loading configuration for '{dataset_name}' from '{config_file}'...")
   config = dm.read_config(config_file, module="classify")
   config.select(dataset_name)

   print("Starting dataset classification...")
   dm.classify_dataset(config)
   print("Classification complete!")

Run the script from your terminal:

.. code-block:: bash

   python run_classify.py

Understanding the Output
------------------------

After the script completes, a new directory named **classify** is created inside your ``data/nrt_bo_001`` folder. This directory contains:

- A ``.parquet`` file with the original data plus new columns for the model's predictions and prediction probabilities.
- A summary report detailing the classification results (e.g., the distribution of predicted classes).

Conclusion
----------

Congratulations! You have successfully completed the entire ``dmqclib`` workflow, from raw data preparation to training a model and using it to generate predictions.

You now have a powerful, repeatable pipeline for your machine learning tasks. You can easily adapt the configuration files to process new datasets or experiment with different models and features.
