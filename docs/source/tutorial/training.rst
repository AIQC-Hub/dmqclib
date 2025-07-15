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

Use ``dmqclib`` to generate a configuration template specifically for training.

.. code-block:: python

   import dmqclib as dm

   # This creates 'training_config.yaml' in '~/aiqc_project/config'
   dm.write_config_template(
       file_name="~/aiqc_project/config/training_config.yaml",
       module="train"
   )

Step 3.2: Customize the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the new ``~/aiqc_project/config/training_config.yaml`` file. Your goal is to tell ``dmqclib``:
1. Where to find the prepared dataset.
2. Where to save the trained model and its artifacts.

You will need to edit the ``path_info_sets`` and ``training_sets`` sections. We will also create a new directory to store our models.

Update your ``training_config.yaml`` to match the following, replacing the placeholder paths with the ones you created.

**Update your ``train_config.yaml`` file:**
   Modify the file to match the following structure, ensuring the paths align with your project setup.

   .. code-block:: yaml

    path_info_sets:
      - name: data_set_1
        common:
          base_path: ~/aiqc_project/data # Root output directory
        input:
          step_folder_name: training

  .. code-block:: yaml

    training_sets:
      - name: training_0001  # Your data set name
        dataset_folder_name: dataset_0001  # Your output folder

.. note::
   The training configuration file includes many other options for model selection, hyperparameter tuning, and cross-validation strategies. For a complete reference, see the :doc:`../../configuration/training` page.

Step 3.3: Run the Training Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, load the training configuration and execute the ``train_and_evaluate`` function.

.. code-block:: python

   config = dm.read_config("~/aiqc_project/config/training_config.yaml")
   dm.train_and_evaluate(config)

Understanding the Output
------------------------

After the commands finishes, ``dmqclib`` generates new folders inside your output directory (``~/aiqc_project/data/dataset_0001/``). The primary outputs are:

- **validate**: Contains detailed results from the cross-validation process, allowing you to inspect model performance across different data folds.
- **build**: Holds a comprehensive report of their evaluation performance on the held-out test dataset.
- **model**: Holds the final, trained model object(s) ready for classification.

Next Steps
----------

You have now successfully trained and evaluated a model! The final step is to use this model to classify new, unseen data.

Proceed to the next tutorial: :doc:`./classification`.
