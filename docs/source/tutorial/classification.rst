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

Use ``dmqclib`` to generate the configuration template for the ``classify`` module.

.. code-block:: python

   import dmqclib as dm

   # This creates 'classification_config.yaml' in '~/aiqc_project/config'
   dm.write_config_template(
       file_name="~/aiqc_project/config/classification_config.yaml",
       module="classify"
   )

Step 4.2: Customize the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the new ``~/aiqc_project/config/classification_config.yaml`` file. You need to configure the paths for the input data, the trained model, and the final output.

Update your ``classification_config.yaml`` file to match the following, ensuring the paths are correct for your project setup.

.. code-block:: yaml

    path_info_sets:
      - name: data_set_1
        common:
          base_path: ~/aiqc_project/data # Root output directory
        input:
          base_path: ~/aiqc_project/input # Directory with input files
          step_folder_name: ""
        model:
          base_path: ~/aiqc_project/data/dataset_0001/model # Directory with models
        concat:
          step_folder_name: classify # Directory with classification results

.. code-block:: yaml

   classification_sets:
      - name: classification_0001  #Your classification name
        dataset_folder_name: dataset_0001  # Your output folder
        input_file_name: nrt_cora_bo_4.parquet   # Your input filename

.. note::
   The classification configuration has similar options to both preparation and training. For a complete reference, see the :doc:`../../configuration/classification` page.

Step 4.3: Run the Classification Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, load the configuration and execute the ``classify_dataset`` function.

.. code-block:: python

   config = dm.read_config("~/aiqc_project/config/classification_config.yaml")
   dm.classify_dataset(config)

Understanding the Output
------------------------

After the commands finishes, your output directory (e.g., ``~/aiqc_project/data``) will contain a new folder named ``dataset_0001`` (from ``dataset_folder_name``). Inside, you will find several subdirectories:

- **summary**: Contains summary statistics of the input data, used for normalization.
- **select**: Stores all profiles.
- **locate**: Contains all observation records.
- **extract**: Holds the features extracted from the observation records.
- **classify**: The final output, containing a ``.parquet`` file with the original data plus new columns for the model's predictions and prediction probabilities, and s summary report detailing the classification results.

Conclusion
----------

Congratulations! You have successfully completed the entire ``dmqclib`` workflow, from raw data preparation to training a model and using it to generate predictions.

You now have a powerful, repeatable pipeline for your machine learning tasks. You can easily adapt the configuration files to process new datasets or experiment with different models and features.
