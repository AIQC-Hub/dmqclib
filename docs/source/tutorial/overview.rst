Overview
========

 As long as configration files are properly set, it is very easy to run a three-stage workflow provided by ``dmqclib``.

1.  **Dataset Preparation**
.. code-block:: python

   import dmqclib as dm

   # Read a configuration file
   prepare_config_file = "/path/to/prepare_config.yaml"
   dataset_name = "dataset_0001"
   prepare_config = dm.read_config(prepare_config_file, module="prepare")
   prepare_config.select(dataset_name)

   # Run data set preparation
   dm.create_training_dataset(prepare_config)

2.  **Training & Evaluation:**
.. code-block:: python
   import dmqclib as dm

   # Read a configuration file
   training_config_file = "/path/to/training_config.yaml"
   training_name = "training_0001"
   training_config = dm.read_config(training_config_file, module="train")
   training_config.select(training_name)

   # Run training and evaluation
   dm.train_and_evaluate(training_config)

3.  **Classification:** Apply a trained model to classify new, unseen data.
.. code-block:: python

   import dmqclib as dm

   # Read a configuration file
   classification_config_file = "/path/to/classification_config.yaml"
   classification_name = "classification_0001"
   classification_config = dm.read_config(classification_config_file, module="classify")
   classification_config.select(classification_name)

   # Run classification
   dm.classify_dataset(classify_config)

Next Steps
----------

In the next pages, you will learn how to install the library and set up how to set up configuration files for successful model building!

Proceed to the next tutorial: :doc:`./installation`.
