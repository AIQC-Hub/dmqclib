Overview
========

Basic usage
------------

As long as configration files are properly set, it is very easy to run a three-stage workflow provided by ``dmqclib``.

1.  **Dataset Preparation**

.. code-block:: python

   import dmqclib as dm

   prepare_config = dm.read_config("/path/to/prepare_config.yaml")
   dm.create_training_dataset(prepare_config)

2.  **Training & Evaluation:**

.. code-block:: python

   training_config = dm.read_config("/path/to/training_config.yaml")
   dm.train_and_evaluate(training_config)

3.  **Classification:**

.. code-block:: python

   classification_config = dm.read_config("/path/to/classification_config.yaml")
   dm.classify_dataset(classify_config)

Next Steps
----------

In the next pages, you will learn how to install the library and set up how to set up configuration files for successful model building!

Proceed to the next tutorial: :doc:`./installation`.
