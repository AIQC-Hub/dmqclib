Step 2: Dataset Preparation
===========================

The dataset preparation workflow is the first crucial step in the ``dmqclib`` pipeline. It takes raw data, extracts features, and creates balanced training, validation, and test sets for your machine learning model.

This process is driven by a YAML configuration file, ensuring your data preparation is repeatable and easy to manage.

.. admonition:: A Note on Running the Examples

   The examples in these tutorials are presented as complete Python scripts (``.py`` files). This approach is great for building a reusable workflow.

   However, you are encouraged to use the method you are most comfortable with. The code can be run in several ways:

   *   **As Python Scripts:** Copy the code into a ``.py`` file and run it from your terminal with ``python your_script_name.py``.
   *   **In an Interactive Python Session:** Launch Python (``python``) or IPython (``ipython``) and paste the code line by line. This is great for quick tests.
   *   **In a Jupyter Notebook or Lab:** This is a fantastic option for experimentation, as it allows you to run code in cells, add notes, and visualize results interactively.

   Feel free to adapt the examples to your preferred environment.

Getting the Example Data
------------------------

This tutorial uses the Copernicus Marine NRT CTD dataset from Kaggle. We provide two methods for downloading it.

First, create the directories for your project, regardless of which download method you choose:

.. code-block:: bash

   # Create a main project directory
   mkdir -p ~/aiqc_project

   # Create subdirectories for input and output data
   mkdir -p ~/aiqc_project/input
   mkdir -p ~/aiqc_project/data

Now, choose one of the following options to download the data.

.. tabs::

   .. tab:: Option 1: Kaggle API (Recommended)

      This method is ideal for reproducibility and for users who frequently work with Kaggle.

      1. **Install and configure the Kaggle API:**
         If you haven't already, install the client and set up your credentials.

         .. code-block:: bash

            pip install kaggle

         Follow the `Kaggle API authentication instructions <https://www.kaggle.com/docs/api#getting-started-installation-&-authentication>`_ to get your ``kaggle.json`` file.

      2. **Download and unzip the data:**
         This single command downloads and extracts the dataset directly into your ``input`` folder.

         .. code-block:: bash

            kaggle datasets download -d takaya88/copernicus-marine-nrt-ctd-data-for-aiqc -p ~/aiqc_project/input --unzip

   .. tab:: Option 2: cURL (Quickstart)

      This method is the fastest way to get the data, as it requires no extra tools or setup.

      1. **Download the zip file using cURL:**

         .. code-block:: bash

            curl -L -o ~/aiqc_project/input/data.zip \
              https://www.kaggle.com/api/v1/datasets/download/takaya88/copernicus-marine-nrt-ctd-data-for-aiqc

      2. **Unzip the file:**
         Navigate to the directory and extract the archive.

         .. code-block:: bash

            unzip ~/aiqc_project/input/data.zip -d ~/aiqc_project/input/

----------

After following either set of instructions, you should now have a file named ``nrt_cora_bo_4.parquet`` inside ``~/aiqc_project/input/``.

The Dataset Preparation Workflow
--------------------------------

The workflow consists of three main steps: generating a configuration template, customizing it, and running the preparation script.

Step 2.1: Generate the Configuration Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, create a Python script (e.g., ``run_prepare.py``) and use ``dmqclib`` to generate a configuration template. This file will contain all the necessary sections for the preparation task.

.. code-block:: python

   import dmqclib as dm

   # This creates 'prepare_config.yaml' in the current directory
   dm.write_config_template(
       file_name="prepare_config.yaml",
       module="prepare"
   )
   print("Configuration template 'prepare_config.yaml' has been created.")


Step 2.2: Customize the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, open the newly created ``prepare_config.yaml`` in a text editor. You need to tell ``dmqclib`` where to find your input data and where to save the processed output.

You will primarily edit two sections: ``path_info_sets`` and ``data_sets``.

Update your ``prepare_config.yaml`` to match the following, replacing the placeholder paths with the ones you created.

.. code-block:: yaml

   path_info_sets:
     - name: data_set_1
       common:
         # EDIT: Set this to your output directory
         base_path: /home/user/aiqc_project/data
       input:
         # EDIT: Set this to your input directory
         base_path: /home/user/aiqc_project/input
         step_folder_name: ""
       split:
         step_folder_name: training

.. code-block:: yaml

   data_sets:
     - name: NRT_BO_001
       dataset_folder_name: nrt_bo_001
       # EDIT: Ensure this matches your downloaded file name
       input_file_name: nrt_cora_bo_4.parquet

.. note::
   For a complete reference of all available configuration options, see the :doc:`../configuration/preparation` page.

Step 2.3: Run the Preparation Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, modify your Python script (``run_prepare.py``) to load the configuration and execute the ``create_training_dataset`` function.

.. code-block:: python

   import dmqclib as dm

   # Path to your customized configuration file
   config_file = "prepare_config.yaml"
   # This name must match the 'name' field in the 'data_sets' section of your YAML
   dataset_name = "NRT_BO_001"

   print(f"Loading configuration for '{dataset_name}' from '{config_file}'...")
   config = dm.read_config(config_file, module="prepare")
   config.select(dataset_name)

   print("Starting dataset preparation...")
   dm.create_training_dataset(config)
   print("Dataset preparation complete!")

Run the script from your terminal:

.. code-block:: bash

   python run_prepare.py

Understanding the Output
------------------------

After the script finishes, your output directory (e.g., ``~/aiqc_project/data``) will contain a new folder named ``nrt_bo_001`` (from ``dataset_folder_name``). Inside, you will find several subdirectories:

- **summary**: Contains summary statistics of the input data, used for normalization.
- **select**: Stores profiles identified as having bad observations (positive samples) and associated good profiles (negative samples).
- **locate**: Contains the specific observation records for both positive and negative profiles.
- **extract**: Holds the features extracted from the observation records.
- **training**: The final output, containing the split training, validation, and test datasets ready for model training.

Next Steps
----------

Congratulations! You have successfully prepared your dataset. You are now ready to train a model.

Proceed to the next tutorial: :doc:`./training`.
