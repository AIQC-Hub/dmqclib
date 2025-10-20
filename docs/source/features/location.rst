Feature: Location
===========================

`Location` is a profile level feature that represents locations of sampling points. All observations belong to the same profile generally have the same `location` feature values. Even though ``dmqclib`` require both ``latitude`` and ``longitude`` values, any other columns in the input dataset can be specified as the `Location` feature.

Configuration
-------------------------------------

To include the `location` feature in your training and classification datasets, the value `location` needs to be specified in the `feature_sets` section.

.. code-block:: yaml

   feature_sets:
     - name: feature_set_1
       features:
         - location

Configuration: Parameters
-------------------------------------

.. code-block:: yaml

   feature_param_sets:
     - name: feature_set_1_param_set_3
       params:
         - feature: location
           stats_set: { type: min_max, name: location }
           col_names: [ longitude, latitude ]

Configuration: Normalization
---------------------------

You need to update the stats values in the configuration files based on the results from `dm.get_summary_stats` and `dm.format_summary_stats`.

.. code-block:: yaml

   feature_stats_sets:
     - name: feature_set_1_stats_set_1
       min_max:
         - name: location
           stats: { longitude: { min: 14.5, max: 23.5 },
                    latitude: { min: 55, max: 66 } }
