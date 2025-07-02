def get_data_set_config_schema():
    yaml_schema = """
---
type: object
properties:
  path_info_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        common:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - base_path
          additionalProperties: false
        input:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - base_path
            - step_folder_name
          additionalProperties: false
        select:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        locate:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        split:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
      required:
        - name
        - common
        - input
      additionalProperties: false

  target_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        variables:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              flag:
                type: string
            required:
              - name
              - flag
            additionalProperties: false
      required:
        - name
        - variables
      additionalProperties: false

  feature_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        features:
          type: array
          items:
            type: string
      required:
        - name
        - features
      additionalProperties: false

  feature_param_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        params:
          type: array
          items:
            type: object
            properties:
              feature:
                type: string
            required:
              - feature
            additionalProperties: true
      required:
        - name
        - params
      additionalProperties: false

  step_class_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        steps:
          type: object
          properties:
            input:
              type: string
            summary:
              type: string
            select:
              type: string
            locate:
              type: string
            extract:
              type: string
            split:
              type: string
          required:
            - input
            - summary
            - select
            - locate
            - extract
            - split
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  step_param_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        type:
          type: string
        steps:
          type: object
          properties:
            input:
              type: object
            summary:
              type: object
            select:
              type: object
            locate:
              type: object
            extract:
              type: object
            split:
              type: object
          required:
            - input
            - summary
            - select
            - locate
            - extract
            - split
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  data_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        dataset_folder_name:
          type: string
        input_file_name:
          type: string
        path_info:
          type: string
        target_set:
          type: string
        feature_set:
          type: string
        feature_param_set:
          type: string
        step_class_set:
          type: string
        step_param_set:
          type: string
      required:
        - name
        - dataset_folder_name
        - input_file_name
        - path_info
        - target_set
        - feature_set
        - feature_param_set
        - step_class_set
        - step_param_set
      additionalProperties: false

additionalProperties: false
required:
  - path_info_sets
  - target_sets
  - feature_sets
  - feature_param_sets
  - step_class_sets
  - step_param_sets
  - data_sets
"""
    return yaml_schema


def get_training_config_schema():
    yaml_schema = """
---
type: object
properties:
  path_info_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        common:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - base_path
          additionalProperties: false
        input:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          required:
            - step_folder_name
          additionalProperties: false
        validate:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
        build:
          type: object
          properties:
            base_path:
              type: string
            step_folder_name:
              type: string
          additionalProperties: false
      required:
        - name
        - common
        - input
      additionalProperties: false

  target_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        variables:
          type: array
          items:
            type: object
            properties:
              name:
                type: string
              flag:
                type: string
            required:
              - name
              - flag
            additionalProperties: false
      required:
        - name
        - variables
      additionalProperties: false

  step_class_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        steps:
          type: object
          properties:
            input:
              type: string
            validate:
              type: string
            model:
              type: string
            build:
              type: string
          required:
            - input
            - validate
            - model
            - build
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  step_param_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        type:
          type: string
        steps:
          type: object
          properties:
            input:
              type: object
            validate:
              type: object
            model:
              type: object
            build:
              type: object
          required:
            - input
            - validate
            - model
            - build
          additionalProperties: false
      required:
        - name
        - steps
      additionalProperties: false

  training_sets:
    type: array
    items:
      type: object
      properties:
        name:
          type: string
        dataset_folder_name:
          type: string
        path_info:
          type: string
        target_set:
          type: string
        step_class_set:
          type: string
        step_param_set:
          type: string
      required:
        - name
        - dataset_folder_name
        - path_info
        - target_set
        - step_class_set
        - step_param_set
      additionalProperties: false

additionalProperties: false
required:
  - path_info_sets
  - target_sets
  - step_class_sets
  - step_param_sets
  - training_sets
"""
    return yaml_schema
