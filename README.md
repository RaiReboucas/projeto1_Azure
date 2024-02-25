# projeto1_Azure
Um projeto feito no Azure, o aluguel de bicicletas.
Aprendizado de máquina automatizado para previsão de aluguel de bicicletas, este projeto foi feito seguindo instruções do aprendizado de Machine Learning.

### arquivo .Json

{
    "runId": "mslearn-bike-automl",
    "runUuid": "018fd599-7560-4511-8d63-d8426ed04084",
    "parentRunUuid": null,
    "rootRunUuid": "018fd599-7560-4511-8d63-d8426ed04084",
    "target": "Serverless",
    "status": "Completed",
    "parentRunId": null,
    "dataContainerId": "dcid.mslearn-bike-automl",
    "createdTimeUtc": "2024-02-23T15:08:32.8845306+00:00",
    "startTimeUtc": "2024-02-23T15:09:06.984Z",
    "endTimeUtc": "2024-02-23T15:18:26.583Z",
    "error": null,
    "warnings": null,
    "tags": {
        "_aml_system_automl_mltable_data_json": "{\"Type\":\"MLTable\",\"TrainData\":{\"Uri\":\"azureml://locations/eastus2/workspaces/64c9e0c7-7392-42b6-a207-8403e8ca4bb2/data/Alugueldebicicletas/versions/1\",\"ResolvedUri\":null,\"AssetId\":null},\"TestData\":null,\"ValidData\":null}",
        "model_explain_run": "best_run",
        "_aml_system_automl_run_workspace_id": "64c9e0c7-7392-42b6-a207-8403e8ca4bb2",
        "_aml_system_azureml.automlComponent": "AutoML",
        "pipeline_id_000": "4bc4ec47eb8df2d5d68b361cd60120e65196f757;2dc95d8bafd84221b8de309021c722b4fa570e77;__AutoML_Ensemble__",
        "score_000": "0.09318736819366397;0.10551532801771185;0.08702496724483741",
        "predicted_cost_000": "0;0.5;0",
        "fit_time_000": "0.10127599999999999;0.156671;1",
        "training_percent_000": "100;100;100",
        "iteration_000": "0;1;2",
        "run_preprocessor_000": "MaxAbsScaler;MaxAbsScaler;",
        "run_algorithm_000": "XGBoostRegressor;ElasticNet;VotingEnsemble",
        "automl_best_child_run_id": "mslearn-bike-automl_2"
    },
    "properties": {
        "num_iterations": "3",
        "training_type": "TrainFull",
        "acquisition_function": "EI",
        "primary_metric": "normalized_root_mean_squared_error",
        "train_split": "0",
        "acquisition_parameter": "0",
        "num_cross_validation": "",
        "target": "Serverless",
        "AMLSettingsJsonString": "{\"is_subgraph_orchestration\":false,\"is_automode\":true,\"path\":\"./sample_projects/\",\"subscription_id\":\"f1be7878-c1bd-4fb1-a79f-f4b9a235eadc\",\"resource_group\":\"Aula1Azure\",\"workspace_name\":\"LabIa900\",\"iterations\":3,\"primary_metric\":\"normalized_root_mean_squared_error\",\"task_type\":\"regression\",\"IsImageTask\":false,\"IsTextDNNTask\":false,\"validation_size\":0.1,\"n_cross_validations\":null,\"preprocess\":true,\"is_timeseries\":false,\"time_column_name\":null,\"grain_column_names\":null,\"max_cores_per_iteration\":-1,\"max_concurrent_iterations\":3,\"max_nodes\":3,\"iteration_timeout_minutes\":15,\"enforce_time_on_windows\":false,\"experiment_timeout_minutes\":15,\"exit_score\":\"NaN\",\"experiment_exit_score\":0.085,\"whitelist_models\":null,\"blacklist_models\":[\"RandomForest\",\"LightGBM\"],\"blacklist_algos\":[\"RandomForest\",\"LightGBM\",\"TensorFlowDNN\",\"TensorFlowLinearRegressor\"],\"auto_blacklist\":false,\"blacklist_samples_reached\":false,\"exclude_nan_labels\":false,\"verbosity\":20,\"model_explainability\":false,\"enable_onnx_compatible_models\":false,\"enable_feature_sweeping\":false,\"send_telemetry\":true,\"enable_early_stopping\":true,\"early_stopping_n_iters\":20,\"distributed_dnn_max_node_check\":false,\"enable_distributed_featurization\":true,\"enable_distributed_dnn_training\":true,\"enable_distributed_dnn_training_ort_ds\":false,\"ensemble_iterations\":3,\"enable_tf\":false,\"enable_cache\":false,\"enable_subsampling\":false,\"metric_operation\":\"minimize\",\"enable_streaming\":false,\"use_incremental_learning_override\":false,\"force_streaming\":false,\"enable_dnn\":false,\"is_gpu_tmp\":false,\"enable_run_restructure\":false,\"featurization\":\"auto\",\"vm_type\":\"Standard_DS3_v2\",\"vm_priority\":\"dedicated\",\"label_column_name\":\"rentals\",\"weight_column_name\":null,\"miro_flight\":\"default\",\"many_models\":false,\"many_models_process_count_per_node\":0,\"automl_many_models_scenario\":null,\"enable_batch_run\":true,\"save_mlflow\":true,\"track_child_runs\":true,\"test_include_predictions_only\":false,\"enable_mltable_quick_profile\":\"True\",\"has_multiple_series\":false,\"_enable_future_regressors\":false,\"enable_ensembling\":true,\"enable_stack_ensembling\":false,\"ensemble_download_models_timeout_sec\":300.0,\"stack_meta_learner_train_percentage\":0.2}",
        "DataPrepJsonString": null,
        "EnableSubsampling": "False",
        "runTemplate": "AutoML",
        "azureml.runsource": "automl",
        "_aml_internal_automl_best_rai": "False",
        "ClientType": "Mfe",
        "_aml_system_scenario_identification": "Remote.Parent",
        "PlatformVersion": "DPV2",
        "environment_cpu_name": "AzureML-AutoML",
        "environment_cpu_label": "prod",
        "environment_gpu_name": "AzureML-AutoML-GPU",
        "environment_gpu_label": "prod",
        "root_attribution": "automl",
        "attribution": "AutoML",
        "Orchestrator": "AutoML",
        "CancelUri": "https://eastus2.api.azureml.ms/jasmine/v1.0/subscriptions/f1be7878-c1bd-4fb1-a79f-f4b9a235eadc/resourceGroups/Aula1Azure/providers/Microsoft.MachineLearningServices/workspaces/LabIa900/experimentids/5a025935-b431-4fa1-af57-2ab8ad5fb332/cancel/mslearn-bike-automl",
        "mltable_data_json": "{\"Type\":\"MLTable\",\"TrainData\":{\"Uri\":\"azureml://locations/eastus2/workspaces/64c9e0c7-7392-42b6-a207-8403e8ca4bb2/data/Alugueldebicicletas/versions/1\",\"ResolvedUri\":\"azureml://locations/eastus2/workspaces/64c9e0c7-7392-42b6-a207-8403e8ca4bb2/data/Alugueldebicicletas/versions/1\",\"AssetId\":\"azureml://locations/eastus2/workspaces/64c9e0c7-7392-42b6-a207-8403e8ca4bb2/data/Alugueldebicicletas/versions/1\"},\"TestData\":null,\"ValidData\":null}",
        "ClientSdkVersion": null,
        "snapshotId": "00000000-0000-0000-0000-000000000000",
        "SetupRunId": "mslearn-bike-automl_setup",
        "SetupRunContainerId": "dcid.mslearn-bike-automl_setup",
        "FeaturizationRunJsonPath": "featurizer_container.json",
        "FeaturizationRunId": "mslearn-bike-automl_featurize",
        "ProblemInfoJsonString": "{\"dataset_num_categorical\": 0, \"is_sparse\": true, \"subsampling\": false, \"has_extra_col\": true, \"dataset_classes\": 552, \"dataset_features\": 64, \"dataset_samples\": 657, \"single_frequency_class_detected\": false}"
    },
    "parameters": {},
    "services": {},
    "inputDatasets": null,
    "outputDatasets": [],
    "runDefinition": null,
    "logFiles": {},
    "jobCost": {
        "chargedCpuCoreSeconds": null,
        "chargedCpuMemoryMegabyteSeconds": null,
        "chargedGpuSeconds": null,
        "chargedNodeUtilizationSeconds": null
    },
    "revision": 13,
    "runTypeV2": {
        "orchestrator": "AutoML",
        "traits": [
            "automl",
            "Remote.Parent"
        ],
        "attribution": null,
        "computeType": null
    },
    "settings": {},
    "computeRequest": null,
    "compute": {
        "target": "Serverless",
        "targetType": "AmlCompute",
        "vmSize": "Standard_DS3_v2",
        "instanceType": "Standard_DS3_v2",
        "instanceCount": 1,
        "gpuCount": null,
        "priority": "Dedicated",
        "region": null,
        "armId": null,
        "properties": null
    },
    "createdBy": {
        "userObjectId": "ee3501df-52b5-4038-af72-eb5c3ab034e8",
        "userPuId": "10032003573955D4",
        "userIdp": "live.com",
        "userAltSecId": "1:live.com:0003BFFD6274903E",
        "userIss": "https://sts.windows.net/1aeb6847-2168-4d06-967b-73411e81191e/",
        "userTenantId": "1aeb6847-2168-4d06-967b-73411e81191e",
        "userName": "Carlos Rai Tomas",
        "upn": null
    },
    "computeDuration": "00:09:19.5983759",
    "effectiveStartTimeUtc": null,
    "runNumber": 1708700912,
    "rootRunId": "mslearn-bike-automl",
    "experimentId": "5a025935-b431-4fa1-af57-2ab8ad5fb332",
    "userId": "ee3501df-52b5-4038-af72-eb5c3ab034e8",
    "statusRevision": 3,
    "currentComputeTime": null,
    "lastStartTimeUtc": null,
    "lastModifiedBy": {
        "userObjectId": "ee3501df-52b5-4038-af72-eb5c3ab034e8",
        "userPuId": "10032003573955D4",
        "userIdp": "live.com",
        "userAltSecId": "1:live.com:0003BFFD6274903E",
        "userIss": "https://sts.windows.net/1aeb6847-2168-4d06-967b-73411e81191e/",
        "userTenantId": "1aeb6847-2168-4d06-967b-73411e81191e",
        "userName": "Carlos Rai Tomas",
        "upn": null
    },
    "lastModifiedUtc": "2024-02-23T15:18:26.2662027+00:00",
    "duration": "00:09:19.5983759",
    "inputs": {
        "training_data": {
            "assetId": "azureml://locations/eastus2/workspaces/64c9e0c7-7392-42b6-a207-8403e8ca4bb2/data/Alugueldebicicletas/versions/1",
            "type": "MLTable"
        }
    },
    "outputs": {
        "best_model": {
            "assetId": "azureml://locations/eastus2/workspaces/64c9e0c7-7392-42b6-a207-8403e8ca4bb2/models/azureml_mslearn-bike-automl_2_output_mlflow_log_model_213552224/versions/1",
            "type": "MLFlowModel"
        }
    },
    "currentAttemptId": 1
}
