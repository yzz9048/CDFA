# The code is coming soon

# Datasets
For AIops18, please refer to https://github.com/BEbillionaireUSD/Maat

For GAIA, please refer to https://github.com/CloudWise-OpenSource/GAIA-DataSet

For SN in our paper, please refer to 

# Details for SN
There are three folders in SN: 

"data_in_paper" is the dataset that this paper used; 

Similar to "data_in_paper" , "Additional_sample" is an additional dataset provided for readers; 

"toy" is a small dataset, which can be used for toy experiment.

The json files in the data warehouse are the raw data collected by tools, i.e.., Prometheus and k6, and the csv files are extracted from json files. 
Metrics are collected at intervals of 10 seconds. Therefore, RPS data can perform average aggregation at intervals of 10 seconds as guiding condition data. We have provided the specific number of requests per second. Readers can handle and generate other conditional data by themselves. The failue injection tool used is ChaosBlade. And the main type of failue injected is cpu failure, using "blade create k8s container-cpu load --cpu-percent xx --container-ids xx --names xx --kubeconfig xx --namespace xx". The last column of the table that combines RPS and metrics is the failure label.

This dataset is collected from microservice benchmark SocialNetwork, which comprises 13 microservices and 13 database microservices, each microservice has four instances deployed on two worker servers, each equipped with Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz, 4* 8G DDR4 Registered (Buffered) 2133 MHz, 2T HHD. A master server (the same equipment as work server) generates workloads using k6, collects monitoring data via Prometheus and inject failure to microservices via ChaosBlade.

The workload script (mix-k6.js) is rewritten from the official document(https://github.com/delimitrou/DeathStarBench/blob/master/socialNetwork/wrk2/scripts/social-network/mixed-workload.lua).


