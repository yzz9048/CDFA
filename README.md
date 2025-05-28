# The code is coming soon

# Datasets
For AIops18, please refer to https://github.com/BEbillionaireUSD/Maat

For GAIA, please refer to https://github.com/CloudWise-OpenSource/GAIA-DataSet

For SN in our paper, please refer to https://zenodo.org/records/15532565

# Details for SN
The json files in the data warehouse are the raw data collected by using tools, e.g., Prometheus and k6, and the csv files are extracted from json files.
Metrics are collected at intervals of 10 seconds. Therefore, RPS data can perform average aggregation at intervals of 10 seconds as guiding condition data. We have provided the specific number of requests per second. Readers can handle and generate other conditional data by themselves.

This dataset is collected from microservice benchmark SocialNetwork, deployed on three servers, each equipped with Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz, 4* 8G DDR4 Registered (Buffered) 2133 MHz, 2T HHD.
The workload script (mix-k6.js) is rewritten from the official document(https://github.com/delimitrou/DeathStarBench/blob/master/socialNetwork/wrk2/scripts/social-network/mixed-workload.lua)
