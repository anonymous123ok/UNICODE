# UNICODE
This repository provides the official implementation of the paper [Paper Title] (published in [Venue]). We organize the code as follows:

📂 Code Structure

CODA/, ALERT/, CodeTAE/, CodeDenoise/
Implementation of each baseline method, with separate modules for different models and tasks.
code/
Core abstract classes and their instantiated implementations.
🗃 Data Availability

Due to the large volume of experimental data:

Currently, we only provide the CodeDenoise dataset for the code cloning task to support reproducibility.
Full datasets will be released via [cloud storage (e.g., Google Drive/Baidu Netdisk)] after the double-blind review process.
(A temporary access link will be updated here once available.)

<img src="./overview.png" alt="drawing" width="800">

To use our abstract and instantiate methods, you can run the following commands:You just need to replace the input and output paths in the code and replace the api_key of the larger model.
```shell
cd code;
python abstract.py
python normalization.py
```
