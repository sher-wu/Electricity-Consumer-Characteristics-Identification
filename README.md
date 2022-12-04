# Electricity-Consumer-Characteristics-Identification

Unofficial re-implementation of [*Electricity Consumer Characteristics Identification: A Federated Learning Approach*](https://ieeexplore.ieee.org/abstract/document/9380668/), which is used to use federated learning to identify electricity consumers' socio-demographic characteristics based on their raw meter data. In addition, after viewing the distribution of the data, some improvements are done in data pre-processing. 

### (My) Environments

* Python 3.7
* Torch 1.11

### Usage

* The used dataset is Irish Commission for Energy Regulation dataset, which can be accessed by filling out the application form on their official website.
* Run `Data_Set_to_Feature_Set.py`, `Feature_Set_Extract.py`, `Generate_C_Set_and_F+C_Set.py` and `F+C_CutUp.py` in `Generate_Data_Set` in order.
* Run `PCA_Number_Test.py` to determine the number of components to be left.
* **Core**: Run `Privacy_Preserving_PCA.py` to perform PPPCA in server side.
* Run `Standardize.py`.
* **Core**: Run `main.py` with different value of parameter $Category$ for different electricity consumer characteristics identification model creating and testing.
