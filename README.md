# Quantum_classifiers_BSQ_201

The *Quantum_Classifiers_BSQ_201* repository is a project by Karlheinz Fedorlensky Forvil, Ugo Massé and Louis-Félix Vigneux for the BSQ201 class at Sherbrooke University. It gives three quantum and two classical classifiers to use easily.

The paper describing the project and the performance of the quantum classification algorithms is called *technical_paper.pdf*. It also describes the goals of the project and the theory behind the algorithms.

The *kernel_method.py*, *vqc_method.py* and *qcnn_method.py* files giving the quantum kernel classifier, quantum variational classifier and quantum convolutional network classes, respectively, are available in the repository.  These classes names are *Quantum_Kernel_Classification*, *VQC_Solver* and *QCNN_Solver*. The tutorial on how to use those classes to classify datasets with their corresponding method is given in the *repository_tutorial.ipynb* notebook.

The dataset folder gives two CSV datasets and the corresponding readme describing them. The *HTRU_2* dataset is used to classify pulsars with eight feature vectors, and the *magic_gamma_telescope* dataset, with ten features, classifies high-energy gamma particles. The references for those datasets are available in their readme. 

The *requirements.txt* file needs to be installed with the following line: 
```
$ pip install -r requirements.txt
```

Depending on the environment, *pip* may need to be replaced with *pip3*. This installation ensures that all the packages needed to run this code are available in the user’s environment.

Finally, some of the utils folders contain multiple utility functions described in the tutorial notebook. The functions are separated in the *error_functions.py*, *quantum_ansatz.py*, *quantum_embeddings.py* and *utils.py* files. 


