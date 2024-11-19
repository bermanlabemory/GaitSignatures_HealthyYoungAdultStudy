# GaitSignatures_HealthyYoungAdultStudy
This repository contains the code and data for our follow up gait signatures study on a cohort of 17 healthy young adults walking on a treadmill across a wide range of walking speeds

Python: Generate Gait Signatures

All associated files can be found in the ‘Python RNN training code’ folder


**Please ensure that you are using version 2.15.0 of tensorflow and keras**

  !pip install keras==2.15.0
  tensorflow==2.15.0


1.	Ensure all files in the ‘input data’  and ‘helper functions’ folders are present in your Python workspace.

2.	Generate Gait signatures for each data type by running the following scripts: 
a)	HYA_Study_GaitSignatures_2Dkinematics_scaled_Regularization_dropoutandL2_Adamlearningdecrease_LB499_codeshare.ipynb
b)	HYA_Study_GaitSignatures_3Dkinematics_scaled_Regularization_dropoutandL2_Adamlearningdecrease_LB499_codeshare.ipynb
c)	HYA_Study_GaitSignatures_kinetics_scaled_Regularization_dropoutandL2_Adamlearningdecrease_LB499_codeshare.ipynb
d)	HYA_Study_GaitSignatures_alldata_scaled_Regularization_dropoutandL2_Adamlearningdecrease_LB499_codeshare.ipynb

3.	Generate phase averaged data of the original data features by executing the following script: 
a)	HYA_Study_GaitSignatures_Phaseaveraged_alldata_noPCA_noScaling_codeshare.ipynb
4.	Get variance explained in the original data by each of the principal components of the gait signatures by executing the following script: 
a)	HYA_Study_GaitSignatures_ReconstructData_IsolatedPCs_VarianceExplained_codeshare.ipynb
As in our previous study (Winner et al., 2023), we use a collection of mathematical tools used in the Biologically Inspired Robotics and Dynamical Systems (BIRDS) lab at the University of Michigan. Most of these tools were either coded or invented to a large degree by the lab's founder, Shai Revzen found in the ‘helper functions’ folder

MATLAB: Analyses and Figures

All associated files can be found in the ‘MATLAB Analysis and Figures Code’ folder

1.	Ensure all subfolders of the folder ‘MATLAB Analysis and Figures Code’ are loaded into the MATLAB workspace.
2.	To generate all the figures from the manuscript, execute the script: ‘HYA_GaitSignatures_FigureGeneration.m’


References

Winner TS, Rosenberg MC, Jain K, Kesar TM, Ting LH, Berman GJ. Discovering individual-specific gait signatures from data-driven models of neuromechanical dynamics. PLoS Computational Biol. 2023 Oct;19(10):e1011556.
