# Deep_RL_RHT

E. Ortiz-Mansilla, J.J. García-Esteban, J. Bravo-Abad, J.C. Cuevas, "Deep Reinforcement Learning for Radiative Heat Transfer Optimization Problems", submitted for publication (2024).

## Files in repository

- REINFORCE: Folder containing all the necessary files for the REINFORCE algorithm used in the work mentioned above. Code is ready to be executed, but require the other files in the folder.

    - 16layers_index.txt: Contains all the possible system configurations for the 16 layers case studied in the paper.
    - 16layers_data.txt: Contains the corresponding values of the Radiative Heat Transfer coefficient for the configurations in the file above.
    - REINFORCE.ipynb: Main file in the folder, the notebook contains all the relevant code for the REINFORCE algorithm.

- A2C_PPO_Optuna: Folder containing all the necessary files for the A2C and PPO algorithms used in the work mentioned above, also including the Optuna hyperparameter search algorithm used. Code is ready to be executed, but require the other files in the folder.

    - 16layers_index.txt: Contains all the possible system configurations for the 16 layers case studied in the paper.
    - 16layers_data.txt: Contains the corresponding values of the Radiative Heat Transfer coefficient for the configurations in the file above.
    - A2C_PPO_Optuna.ipynb: Main file in the folder, the notebook contains all the relevant code for the A2C and PPO algorithms, also including the Optuna hyperparameter search algorithm.
