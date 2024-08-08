# Deep_RL_RHT

E. Ortiz-Mansilla, J.J. García-Esteban, J. Bravo-Abad, J.C. Cuevas, "Deep Reinforcement Learning for Radiative Heat Transfer Optimization Problems", submitted for publication (2024).

## Repository folders and files

The repository is structured into the following directories:

- `/data`: required/generated data.
- `/img`: generated images by the code.
- `/notebook`: notebooks with relevant code of *policy-based and combined algorithms*.
- `/script`: `.sh` scripts.
- `/src`: source code of the repository with *value-based algorithms*.
- `/workflows`: set of workflows via Github actions.

In addition, there are some aditional relevant files:

- `env.yml`: define the environment and dependencies to install with conda.
- `requirements.txt`: specific versions of all the libraries used in this repo.
- `setup.py`: soft requirements of the environment. Where we specifiy the libraries to be installed.

More specifically, relevant code is available `/notebook` folder:

- `REINFORCE.ipynb`: Relevant code for the REINFORCE algorithm. Code is ready to be executed.
- `A2C_PPO_Optuna.ipynb`: Relevant code for the A2C and PPO algorithms, also including the Optuna hyperparameter search algorithm. Code is ready to be executed.
- `SARSA.ipynb`: A representative example about the execution of SARSA algorithm. It's suggested to follow it in order to execute `src/RL_simulations/RHT_simulation_RUNS.py`.
- `double_DQN.ipynb`: A representative example about the execution of Double DQN algorithm. It's suggested to follow it in order to execute `src/RL_simulations/RHT_simulation_RUNS.py`.
- `random.ipynb`: Relevant code for the random algorithm. Code is ready to be executed.

## Environment basics

We can create an isolated environment with required Python version. This can be done with conda or any other number of tools like venv.

Instructions to install **miniconda** in order to create an environment:

- [Get conda](https://docs.conda.io/en/latest/miniconda.html)
- [Install conda](https://engineeringfordatascience.com/posts/install_miniconda_from_the_command_line/)

## Recreate this environment

1. Clone this repository and enter in the generated folder.

2. Create and activate a new environment with `env.yml`(`--force` removes the previous environment with the same name):

   ```{bash}
   conda env create -f env.yml
   conda activate rl_rht_py3.9
   ```

3. Select the new environment. If you use VSCode:
    - In a notebook, select the created environment at the top right corner.
    - In a script, select the created environment at the bottom right corner.
    For more information, refer to the [documentation.](https://code.visualstudio.com/docs/python/environments#_work-with-python-interpreters)

4. If custom submodules are not found, set `PYTHONPATH` environment variable typing on terminal:

    ```{bash}
    echo $PYTHONPATH
    export PYTHONPATH="/home/my_user/code_path"
    echo $PYTHONPATH
    ```

    Or add ir permanently to the end of your `~/.bashrc` file (at ubuntu, `/home/user/.bashrc`):

    ```{bash}
    export PYTHONPATH="/home/my_user/code_path"
    ```

    For more info, you can refer to this [tutorial.](https://www.simplilearn.com/tutorials/python-tutorial/python-path)

## Requirements setup

To install required dependencies:

```{bash}
pip install -r requirements.txt
```

`pip-tools` also has a `pip-sync` command to make sure that the local environment is in sync with the `requirements.txt` file.
After `pip-sync`, we must install the packages in editable mode:

```{bash}
pip install -e .[dev]
```

If, after adding new dependencies, custom submodules are not found, [set again `PYTHONPATH` environment variable.](#recreate-this-environment)

## Acknowledgements

### Our thesis directors

- Juan Carlos Cuevas Rodríguez
- Jorge Bravo-Abad

[Juan Carlos page](http://webs.ftmc.uam.es/juancarlos.cuevas/)
[Jorge's page](http://webs.ftmc.uam.es/jorge.bravo/team.html)

### Komorebi's team

[Komorebi's page](https://komorebi.ai/es/)

## Additional info

### Personal info

If you find some problems or you have anything to discuss about the code, feel free to start a discussion or please reach us through our e-mails: <eva.ortizm@estudiante.uam.es> <juanjose.garciae@uam.es>.

### Template

Template for Python libraries by Komorebi-AI

### Fonts

If employed fonts are not available in your PC, you can get them through conda-forge:

```{bash}
conda activate env_name_py3.9 
conda install -c conda-forge -y mscorefonts
```

Or, alternatively, [you can install them in your PC](https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts/49884009#49884009):

```{bash}
sudo apt install msttcorefonts -qq
rm ~/.cache/matplotlib -rf           # remove cache
```

This last method will install `msttcorefonts` in path `/usr/share/fonts/truetype/msttcorefonts`.
