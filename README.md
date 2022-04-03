Requirements:

    OS: Fedora 34
    Python: version 3.9.6
    Python packages:
        pandas==1.4.2
        numpy==1.22.3
        scikit_learn==0.24.0


These packages can be installed using pip.

The folder 'data' includes csv files for the running examples. 
The folder 'results' contains all decision rule histories mentioned in the paper.

To start the script in terminal: python continous_rule_mining.py. Per default, all use cases are started subsequently.

OR run using pipenv, which creates a virtual environment with all needed packages:

    python 3.9 needed
    install pipenv via pip
    run "pipenv install"
    run "pipenv run python continous_rule_mining.py"


