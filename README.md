# Installation

This project's python environment and dependencies are managed through `uv`. Activate the project's virtual environment and install all necessary dependencies by running the following script. Note that this has only been tested on MacOS.

`./start_env.sh && source .venv/bin/activate && python utils/install_cmdstan.py`

# Load Data

Project data was obtained from [USGS](https://waterdata.usgs.gov/nwis/uv?site_no=04294500&legacy=1). Select the max available date range and copy the outputs to `data/raw.txt`

# Run the Model

The model is executed by running the following script:

`python model.py`

Including a `--debug` flag generates diagnostic plots as well.

# Analysis

[Project Report](report.md)