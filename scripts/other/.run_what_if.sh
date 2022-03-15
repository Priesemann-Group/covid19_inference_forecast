#!/bin/bash
#Change dir to this directory
this_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $this_DIR
# >>>  conda initialize >>>
. /data.nst/smohr/anaconda3/etc/profile.d/conda.sh
conda activate pymc3_new
# >>>  conda initialize >>>

which python
# Pull from git
git pull
# Run the two scripts in parrallel
#/home/smohr/anaconda3/bin/python3.7 what_if_scenarios_lockdown_dez.py
python what_if_scenarios_jan.py


#/home/smohr/anaconda3/bin/python3.7 plot_scenarios_dez.py
#/home/smohr/anaconda3/bin/python3.7 plot_scenarios_dez_2.py

# Commit figures
git add "./figures/*"
git commit -m "automatic figures update"
git pull
git push



