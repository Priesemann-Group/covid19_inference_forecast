#Change dir to this directory
this_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $this_DIR
# >>>  conda initialize >>>
. /home/smohr/anaconda3/etc/profile.d/conda.sh
conda activate base
# >>>  conda initialize >>>

which python
# Pull from git
git pull
# Run the two scripts in parrallel
#/home/smohr/anaconda3/bin/python3.7 what_if_scenarios_lockdown_dez.py
/home/smohr/anaconda3/bin/python3.7 what_if_scenarios_lockdown_dez_2.py


/home/smohr/anaconda3/bin/python3.7 plot_scenarios_dez.py
/home/smohr/anaconda3/bin/python3.7 plot_scenarios_dez_2.py

# Commit figures
git add "./figures/*"
git commit -m "automatic figures update"
git pull
git push



