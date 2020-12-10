#Change dir to this directory
this_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $this_DIR
# >>>  conda initialize >>>
. /home/smohr/anaconda3/etc/profile.d/conda.sh
conda activate cc_env
# >>>  conda initialize >>>

which python
# Pull from git
git pull
# Run the two scripts in parrallel
python what_if_scenarios_lockdown_dez.py &> ~/logs/what_if_scenarios_lockdown_dez & P1=$!
python what_if_scenarios_lockdown_dez_2.py &> ~/logs/what_if_scenarios_lockdown_dez_2 & P2=$!

# Wait for scripts
wait $P1
wait $P2

# Plot new results
python plot_scenarios_dez.py
python plot_scenarios_dez_2.py

# Commit figures
git add "./figures/*"
git commit -m "automatic figures update"
git pull
git push



