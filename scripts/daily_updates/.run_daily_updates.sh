# @Author: Sebastian B. Mohr
# @Date:   2020-05-29 12:38:21
# @Last Modified by:   Sebastian
# @Last Modified time: 2020-05-30 10:28:41

# Pull from git
git pull
# Run the two scripts
python weekly_changepoints.py
python what_if_scenarios.py
# Commit figures
git add "../../figures/*"
git commit -m "automatic figures update"
git push



