#!/bin/bash
# @Author: Sebastian B. Mohr
# @Date:   2020-05-29 12:38:21
# @Last Modified by:   Sebastian
# @Last Modified time: 2020-05-30 10:28:41

#Change dir to this directory
this_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $this_DIR

which python
# Pull from git
git pull
# Run the two scripts
/home/smohr/anaconda3/bin/python weekly_changepoints.py
/home/smohr/anaconda3/bin/python weekly_changepoints_with_RKI_Nowcast.py
# Commit figures
git add "../../figures/*"
git commit -m "automatic figures update"
git push



