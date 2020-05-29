#!/bin/bash
echo "getting p2j"
pip install -U --quiet git+https://github.com/pSpitzner/python2jupyter.git > /dev/null

it_dir="$(cd "$(dirname "$0")"; pwd -P)"
py_dir=$(dirname "$it_dir")

cd $it_dir
echo "getting latest repo"
git pull > /dev/null

convert() {
    ipynb=$(basename $1 ".py")".ipynb"
    p2j -o $1 -t $(basename $1 ".py")".ipynb"
}

for filename in $py_dir/*.py; do
    convert $filename &
done
wait

echo "all converted"
