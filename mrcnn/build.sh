#!/bin/bash

# this needs to point to virtual env where build is done
virtualenv --python python3 venv
source venv/bin/activate

cd "$(dirname "$0")"

pip install --upgrade pip
pybuilderInstalled=`pip freeze | grep 'pybuilder' | wc -l`

if [ $pybuilderInstalled != 1 ]
then
   echo "Installing pybuilder"
   pip install pybuilder
fi

pyb install_dependencies clean publish
pyb sphinx_generate_documentation

rm -rf venv/

if [ ! -d "bin" ]; then
  mkdir 'bin'
fi

cp target/dist/mrcnn*/dist/* bin/

#rm -rf target/