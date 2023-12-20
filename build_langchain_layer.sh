set -eo

LAYER_NAME=langchain_layer.zip
CURRENT_PATH=`pwd`

mkdir -p package/python

# langchain layer - run in linux container to be ccompatible with lambda
echo "Create Virtual env"
python3 -m venv .venv 
source .venv/bin/activate
echo "Activated Virtual Env. Installing requirements"
pip install -r ${CURRENT_PATH}/src/bot_dispatcher/requirements.txt --target ./package/python
cd package
echo "Zipping into lambda layer"
zip -v -r9 $LAYER_NAME .
mv $LAYER_NAME ${CURRENT_PATH}/build
cd ${CURRENT_PATH}
rm -rf package/
rm -rf .venv/
deactivate
