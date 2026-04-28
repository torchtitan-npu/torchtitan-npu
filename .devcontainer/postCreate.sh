source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip install -r requirements.txt
pip install -r requirements_dev.txt

pytest
