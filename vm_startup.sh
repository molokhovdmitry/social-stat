# Script for an automatic startup on a virtual machine.
. /home/user/python_venv/social-stat/bin/activate
cd /home/user/social-stat
git pull
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000 > /home/user/log.txt 2>&1