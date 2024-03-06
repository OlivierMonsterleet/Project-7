Project-7
Within P7 project, repo dedicated to API testing and release.

One single branch (main)

Description
In this Repo, you will find:

P7_api.py
Python script to launch an API including several path in order to get:

all data
single client data
single client proba
single client prediction

test_api.py
Unit testing cases: 1 per fonction defined in P7_API.py

Github/workflow/test.yml
-> allow automatic unit testings on push + automatic deployment in render

Installation
Through conda / .venv environment launch pip install -r requirements.txt

To launch API:
Locally: In terminal, run P7_API, API is displayed on http://127.0.0.1:5080

Production: https://dashboard.render.com/web/srv-cn2v09f109ks73ele2gg

TESTINGS:
For local testings, at root level, run pytest in terminal

While pushing to Github a new version, testings are performed automatically. If passed then auto deployment on 
https://dashboard.render.com/web/srv-cn2v09f109ks73ele2gg
