from Flask_Example import app
from flask import render_template
from ..utilities.loadjson import load_json
from ..utilities.load_wow_api import get_enhanced_char


NAV_HISTORY = []

def add_nav_history(realm,charName):
    global NAV_HISTORY
    
    for x in NAV_HISTORY:
        if x['charName'] == charName:
            if x['realm'] == realm:
                return None

    NAV_HISTORY.append({'realm':realm,'charName':charName})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/char')
def defaultChar():
    data =(load_json('grymstone.json'))
    return render_template('char.html',data=data)
 
@app.route('/char/<realm>/<charName>')
def getChar(charName,realm):
    global NAV_HISTORY

    data =(get_enhanced_char(charName,realm))
    add_nav_history(realm,charName)
    # return data
    # return realm
    return render_template('char.html',data=data,nav_history=NAV_HISTORY)

