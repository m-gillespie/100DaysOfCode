from Flask_Example import app
from flask import render_template
from ..utilities.loadjson import load_json



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/char')
def defaultChar():
    data =(load_json('grymstone.json'))
    return render_template('char.html',data=data)

@app.route('/char/<charName>')
def getChar(charName):
    data =(load_json('{}.json'.format(charName)))
    return render_template('char.html',data=data)

