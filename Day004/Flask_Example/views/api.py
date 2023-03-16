from Flask_Example import app
from flask import jsonify,request

default_error_json = {'Error':True}


@app.route('/api/<param1>/<param2>',methods = ['POST'])
def api(param1,param2):

    if request.headers.get('api-key'):
        random_object = {'p1':param1,'p2':param2}
    else:
        return default_error_json

    return jsonify(random_object)

@app.route('/api/<param1>')
def api2(param1):
    random_object = {'p1':param1,'Connection':request.headers['Connection']}
    return jsonify(random_object)