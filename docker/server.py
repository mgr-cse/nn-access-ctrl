from flask import Flask
from flask import request
from subprocess import check_output

data = []
with open('dataset/u4k-r4k-auth11k.sample', 'r') as f:
    for line in f.readlines():
        data.append(line.strip().split())
    print('+++ data read')

app = Flask(__name__)

@app.route('/list_files', methods=['GET'])
def list_files():
    try:
        user_id = int(request.args.get('user_id'))
    except:
        return {
            "status": "failure",
            "reason": "can't parse user id"
        }
    
    files = []
    for d in data:
        if user_id == int(d[0]):
            files.append(d[1])
    
    return {
        "status": "success",
        "files": files
    }

@app.route('/list_users', methods=['GET'])
def list_users():
    users = set()
    for d in data:
        users.add(d[0])
    
    users = list(users)
    return {
        "status": "success",
        "users": users
    }

@app.route('/access_file', methods=['GET'])
def access_file():
    try:
        user_id = int(request.args.get('user_id'))
        resr_id = int(request.args.get('resr_id'))
        op = request.args.get('op')
        
        if op is None:
            raise Exception("Operation not specified")
    except:
        return {
            "status": "failure",
            "reason": "can't parse request parameters" 
        }
    
    # invoke decision engine
    output = check_output(['python', 'decision_engine.py', '--uid', str(user_id), '--rid', str(resr_id), '--operation', op])
    output = output.decode(encoding='utf-8')

    if output.find('granted') != -1:
        return {
            "status": "success",
            "result": "granted"
        }
    
    return {
        "status": "success",
        "result": "denied"
    }

@app.route('/')
def index():
    return '''
    Welcome to the decision engine server of the DEEP learning based Access control framework
    To access the API of this server,
    1) clone the repsitory: https://github.com/mgr-cse/nn-access-ctrl
    2) run the client python script docker/client.py
    '''

app.run(host='0.0.0.0', port=80)