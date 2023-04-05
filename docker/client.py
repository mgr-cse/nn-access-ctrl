import requests
from typing import Dict

SERVER_IP = '172.17.0.2'    # docker's default ip, can change

def make_request(endpoint: str, params: Dict):
    try:
        res = requests.get(f'http://{SERVER_IP}{endpoint}', params=params)
        if not res.ok:
            raise Exception('Invalid response code received')
        data = res.json()
        return data
    except:
        return {
            "status": "failure",
            "reason": "exception occured while making a request to server"
        }

if __name__ == '__main__':
    while True:
        cmd = input(">> ").split()
        
        if not cmd:
            continue
        
        elif cmd[0] == 'list_users':
            recv = make_request(f'/{cmd[0]}', {})
            if recv['status'] == 'success':
                for u in recv['users']:
                    print(u)
            else:
                print(recv['reason'])
        
        elif cmd[0] == 'list_files':
            recv = make_request(f'/{cmd[0]}', {'user_id':int(cmd[1])})
            if recv['status'] == 'success':
                for u in recv['files']:
                    print(u)
            else:
                print(recv['reason'])
        
        elif cmd[0] == 'access_file':
            recv = make_request(f'/{cmd[0]}', {'user_id':int(cmd[1]), 'resr_id': int(cmd[2]), 'op':cmd[3]})
            if recv['status'] == 'success':
                print(recv['result'])
            else:
                print(recv['reason'])
        
        else:
            print('Wrong command')