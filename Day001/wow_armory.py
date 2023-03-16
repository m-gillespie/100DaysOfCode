import cred
import requests
from requests.auth import HTTPBasicAuth
import pprint as pp

BASEURL = 'https://us.api.blizzard.com/'
ACCESSTOKEN = None

def get_access_token():
    url= 'https://oauth.battle.net/token'
    data = {'grant_type':'client_credentials'}
    token = requests.post(url=url,data=data,auth=HTTPBasicAuth(cred.CLIENT,cred.SECRET))
    if(token.status_code == 200):

            return token.json()
    else:
        return None
def get_appearance(charName,realmName):
    url = BASEURL + 'profile/wow/character/'+realmName+'/'+charName+'/appearance?namespace=profile-us&locale=en_US&access_token='+ACCESSTOKEN
    print(url)
    r = requests.get(url)
    if(r.status_code == 200):
         return r.json()
    else:
         return None

def get_appearance(charName,realmName):
    url = BASEURL + 'profile/wow/character/'+realmName+'/'+charName+'/appearance?namespace=profile-us&locale=en_US&access_token='+ACCESSTOKEN
    print(url)
    r = requests.get(url)
    if(r.status_code == 200):
         return r.json()
    else:
         return None
def get_keystone_profile_index(charName,realName):
    url = BASEURL + 'profile/wow/character/'+realmName+'/'+charName+'/mythic-keystone-profile?namespace=profile-us&locale=en_US&access_token='+ACCESSTOKEN
    r = requests.get(url)
    if(r.status_code == 200):
         return r.json()
    else:
         return None

if __name__ =='__main__':
    charName = 'clohelpyo'
    realmName = 'daggerspine'
    ACCESSTOKEN = get_access_token()['access_token']
    pp.pprint(get_keystone_profile_index(charName,realmName))
