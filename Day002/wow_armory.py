import cred
import requests
from requests.auth import HTTPBasicAuth
import pprint as pp
import json

BASEURL = 'https://us.api.blizzard.com'
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
    url = BASEURL + '/profile/wow/character/'+realmName+'/'+charName+'/appearance?namespace=profile-us&locale=en_US&access_token='+ACCESSTOKEN
#     print(url)
    r = requests.get(url)
    if(r.status_code == 200):
         return r.json()
    else:
         return None
    

def get_item_media(item_id):
     url = '{BASEURL}/data/wow/media/item/{item_id}?namespace=static-us&locale=en_US&access_token={ACCESSTOKEN}'.format(BASEURL=BASEURL,item_id=item_id,ACCESSTOKEN=ACCESSTOKEN)

     r = requests.get(url)
     print(url)
     print(r.status_code)
     if(r.status_code == 200):
          # Need to finish cleaning up
          item =r.json()['assets']
          return item
     else:
          return None
               


def get_equipment(charName,realmName):
     url = BASEURL + '/profile/wow/character/'+realmName+'/'+charName+'/equipment?namespace=profile-us&locale=en_US&access_token='+ACCESSTOKEN
     r = requests.get(url)
     if(r.status_code == 200):
          
          cleanedItems = {}
          for item in r.json()['equipped_items']:
               
               cleanedItems[item['slot']['type']]={
                    'name':item['name'],
                    # media lookup?
                    'media':item['media'],
                    'iLvl':item['level'],
                    # should lookup transmog item media
                    'transmog':None if not item.get('transmog') else item['transmog'],
                    # Probably don't need durability
                    'durability':None if not item.get('durability') else item['durability']
               }
               

          return cleanedItems
     else:
          return None
    
def get_keystone_profile_index(charName,realName):
    url = BASEURL + '/profile/wow/character/'+realmName+'/'+charName+'/mythic-keystone-profile?namespace=profile-us&locale=en_US&access_token='+ACCESSTOKEN
    r = requests.get(url)
    if(r.status_code == 200):
         return r.json()
    else:
         return None

def save_json(file,dict):
     out_json = json.dumps(dict,indent=4)
     with open(file,"w") as outfile:
          outfile.write(out_json)

if __name__ =='__main__':
     charName = 'clopenzo'
     realmName = 'daggerspine'
     ACCESSTOKEN = get_access_token()['access_token']

     # equipment =get_equipment(charName,realmName)
     
     item_media = get_item_media('200251')
     print(item_media)
     # Save the json to make it easier to view.
     # save_json('equipment.json',equipment)

     # pp.pprint(equipment['equipped_items'])

     #I want to clean up the equipement, lots of things not needed here.


     # pp.pprint(equipment)
 
 



