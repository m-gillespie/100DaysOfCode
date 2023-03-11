from .cred import creds
import requests
from requests.auth import HTTPBasicAuth
import pprint as pp
import json
import requests_cache
 
requests_cache.install_cache('Wow_Api',backend='sqlite',expires_after=180)

BASEURL = 'https://us.api.blizzard.com'
ACCESSTOKEN = None

def get_access_token():
    global ACCESSTOKEN

    url= 'https://oauth.battle.net/token'
    data = {'grant_type':'client_credentials'}
    token = requests.post(url=url,data=data,auth=HTTPBasicAuth(creds['CLIENT'],creds['SECRET']))

    if(token.status_code == 200):
            ACCESSTOKEN = token.json()['access_token']
    else:
        return None
def get_appearance(charName,realmName):
    global ACCESSTOKEN
    if ACCESSTOKEN == None:
        get_access_token()
    
    url = BASEURL + '/profile/wow/character/'+realmName+'/'+charName+'/appearance?namespace=profile-us&locale=en_US&access_token='+ACCESSTOKEN
#     print(url)
    r = requests.get(url)
    if(r.status_code == 200):
         return r.json()
    else:
         return None

def get_character_media(charName,realmName):
    global ACCESSTOKEN
    if ACCESSTOKEN == None:
        get_access_token()

    url = '{BASEURL}/profile/wow/character/{realmName}/{charName}/character-media?namespace=profile-us&locale=en_US&access_token={ACCESSTOKEN}'.format(BASEURL=BASEURL,realmName=realmName,charName=charName,ACCESSTOKEN=ACCESSTOKEN)
#     print(url)
    r = requests.get(url)
    if(r.status_code == 200):
          response=r.json()
          rObj={}
          for key in response['assets']:
               rObj[key['key']] = key['value']
          return rObj
    else:
         return None

def get_profile(charName,realmName):
    global ACCESSTOKEN
    if ACCESSTOKEN == None:
        get_access_token()

    url = '{BASEURL}/profile/wow/character/{realmName}/{charName}?namespace=profile-us&locale=en_US&access_token={ACCESSTOKEN}'.format(BASEURL=BASEURL,realmName=realmName,charName=charName,ACCESSTOKEN=ACCESSTOKEN)
#     print(url)
    r = requests.get(url)
    if(r.status_code == 200):
          response=r.json()
          rObj = {
               'name':response['name'],
               'gender':response['gender']['name'],
               'faction':response['faction']['name'],
               'race':response['race']['name'],
               'class':response['character_class']['name'],
               'spec':response['active_spec']['name'],
               'realm':response['realm']['name'],
               'guild':None if not response.get('guild') else response['guild']['name'],
               'level':response['level'],
               'achievement_points':response['achievement_points'],
               'last_login':response['last_login_timestamp'],
               'equipped_ilvl':response['average_item_level'],
               'title':None if not response.get('title') else {'name':response['active_title']['name'],'format':response['active_title']['display_string']},
          }
          return rObj
    else:
         return None
    
def get_item_media(item_id):
     url = '{BASEURL}/data/wow/media/item/{item_id}?namespace=static-us&locale=en_US&access_token={ACCESSTOKEN}'.format(BASEURL=BASEURL,item_id=item_id,ACCESSTOKEN=ACCESSTOKEN)

     r = requests.get(url)
     # print(url)
     # print(r.status_code)
     if(r.status_code == 200):
          # Need to finish cleaning up
          item =r.json()['assets'][0]
          return item
     else:
          return None
               


def get_equipment(charName,realmName):
     global ACCESSTOKEN
     if ACCESSTOKEN == None:
         get_access_token()

     url = BASEURL + '/profile/wow/character/'+realmName+'/'+charName+'/equipment?namespace=profile-us&locale=en_US&access_token='+ACCESSTOKEN
     r = requests.get(url)
     if(r.status_code == 200):
          
          cleanedItems = {}
          for item in r.json()['equipped_items']:
               mediaObj = get_item_media(item['media']['id'])     
               transmog = None
               if item.get('transmog'):
                    transmogMedia = get_item_media(item['transmog']['item']['id'])
                    transmog = {
                         'itemId':item['transmog']['item']['id'],
                         'name':item['transmog']['item']['name'],
                         'media':{
                              'mediaType':transmogMedia['key'],
                              'mediaurl':transmogMedia['value'],
                         }
                    }
               


               cleanedItems[item['slot']['type']]={
                    'itemId':item['item']['id'],
                    'name':item['name'],
                    # media lookup?
                    'media':{
                         'mediaType':mediaObj['key'],
                         'mediaURL':mediaObj['value'],
                    },
                    'fileId':mediaObj['file_data_id'],
                    'iLvl':item['level'],
                    # should lookup transmog item media
                    'transmog':transmog,
                    # Probably don't need durability
                    'durability':None if not item.get('durability') else item['durability']
               }
               

          return cleanedItems
     else:
          return None
    
def get_keystone_profile_index(charName,realName):
    global ACCESSTOKEN
    if ACCESSTOKEN == None:
        get_access_token()
    url = BASEURL + '/profile/wow/character/'+realmName+'/'+charName+'/mythic-keystone-profile?namespace=profile-us&locale=en_US&access_token='+ACCESSTOKEN
    r = requests.get(url)
    if(r.status_code == 200):
         return r.json()
    else:
         return None

def get_enhanced_char(charName,realmName,optionList=['profile','equipment','media']):
     global ACCESSTOKEN
     if ACCESSTOKEN == None:
        get_access_token()
     validOptions = ['profile','equipment','media']

     rObj ={}

     lowerOptionList=[x.lower() for x in optionList]
     trimmedList = [x for x in lowerOptionList if x in validOptions]

     if len(trimmedList)==0:
          trimmedList = ['profile','equipment','media']
     


     if('profile' in trimmedList):
          profile =get_profile(charName,realmName)
          rObj['profile']=profile
     if('equipment' in trimmedList):
          equipment =get_equipment(charName,realmName)
          rObj['equipment']=equipment
          # check all equipment and put in blank placeholders.should be able to do this differently when switched to front end rendering.
          slots = ['HEAD','NECK','SHOULDER','BACK','CHEST','SHIRT','TABARD','WRIST','WAIST','HANDS','TRINKET_1','OFF_HAND','LEGS','FINGER_1','TRINKET_2','FEET','FINGER_2','MAIN_HAND']
          pholder={'media':None,'iLvl':None}
          for slot in slots:
               if not rObj['equipment'].get(slot):
                    rObj['equipment'][slot]=pholder


                              # 'durability':None if not item.get('durability') else item['durability']


     
     if('media' in trimmedList):
          media =get_character_media(charName,realmName)
          rObj['media']=media


     



     return rObj



def save_json(file,dict):
     out_json = json.dumps(dict,indent=4)
     with open(file,"w") as outfile:
          outfile.write(out_json)

if __name__ =='__main__':
     charName = 'clohelpyo'
     realmName = 'daggerspine'
    #  ACCESSTOKEN = get_access_token()['access_token']

     # apperance = get_appearance(charName,realmName)
     # pp.pprint(apperance)

     # profile = get_profile(charName,realmName)
     # pp.pprint(profile)

     # character_media = get_character_media(charName,realmName)
     # pp.pprint(character_media)

     # equipment =get_equipment(charName,realmName)
     # pp.pprint(equipment)

     pp.pprint(get_enhanced_char(charName,realmName))


     # save_json('profile.json',profile)
 



