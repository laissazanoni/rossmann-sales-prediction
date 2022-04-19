import os
import pandas as pd
import json
import requests


from flask import Flask, request, Response

# parameters
token = TOKEN

# # info about the boot
# https://api.telegram.org/bot[token]/getMe

# # get updates
# https://api.telegram.org/bot[token]/getUpdates

# # webhook
# https://api.telegram.org/bot[token]/setWebhook?url=https://6073a2e078bf61.lhrtunnel.link

# # webhook heroku
# https://api.telegram.org/bot[token]/setWebhook?url=https://lz-rossmann-telegram-bot.herokuapp.com/

# # delete webhook
# https://api.telegram.org/bot[token]/deleteWebhook

# # send message
# https://api.telegram.org/bot[token]/sendMessage?chat_id=640276497&text=Hi Laissa, I am doing good, tks!


def parse_message( message ):
 
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']
    
    store_id = store_id.replace( '/', '')
    
    try:
        store_id = int( store_id )
        
    except ValueError:
        store_id = 'error'
    
    return chat_id, store_id


def send_message( chat_id, text):
    
    url = 'https://api.telegram.org/bot{}/'.format( token )
    url = url + 'sendMessage?chat_id={}'.format( chat_id ) 

    r = requests.post( url, json={ 'text': text })
    print( 'Status Code {}'.format( r.status_code) )
    
    return None


def load_dataset( store_id ):

    # loading test dataset

    df_test = pd.read_csv( 'test.csv' )
    df_store_raw = pd.read_csv( 'store.csv' , low_memory=False)

    # merge test dataset + store
    df_test = pd.merge( df_test, df_store_raw, how='left', on='Store' )

    # choose store for prediction
    df_test = df_test[ df_test['Store'] == store_id ]
    
    # checking if store id exist
    if not df_test.empty:   # yes
        # remove closed days
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[~df_test['Open'].isnull()]
        df_test = df_test.drop( 'Id', axis=1 )

        #  convert dataframe to json
        data = json.dumps( df_test.to_dict( orient='records' ) )
        
    else:   # no
        data='error'
    
    return data


def predict( data ):

    # API call
    url = 'https://lz-rossmann-model.herokuapp.com/rossmann/predict' #url heroku 
    header = {'Content-type': 'application/json' } 
    data = data

    r = requests.post( url, data=data, headers=header )
    print( 'Status Code {}'.format( r.status_code ) )

    d1 = pd.DataFrame( r.json(), columns=r.json()[0].keys() )
    
    return d1


# API initialize
app = Flask( __name__ )

@app.route( '/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.get_json()
        
        chat_id, store_id = parse_message( message )
        
        if 'hello' in str(store_id):
            
            msg = 'Hello, I\'am Rossmann Bot. Please, type the desired store id that I will tell you how much the store will sell in the next 6 weeks.'
            send_message( chat_id, msg) 
            return Response( 'OK', status=200) 
        
        else: 

            if store_id != 'error':
                
                # loading data
                data = load_dataset( store_id )
                
                if data != 'error':
                    
                    # prediction
                    d1 = predict(data)
                    
                    # send message
                    msg = 'Store Number {} will sell R${:,.2f} in the next 6 weeks. '.format( 
                            d1['store'].values[0], 
                            d1['prediction'].values[0] )

                    send_message( chat_id, msg)
                    return Response( 'OK', status=200)
                
                else:
                    send_message( chat_id, 'Store not available')
                    return Response( 'OK', status=200)
            
            else:
                send_message( chat_id, 'Store ID is wrong')
                return Response( 'OK', status=200)
             
    else:
        return '<h1> Rossmann Telegram BOT </h1>'
    
    
if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( host='0.0.0.0', port=port)
    
