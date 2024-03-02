import firebase_admin
from firebase_admin import credentials, messaging

cred = credentials.Certificate("C:/seminarska/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

def sendPush(title, msg, registration_token, akcija):
    message = messaging.Message(
        data={
            'parkirisce': akcija[1],
            'prihod': akcija[2],
            'odhod': akcija[3],
            'title' : title,
            'body' : msg,
            'sound': 'default'
        },

        token=registration_token,
        android=messaging.AndroidConfig(
            priority='high',
        )
    )
    response = messaging.send(message)
    print("Sporočilo poslano: ", response)



