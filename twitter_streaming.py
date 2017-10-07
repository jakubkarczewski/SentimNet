import json

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

from infer import classify_tweet


#consumer key, consumer secret, access token, access secret.
ckey="uwQoag4aBHKJcFsOsKi8q94iS"
csecret="rz89scxAxLla6OQDmNUyQ0TFCZt5L4IT6XoVzBEfRmcuRTgvS2"
atoken="878227501880946688-zaypdJupSY5bQOYg6DIbGTwIp3JlF50"
asecret="hafrjgx2zoIVATBLmb4HppsaDiQLLkZGZjqfYXK6D6qSc"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        print(classify_tweet(tweet))
        #print(tweet)
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])