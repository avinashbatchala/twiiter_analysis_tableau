#Import Required Modules
from tweepy.streaming import StreamListener #prints live tweets to console
from tweepy import OAuthHandler #Authenticates User APIs
from tweepy import Stream

# Twitter API keys are generted using twitter developer account. https://dev.twitter.com/apps/new, use this link to generate API keys
'''
Replaces the empty strings with API keys genrated, I removed them here as they are private
'''


# List of keywords that must be included in the tweets, which we will extract
hash_tags = ['Kabul', 'Afghanistan', 'Taliban']

# Create the class that will handle the tweet stream.
class StdOutListener(StreamListener):
    '''
    This class is taken from tweepy documentation and a minor modification is made to download reuired number of tweets.
    '''
    def on_data(self, data):
        global count
        global num_tweets
        global stream
        # if count < num_tweets:
        file = open("testTweets.txt", "a")
        file.write(data)
        file.close()
            # count += 1
        return True
        # else:
        #     stream.disconnect()

    def on_error(self, status):
        print(status)


# Handles Twitter authentication
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, l)

stream.filter(languages=["en"], track=hash_tags)  #stream.filter param is used to extract only desired tweets. In this program the params used are languages=['en'], this is used to extract only tweets in english language.

import testCleaner