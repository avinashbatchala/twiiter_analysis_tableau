# Import Required Modules
from tweepy.streaming import StreamListener  # prints live tweets to console
from tweepy import OAuthHandler  # Authenticates User APIs
from tweepy import Stream

# Twitter API keys are generted using twitter developer account. https://dev.twitter.com/apps/new, use this link to generate API keys
"""
Replaces the empty strings with API keys genrated, I removed them here as they are private
"""
access_token = "4824134823-0RmJm6xrG905UVP6CIsQ0kvTISasUVkImv7TR6O"
access_token_secret = "VCqgnYiDPF6elHcnbRGJRiTGs3Vo3TiEKXcoM4G2oWMIS"
consumer_key = "RMPUEZXZcYTe0UzIhCqJ7hVDy"
consumer_secret = "Or5j6mnXcIOKVLeF5aHBpkecfFIdmvivoLdX3SaiJK3ptJYLwX"

# List of keywords that must be included in the tweets, which we will extract
hash_tags = ["Apple", "Iphone13", "series7"]

# Create the class that will handle the tweet stream.
class StdOutListener(StreamListener):
    """
    This class is taken from tweepy documentation and a minor modification is made to download reuired number of tweets.
    """

    def on_data(self, data):
        global stream
        file = open("testTweets.txt", "a")
        file.write(data)
        file.close()
        return True

    def on_error(self, status):
        print(status)


# Handles Twitter authentication
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
stream = Stream(auth, l)

stream.filter(
    languages=["en"], track=hash_tags
)  # stream.filter param is used to extract only desired tweets. In this program the params used are languages=['en'], this is used to extract only tweets in english language.
