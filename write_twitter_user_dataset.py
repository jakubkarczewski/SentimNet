import tweepy #https://github.com/tweepy/tweepy
import csv
import pandas as pd


#Twitter API credentials
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""


def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method

	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)

	#initialize a list to hold all the tweepy Tweets
	alltweets = []

	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)

	#save most recent tweets
	alltweets.extend(new_tweets)

	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1

	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print("getting tweets before %s" % (oldest))

		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

		#save most recent tweets
		alltweets.extend(new_tweets)

		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1

		print("...%s tweets downloaded so far" % (len(alltweets)))

	#transform the tweepy tweets into a 2D array that will populate the csv
	outtweets = [[tweet.created_at.strftime('%m-%d-%Y'), tweet.text] for tweet in alltweets]
	df_tweets = pd.DataFrame(outtweets)
	df_tweets_gr = df_tweets.groupby(0)
	all_days = [df_tweets_gr.get_group(x) for x in df_tweets_gr.groups]
	import ipdb; ipdb.set_trace()
	#write the csv
	for day in all_days:
		# day = day.str.encode('utf-8')
		day_raw = day[day.columns[1]]
		day_raw.to_csv('./musk_grouped/%s_from_%s.csv' % (screen_name, day[0].values[0]), index=False, sep=',',  encoding='utf8')
	pass


if __name__ == '__main__':
	#pass in the username of the account you want to download
	get_all_tweets("elonmusk")
