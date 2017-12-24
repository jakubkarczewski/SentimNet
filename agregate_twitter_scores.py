import pandas as pd
import re
import os
from datetime import datetime


def _get_daily_score(date, tweets_dir):
    all_data = pd.read_csv('./%s/%s.csv' % (tweets_dir, date), header=None)
    string_tuples = [tuple(x) for x in all_data.values]
    numerical_values = []
    for string_tuple in string_tuples:
        numerical_values.append(re.findall("\d+\.\d+", string_tuple[0]))
    score_per_tweet = [float(score[1]) - float(score[0]) for score in numerical_values]
    daily_score = sum(score_per_tweet)
    return daily_score


def iter_through_scores(tweets_dir, company):
    all_files = os.listdir('./%s' % tweets_dir)
    all_scores = [score for score in all_files if ('score' in score and company
                                                   in score)]
    daily_scores = []
    for score in all_scores:
        daily_score = _get_daily_score(score[:-4], tweets_dir)
        str_date = score[:-10].split('from_', 1)[1]
        daily_scores.append([daily_score, str_date, datetime.strptime(str_date,
                                                                       "%Y-%m-%d")])
        # daily_scores.append([daily_score, str_date, datetime.strptime(str_date,
        #                                                               "%m-%d-%Y")])
    daily_scores.sort(key=lambda x: x[2])

    weekend_agregate = 0
    for i, score in enumerate(daily_scores):
        # import ipdb; ipdb.set_trace()

        weekno = score[2].weekday()

        if weekno == 5:
            weekend_agregate += 0.2 * score[0]
        elif weekno == 6:
            weekend_agregate += 0.4 * score[0]
            if(i+1 < len(daily_scores)):
                daily_scores[i+1][0] += weekend_agregate
                daily_scores[i + 1][0] /= 1.6
            weekend_agregate = 0

    daily_scores_weekdays = [daily for daily in daily_scores
                             if daily[2].weekday() < 5]

    return daily_scores_weekdays
