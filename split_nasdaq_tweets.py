import subprocess

import pandas as pd


def split_datasets(file_path, sheet_number, dir_name):

    subprocess.call(['mkdir', dir_name])
    all_sheets = pd.ExcelFile(file_path)
    one_sheet = all_sheets.parse(sheet_number)

    date_body_lang = one_sheet.filter(['Date', 'Tweet content',
                                       'Tweet language (ISO 639-1)'])
    date_body_lang = date_body_lang.groupby('Date')

    all_days = [date_body_lang.get_group(x) for x in date_body_lang.groups]
    for one_day in all_days:
        one_day.to_csv('./nasdaq_tweets/%s.csv' % one_day['Date'].iloc[2])

