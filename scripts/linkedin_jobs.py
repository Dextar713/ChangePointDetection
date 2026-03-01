import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os 

def load_and_clean():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'linkedin_job_posts.csv')
    jobs_df = pd.read_csv(data_path, parse_dates=True)
    cols_to_keep = ['jobpost', 'date', 'Title', 'Company', 'Location', 'RequiredQual', 'Year', 'Month', 'IT']
    jobs_df = jobs_df[cols_to_keep] 
    # print(jobs_df.isna().sum(axis=0))
    print(jobs_df['Title'].value_counts())


if __name__ == '__main__':
    load_and_clean()