"""
24 July 2023
Econ 890 session II: working with data

Example using "Survivor" data
(data sourced from: https://github.com/doehm/survivoR/tree/master/data)

For more pandas basics, see the documentation here:
https://pandas.pydata.org/docs/user_guide/basics.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########################################################################################################################
# directories
########################################################################################################################

# input the location where you have saved the survivor data .csv files
data = 'C:/Users/jpkad/Documents/survivor_data/data/'

########################################################################################################################
# read in dataset with information on Survivor 'castaways'
########################################################################################################################

# read in .csv files using pandas' .read_csv function
# (pandas can also read in many other file types, including excel, html, json, etc.)
df = pd.read_csv(data + 'castaways_w_detail.csv')

# take a quick look at the dataset using .head() to print the first 5 rows
print(df.head(5))

########################################################################################################################
# data structures in pandas
########################################################################################################################

# two main data structures: dataframes and series

# dataframe
####################################################

# a dataframe is a two-dimensional tabular data structure (like an excel spreadsheet or a SQL table)
# it consists of rows, columns, and data

# the simplest way to select a single column is to use the column name
# you can get all the column names by calling '.columns.to_list()' on a dataframe
col_names = df.columns.to_list()
print(col_names)

# let's look at the column with all the season names (the first column, or item zero in our list of column names)
my_col = col_names[0]
print(df[my_col])   # the output will also display the length (number of observations) and datatype ('dtype')

# you can select a single row using the row index and the .loc method
# the index in this dataframe is simply the range of integers from 0 to 761, but a pandas index can be anything you'd
# like (e.g., strings, floats, etc.)
print(df.loc[3])   # print the fourth row of the dataframe

# you can select a single data point by extending the .loc method to include a column reference
print(df.loc[3, my_col])

# series
####################################################

# a series is a list or array of values (can be any object type or of mixed types, though the latter is rarely
# desirable in practice)
names_series = df['full_name']
print(type(names_series))

# series are indexed, and, if they are taken from a dataframe, they inherit the index of the larger dataframe
print(names_series.head(5))
print(names_series[4])

########################################################################################################################
# data wrangling with pandas
########################################################################################################################

# summarizing data
######################################################

# standard summary stats are built in to pandas, and can be called on the whole dataframe or on individual series

# describe the whole dataframe using .describe
print(df.describe())   # this will display distributional summary statistics for NUMERIC variables only
print(df.mean())   # similarly, this will display the mean for NUMERIC variables only

# you can call mean, median, mode, min, etc. on single series as well
print('the maximum number of days in a season is:', df['day'].max())

# you can use functions like 'nunique' (number of unique values) and 'value_counts' to describe non-numeric variables
print('there are', df['personality_type'].nunique(), 'unique personality types in the data')
print('the distribution of personality types is:\n', df['personality_type'].value_counts())

# you can also easily check correlations between different columns using .corr
print(df.corr())   # calling .cor on the whole dataframe shows correlations between all numeric columns

# change display settings to show all the correlations
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
print(df.corr())


# filtering data
######################################################

# let's see how many survivor contestants are over 40
# first we can set our condition
over_forty = df['age'] > 40

# we can then use this condition to limit the dataframe to only rows with 'age' values meeting this condition
print(df[over_forty].head(5))

# the length of this dataframe will give us the total number of contestants who are currently over forty
print('The total number of survivor contestants over 40 is:', len(df[over_forty]), 'out of', len(df))

# let's use this approach to create a new dataframe containing only survivor winners ('sole survivors')
winner = df['result'] == 'Sole Survivor'
df_winners = df[winner]

# pandas also has built-in methods for identifying and dealing wtih duplicates
df['duplicated'] = df.duplicated(keep=False)   # create a new variable that identifies duplicate observations

# let's check whether there are any duplicated castaway observations (there shouldn't be)
print(df[df['duplicated'] == True])

# we can also check for duplicates based on a subset of variables
df['duplicated_season'] = df.duplicated(subset='season', keep='first')
print(df[df['duplicated_season'] == True])   # unsurprisingly, lots of castaways have the same season value

# we can also directly drop duplicates
df.drop_duplicates(keep='last', inplace=True)   # if no subset argument is given, duplication is based on all variables

# we can use the 'drop' function to drop rows or columns based on other criteria
df.drop(columns='duplicated_season', inplace=True)

# grouping / aggregating data
######################################################

# use the pandas 'groupby' method to aggregate data based on a given variable or subset of variables

# let's create a new dataframe of the winners grouped by gender
# the .agg function allows us to choose the aggregation method for the variables we include in the grouped dataframe
df_winners_by_gender = df_winners.groupby('gender', as_index=False).agg({'season_name': 'count', 'age': 'mean'})
# let's rename the season count column to make the data a bit clearer
df_winners_by_gender.rename(columns={'season_name': 'seasons_won'}, inplace=True)

print(df_winners_by_gender)

# we can also group by multiple variables, e.g., gender and personality type:
df_winners_by_gender_and_ptype = df_winners.groupby(['gender', 'personality_type'],
                                                    as_index=False).agg({'season_name': 'count', 'age': 'mean'})
df_winners_by_gender_and_ptype.rename(columns={'season_name': 'seasons_won'}, inplace=True)   # again, rename column for clarity

# let's print the "winningest" gender / personality type combo
max_wins = df_winners_by_gender_and_ptype['seasons_won'].max()
max_winner = df_winners_by_gender_and_ptype['seasons_won'] == max_wins
print(r'the winningest gender-ptype combo is:\n', df_winners_by_gender_and_ptype[max_winner][['gender', 'personality_type']])

# merging data from multiple dataframes
######################################################

# read in dataset with challenge result information
df_chal = pd.read_csv(data + 'challenge_results.csv')

# let's inspect the data to get a sense of what it looks like
print(df_chal.head(5))

# to simplify, let's drop challenges with more than one winner id
# we saw above that lists of winners have the format c("2", "4", ...) in this dataset

# first, we need to drop "nan" ("not a number") values, which will break our conditional statement
df_chal.dropna(subset=['winner_ids'], inplace=True)

multiple_winners = df_chal['winner_ids'].str.contains('c')   # create a condition identifying when there's >1 winner
df_chal = df_chal[~multiple_winners]   # keep only challenges that DO NOT satisfy the above condition ("~" negates)

# finally, let's convert 'winner_ids' to dataype 'int' (integer) so that it matches the castaway id type in our
# other dataframe
df_chal['winner_ids'] = df_chal['winner_ids'].astype('int')

# now we can merge the castaway details for each challenge winner into this dataset
# NOTE: we must merge on BOTH castaway id AND season name, since some castaways appear in multiple seasons
df_chal.rename(columns={'winner_ids': 'castaway_id'}, inplace=True)   # rename id column to match with our other df
df_chal = df_chal.merge(df, how='left', on=['castaway_id', 'season_name'])

# there are LOTS of optional arguments in the merge function--see the pandas documentation for more detail:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html

# transforming data using functions
######################################################

# you can apply basic math to variables with intuitive notation
# e.g., let's add a variable calculating how long each castaway survived as a percent of the total season length

# first, make a total season length variable which takes the max number of days from each season group
# use groupby + transform to apply a function groupwise
df['season_length'] = df.groupby('season')['day'].transform('max')

# next, divide the 'day' variable by season_length to get our outcome of interest (days survived as % of total)
# you can use the same basic math notation (+, -, *, /) on columns as you can with regular numbers in python (as long
# as the datatypes are numeric)
df['days_survived_pct'] = df['day'] / df['season_length']

# you can also write your own custom functions to transform variables and use them with '.apply'
# E.g., let's write a function to extract the year of birth from the 'date_of_birth' variable
# (NOTE: we could also do this by converting this variable to pandas built-in datetime dtype; see documentation for
# more: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)


# first, define our function (recalling that the date of birth variable is in the format 'yyyy-mm-dd')
def get_birth_year(bday):
    if isinstance(bday, str):
        return bday[:4]
    else:
        return 'NA'


# now we apply the function to get each castaway's birth year:
df['birth_year'] = df['date_of_birth'].apply(get_birth_year)

# exporting data
######################################################

# to save a dataframe, simply use pandas 'to_csv' command
# (as with importing data, there are equivalent functions for other file types, e.g., 'to_excel', 'to_json', etc.)
# see documentation for the full set of output options:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
my_filename = 'survivor_data_edited.csv'
df.to_csv(data + my_filename, index=False)

########################################################################################################################
# simple data visualization with matplotlib and seaborn
########################################################################################################################

# we can make a very basic bar plot with one line of code using sns.barplot, comparing win rates by gender
sns.barplot(data=df_winners_by_gender, x='gender', y='seasons_won')

# call plt.close to close the current figure before creating a new one
plt.close()

# similarly, let's make a basic scatter plot comparing age with order of elimination
sns.scatterplot(data=df, x='age', y='order')

plt.close()

# regplot can show us the line of best fit (surprisingly, there's not a clear relationship between age and
# survival)
sns.regplot(data=df, x='age', y='order')

plt.close()

# matplotlib / seaborn have lots of options for making your plots prettier
# let's plot how many days castaways survived, by personality type and gender, using violinplot and some of these
# formatting options
# (violinplot visualizes the distribution of data across different categories)

# first, drop observations missing gender data
nonmissing_gender = (df['gender'] == 'Female') | (df['gender'] == 'Male')
df = df[nonmissing_gender]

# now let's create a variable identifying whether an individual is an extrovert or not based on their personality type
# (we'll use this in our plot)
df['extrovert'] = np.where(df['personality_type'].str.contains('E'), True, False)

# sort dataframe by personality type (this will determine order on the plot)
df = df.sort_values(by='personality_type')

# set seaborn plot style to "darkgrid"
# (read more about style options here: https://seaborn.pydata.org/tutorial/aesthetics.html)
sns.set_style('darkgrid')

# FacetGrid allows us to quickly generate multiple subplots from the same dataset, for different data categories
# See documentation for more detail: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
my_plot = sns.FacetGrid(df, col='extrovert', sharex=False, sharey=True)

my_plot.map_dataframe(sns.violinplot, x='personality_type', y='days_survived_pct',
                      hue='gender',   # the hue argument assigns different values of this variable to different colors
                      split=True,   # split is an argument for violinplot, splitting the distribution plot by the hue variable
                      palette='Paired')   # seaborn has a number of preset color palettes, or you can build your own

my_plot.set_xticklabels(rotation=45)   # rotate axis labels for visibility

my_plot.set_xlabels("")   # remove x labels (unnecssary from context)
my_plot.set_ylabels('Days survived (% of total)', x=-0.1, fontsize=11)   # set y labels

my_plot.set_titles(size=11)   # set subplot title attributes

my_plot.fig.suptitle('Survival time by personality type and gender', y=0.95, fontsize=14)   # set overall plot title

my_plot.tight_layout()   # automatically update plot size to fit figure and labels

my_plot.add_legend()   # add legend

# save plot using savefig
my_plot.savefig(data + 'my_plot.png')
