"""
24 July 2023
Econ 890 session III: data vizualization with Matplotlib and Seaborn

Modified version of materials from Eric Monson
(https://github.com/emonson/pandas-jupyterlab/blob/master/SeabornAdvanced.ipynb)

For more pandas basics, see the documentation here:
https://pandas.pydata.org/docs/user_guide/basics.html

Seaborn documentation: https://seaborn.pydata.org/api.html

"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
from urllib.request import urlopen
import os
import pandas as pd
sns.set_style("whitegrid")

########################################################################################################################
# Read in data
# (The pre-loaded Tips data set is a really nice data set for exploring differences between numerical values and
# distributions across a population distinguished by lots of categorical variables.)
########################################################################################################################

tips = sns.load_dataset("tips")

tips.head(10)

########################################################################################################################
# Basic chart creation, formatting, and saving
########################################################################################################################

# it's often helpful to define your desired/standard plot dimensions at the top of your script
plt_dim = (14, 14)
label_size = 25
title_size = 30
tick_size = 20

# this creates a single figure with a single axis; you can create figures with multiple axes using additional arguments
fig, ax = plt.subplots(figsize=plt_dim)

# using seaborn, create a simple bar plot on the axis we just initialized
bar_chart = sns.barplot(data=tips, x="day", y="tip", ax=ax, palette='Paired')

# seaborn has a number of pre-defined color palettes, or you can customize your own
# more info here: https://seaborn.pydata.org/tutorial/color_palettes.html
fig, ax = plt.subplots(figsize=plt_dim)
sns.barplot(data=tips, x="day", y="tip", ax=ax, palette='RdPu')

# setting chart labels
ax.set_xlabel("Day of week", fontsize=label_size)
ax.set_ylabel("Total tip amount ($)", fontsize=label_size)
ax.set_title("Tips by day of week", fontsize=title_size)
ax.tick_params(labelsize=tick_size)

# you can also rotate the x labels (this comes in handy often!)
plt.xticks(rotation=90)

# tight layout cuts off border/white space
plt.tight_layout()

# save plot
output = 'C:/Users/jpkad/Documents/'
plot_loc = os.path.join(output, 'my_bar_chart.png')
fig.savefig(plot_loc)

# close plot (avoids accidentally plotting multiple graphs on the same axis)
plt.close()

########################################################################################################################
# Individual variables & distributions
########################################################################################################################

# Histogram
##################
# This function combines the matplotlib hist function (with automatic calculation of a good default bin size) with
# the seaborn kdeplot() and rugplot() functions. It can also fit scipy.stats distributions and plot the estimated PDF
# over the data.
ax = sns.displot(tips.total_bill, kde=True)

# You can turn on and off various aspects, and easily control how many bins are used.
ax = sns.displot(tips.total_bill, bins=30, kde=False, rug=True)

# Since displot() is a convenience combination of a histogram, kdeplot(), and rugplot(), you can also include
# keywords for each sub-plot type (e.g., you can pass the bw_method argument that is associated with kdeplot)
ax = sns.displot(tips.total_bill, kind='kde', rug=True, bw_method=0.1)

# setting axis labels and chart title for facet grid
ax.set_axis_labels("Total bill ($)", "Density")
ax.set_titles("Distribution of total bill")

# Strip plot
#################
# You notice that the rug plot shows all individuals, but it's hard without the histogram or kde to tell how many
# points are overlapping. One solution is to "jitter" the points randomly along the categorical axis with stripplot().

ax = sns.stripplot(x='total_bill', data=tips, alpha=0.5, jitter=0.2)

# Swarm plot
##################
# Jitter doesn't get rid of all overlap, though, so an interesting alternative to stripplot() is a swarmplot().
# Points are stacked at their data value.
ax = sns.swarmplot(x="total_bill", data=tips)

########################################################################################################################
# Splitting by a categorical variable
########################################################################################################################

# Now, split the data by categorical variables (both in space, by "day", and in hue (color) by "sex")
ax = sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips)

# We can also use a violin plot "split" by a variable.
ax = sns.violinplot(x="day", y="tip", hue="sex", split=True, data=tips)

# You can even superimpose plots by putting them on the same "axis".
ax = sns.boxplot(x="day", y="total_bill", color=(0.8, 0.8, 0.65, 0.5), data=tips)
sns.swarmplot(ax=ax, x="day", y="total_bill", color='black', alpha=0.3, data=tips)

# Catplot for drawing categorical plots onto a (facet)grid
###########################################################
# Shows the relationship between a numerical and one or more categorical variables using one of several visual
# representations. The kind parameter selects the underlying function to use:

# Categorical scatterplots:
# stripplot() (with kind="strip"; the default)
# swarmplot() (with kind="swarm")

# Categorical distribution plots:
# boxplot() (with kind="box")
# violinplot() (with kind="violin")
# boxenplot() (with kind="boxen")

# Categorical estimate plots:
# pointplot() (with kind="point")
# barplot() (with kind="bar")
# countplot() (with kind="count")

# comparing tips from smokers vs. non-smokers, broken out by day with catplot
ax = sns.catplot(x="smoker", y="total_bill",
                    hue="sex", col="day",
                    data=tips, kind="bar")

# point plot
#####################
# comparing tips, and difference in tips between lunch and dinner, for smokers vs. non-smokers with pointplot
# A point plot represents an estimate of central tendency for a numeric variable by the position of scatter plot
# points and provides some indication of the uncertainty around that estimate using error bars.
ax = sns.pointplot(x="time", y="total_bill", hue="smoker",
                    data=tips, dodge=True)

# Joint plot (bivariate distributions)
#######################################
# It can also be useful to visualize a bivariate distribution of two variables. The easiest way to do this in
# seaborn is to just use the jointplot() function, which creates a multi-panel figure that shows both the bivariate
# (or joint) relationship between two variables along with the univariate (or marginal) distribution of each on
# separate axes

ax = sns.jointplot(x="total_bill", y="tip", data=tips)

# When there is a lot of overlap, it can be helpful to show a heatmap rather than individual points.
# (Unfortunately, this doesn't produce a color legend)
ax = sns.jointplot(x="total_bill", y="tip", data=tips, kind="hex", stat_func=None)

# Basic linear regression
###########################

# Tip regressed on total bill
ax = sns.regplot(x="total_bill", y="tip", data=tips)

# Total bill regressed on table size (number of diners)
ax = sns.regplot(x="size", y="total_bill", data=tips, x_jitter=.1)

# lmplot() – Plot data and simple regressions across a FacetGrid
####################################################################
# This function combines regplot() and FacetGrid. It is intended as a convenient interface to fit regression models
# across conditional subsets of a dataset.

ax = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)

# It's very convenient to be able to wrap the columns after a certain count
ax = sns.lmplot(x="total_bill", y="tip", col="day", hue="day", data=tips, col_wrap=2, height=4)

# Pairwise comparisons with pairplot
######################################
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species')

########################################################################################################################
# Quick and easy geo mapping with plotly
########################################################################################################################

# use example dataset (unemployement rate by county fips code)
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv", dtype={"fips": str})
print(df.head())

# read in county fips code mapping
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# create chloropleth map
# see here for full documentation of the chloropleth function and arguments:
# https://plotly.com/python/choropleth-maps/
fig = px.choropleth(df, geojson=counties, locations='fips', color='unemp',
                    color_continuous_scale="Viridis",
                    range_color=(0, 12),
                    scope="usa",
                    labels={'unemp':'unemployment rate'}
                    )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# have to use "write_image" function to save plotly figures
fig.write_image('my_chloropleth.png')

