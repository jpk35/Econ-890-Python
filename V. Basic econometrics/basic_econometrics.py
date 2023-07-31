"""
24 July 2023
Econ 890 session VI: basic econometrics

Based on Part 78 Linear Regression in Python of Intermediate Quantitative Economics with Python
https://python.quantecon.org/intro.html#
"""

import matplotlib
matplotlib.use("Qt5Agg")   # set backend
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (11, 5)  #set default figure size
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from linearmodels.iv import IV2SLS
import pylatex as pl
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat
from pylatex.utils import bold, italic, NoEscape

########################################################################################################################
# In this lecture, we’ll use the Python package statsmodels to estimate, interpret, and visualize linear
# regression models.
#
# Along the way, we’ll discuss a variety of topics, including
# --simple and multivariate linear regression
# --visualization
# --endogeneity and omitted variable bias
# --two-stage least squares
#
# As an example, we will replicate results from Acemoglu, Johnson and Robinson’s seminal paper [AJR01].
#
# You can download a copy here:
# https://economics.mit.edu/sites/default/files/publications/colonial-origins-of-comparative-development.pdf
#
# In the paper, the authors emphasize the importance of institutions in economic development.
#
# The main contribution is the use of settler mortality rates as a source of exogenous variation in institutional
# differences.
#
# Such variation is needed to determine whether it is institutions that give rise to greater economic growth, rather
# than the other way around.
########################################################################################################################

########################################################################################################################
# Simple Linear Regression
########################################################################################################################

# [AJR01] wish to determine whether or not differences in institutions can help to explain observed economic outcomes.
#
# How do we measure institutional differences and economic outcomes?
#
# In this paper,
# --economic outcomes are proxied by log GDP per capita in 1995, adjusted for exchange rates.
# --institutional differences are proxied by an index of protection against expropriation on average over 1985-95,
#    constructed by the Political Risk Services Group.
# --These variables and other data used in the paper are available for download on Daron Acemoglu’s webpage.

# read in data
df1 = pd.read_stata('https://github.com/QuantEcon/lecture-python/blob/master/source/_static/lecture_specific/ols/maketable1.dta?raw=true')
print(df1.head())

# Let’s use a scatterplot to see whether any obvious relationship exists between GDP per capita and the protection
# against expropriation index
df1.plot(x='avexpr', y='logpgp95', kind='scatter')

# The plot shows a fairly strong positive relationship between protection against expropriation and log GDP per capita.
# Given the plot, choosing a linear model to describe this relationship seems like a reasonable assumption:
# logpgp95_i = beta_0 + beta_1 * avexpr_i + e_i

# Plot a basic OLS estimate (regressing per capita GDP on protection against expropriation)
###########################################################################################

# Dropping NA's is required to use numpy's polyfit
df1_subset = df1.dropna(subset=['logpgp95', 'avexpr'])

# Use only 'base sample' for plotting purposes
df1_subset = df1_subset[df1_subset['baseco'] == 1]

X = df1_subset['avexpr']
y = df1_subset['logpgp95']
labels = df1_subset['shortnam']

# Replace markers with country labels
fig, ax = plt.subplots()
ax.scatter(X, y, marker='')

for i, label in enumerate(labels):
    ax.annotate(label, (X.iloc[i], y.iloc[i]))

# Fit a linear trend line
ax.plot(np.unique(X),
        np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
        color='black')

ax.set_xlim([3.3,10.5])
ax.set_ylim([4,10.5])
ax.set_xlabel('Average Expropriation Risk 1985-95')
ax.set_ylabel('Log GDP per capita, PPP, 1995')
ax.set_title('Figure 2: OLS relationship between expropriation \
    risk and income')

# Get a parameter estimate for the coefficient of interest (beta_1) in our OLS estimate
###########################################################################################

# To estimate the constant term beta_0, we need to add a column of 1’s to our dataset
df1['const'] = 1

# Now we can construct our model in statsmodels using the OLS function.
# We will use pandas dataframes with statsmodels, however standard arrays can also be used as arguments
reg1 = sm.OLS(endog=df1['logpgp95'], exog=df1[['const', 'avexpr']], missing='drop')
print(type(reg1))

# So far we have simply constructed our model.
# We need to use .fit() to obtain parameter estimates beta_0_hat and beta_1_hat
results = reg1.fit()
print(type(results))

# We now have the fitted regression model stored in results.
# To view the OLS regression results, we can call the .summary() method.
# Note that an observation was mistakenly dropped from the results in the original paper (see the note located in
# maketable2.do from Acemoglu’s webpage), and thus the coefficients differ slightly.
print(results.summary())

# From our results, we see that:
# --The intercept beta_0_hat = 4.63
# --The slope beta_1_hat = 0.53
# --The positive beta_1 parameter estimate implies that institutional quality has a positive effect on economic outcomes
# --The p-value of 0.000 for beta_1 implies that the effect of institutions on GDP is statistically significant

# Using our parameter estimates, we can now write our estimated relationship as:
#    logpgp95_i_hat = 4.63 + 0.53 * avexpr_i

# We can use this equation to predict the level of log GDP per capita for a value of the index of expropriation
# protection, use .predict() and setting constant = 1 and avexpr_i = mean_expr
mean_expr = np.mean(df1_subset['avexpr'])
print(results.predict(exog=[1, mean_expr]))

# We can obtain an array of predicted values for every value of avexpr in our dataset by calling .predict() on our
# results.
#
# Plotting the predicted values against the actual values shows that the predicted values lie along the linear line
# that we fitted above.

# Drop missing observations from whole sample
df1_plot = df1.dropna(subset=['logpgp95', 'avexpr'])

# Plot predicted values
fig, ax = plt.subplots()
ax.scatter(df1_plot['avexpr'], results.predict(), alpha=0.5, label='predicted')

# Plot observed values
ax.scatter(df1_plot['avexpr'], df1_plot['logpgp95'], alpha=0.5, label='observed')
ax.legend()
ax.set_title('OLS predicted values')
ax.set_xlabel('avexpr')
ax.set_ylabel('logpgp95')

########################################################################################################################
# Exporting regression results to latex
########################################################################################################################

# convert summary table to latex
results_latex = results.summary().as_latex()

# use pylatex pacakge to compile results table into a simple pdf
geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
doc = Document(geometry_options=geometry_options)
doc.preamble.append(pl.Package('booktabs'))   # add booktabs package so table can compile properly

with doc.create(Section('OLS regression results')):
    doc.append(NoEscape(results_latex))

doc.generate_pdf('ols_results', clean_tex=False)

########################################################################################################################
# Multivariate regression
########################################################################################################################

# Leaving out variables that affect per capita gdp will result in omitted variable bias, yielding biased and
# inconsistent parameter estimates.
#
# We can extend our bivariate regression model to a multivariate regression model by adding in other factors that
# may affect per capita gdp
#
# [AJR01] consider other factors such as:
# --the effect of climate on economic outcomes; latitude is used to proxy this
# --differences that affect both economic performance and institutions, eg. cultural, historical, etc.; controlled for with the use of continent dummies
#
# Let’s estimate some of the extended models considered in the paper (Table 2) using data from maketable2.dta

# read in data
df2 = pd.read_stata('https://github.com/QuantEcon/lecture-python/blob/master/source/_static/lecture_specific/ols/maketable2.dta?raw=true')

# Add constant term to dataset
df2['const'] = 1

# Create lists of variables to be used in each regression
X1 = ['const', 'avexpr']
X2 = ['const', 'avexpr', 'lat_abst']
X3 = ['const', 'avexpr', 'lat_abst', 'asia', 'africa', 'other']

# Estimate an OLS regression for each set of variables
reg1 = sm.OLS(df2['logpgp95'], df2[X1], missing='drop').fit()
reg2 = sm.OLS(df2['logpgp95'], df2[X2], missing='drop').fit()
reg3 = sm.OLS(df2['logpgp95'], df2[X3], missing='drop').fit()

# Now that we have fitted our model, we will use summary_col to display the results in a single table
# (model numbers correspond to those in the paper)
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

results_table = summary_col(results=[reg1,reg2,reg3],
                            float_format='%0.2f',
                            stars = True,
                            model_names=['Model 1',
                                         'Model 3',
                                         'Model 4'],
                            info_dict=info_dict,
                            regressor_order=['const',
                                             'avexpr',
                                             'lat_abst',
                                             'asia',
                                             'africa'])

results_table.add_title('Table 2 - OLS Regressions')

print(results_table)

########################################################################################################################
# IV regression
########################################################################################################################

# As [AJR01] discuss, the OLS models likely suffer from endogeneity issues, resulting in biased and inconsistent model
# estimates.

# Namely, there is likely a two-way relationship between institutions and economic outcomes:
# --richer countries may be able to afford or prefer better institutions
# --variables that affect income may also be correlated with institutional differences
# --the construction of the index may be biased; analysts may be biased towards seeing countries with higher income
#     having better institutions

# To deal with endogeneity, we can use two-stage least squares (2SLS) regression, which is an extension of OLS
# regression. This method requires replacing the endogenous variable with a variable that is:
# --correlated with expropriation risk
# --NOT correlated with the error term (ie. it should not directly affect the dependent variable, otherwise it
#     would be correlated with e_i due to omitted variable bias)

# The main contribution of [AJR01] is the use of settler mortality rates to instrument for institutional differences.
# They hypothesize that higher mortality rates of colonizers led to the establishment of institutions that were more
# extractive in nature (less protection against expropriation), and these institutions still persist today.

# Using a scatterplot (Figure 3 in [AJR01]), we can see protection against expropriation is negatively correlated
# with settler mortality rates, coinciding with the authors’ hypothesis and satisfying the first condition of a valid
# instrument.

# Dropping NA's is required to use numpy's polyfit
df1_subset2 = df1.dropna(subset=['logem4', 'avexpr'])

X = df1_subset2['logem4']
y = df1_subset2['avexpr']
labels = df1_subset2['shortnam']

# Replace markers with country labels
fig, ax = plt.subplots()
ax.scatter(X, y, marker='')

for i, label in enumerate(labels):
    ax.annotate(label, (X.iloc[i], y.iloc[i]))

# Fit a linear trend line
ax.plot(np.unique(X),
         np.poly1d(np.polyfit(X, y, 1))(np.unique(X)),
         color='black')

ax.set_xlim([1.8,8.4])
ax.set_ylim([3.3,10.4])
ax.set_xlabel('Log of Settler Mortality')
ax.set_ylabel('Average Expropriation Risk 1985-95')
ax.set_title('Figure 3: First-stage relationship between settler mortality and expropriation risk')

# First stage
########################

# The first stage involves regressing the endogenous variable (expropriation risk) on the instrument

# The instrument is the set of all exogenous variables in our model (and not just the variable we have replaced)

# Using model 1 as an example, our instrument is simply a constant and settler mortality rates

# Therefore, we will estimate the first-stage regression as:
#        avexpr_i = delta_0 + delta_1 * settler_mortality_i + u_i

# The data we need to estimate this equation is located in maketable4.dta (only complete data, indicated
# by baseco = 1, is used for estimation)

# Import and select the data
df4 = pd.read_stata('https://github.com/QuantEcon/lecture-python/blob/master/source/_static/lecture_specific/ols/maketable4.dta?raw=true')
df4 = df4[df4['baseco'] == 1]

# Add a constant variable
df4['const'] = 1

# Fit the first stage regression and print summary
results_fs = sm.OLS(df4['avexpr'],
                    df4[['const', 'logem4']],
                    missing='drop').fit()
print(results_fs.summary())

# Second stage
######################

# We need to retrieve the predicted values of avexpr_i using .predict()

# We then replace the endogenous variable avexpr_i with the predicted values avexpr_i_hat in the original linear model

# Our second stage regression is thus:
#        logpgp95_i = beta+0 + beta_1 * avexpr_i_hat + e_i
df4['predicted_avexpr'] = results_fs.predict()

results_ss = sm.OLS(df4['logpgp95'],
                    df4[['const', 'predicted_avexpr']]).fit()
print(results_ss.summary())

# IV using statsmodels
################################

# The second-stage regression results give us an unbiased and consistent estimate of the effect of institutions on
# economic outcomes. The result suggests a stronger positive relationship than what the OLS results indicated.

# NOTE that while our parameter estimates are correct, our standard errors are not and for this reason,
# computing 2SLS ‘manually’ (in stages with OLS) is not recommended.

# We can correctly estimate a 2SLS regression in one step using the linearmodels package, an extension of statsmodels
# Note that when using IV2SLS, the exogenous and instrument variables are split up in the function arguments
# (whereas before the instrument included exogenous variables)
iv = IV2SLS(dependent=df4['logpgp95'],
            exog=df4['const'],
            endog=df4['avexpr'],
            instruments=df4['logem4']).fit(cov_type='unadjusted')

print(iv.summary)

########################################################################################################################
# Exercises:
# https://python.quantecon.org/ols.html#id17
########################################################################################################################

# Exercise 78.1: Hausman test
###############################


# Exercise 78.2: Estimating OLS coefficient from scratch
##########################################################
