import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.formula.api as sm


#Indlæs datasæt. Husk at opdatere stien når du skifter mellem computere.
df = pd.read_excel(r"dataproject\Combined.xlsx")

df.head()
#We see in the head that there is a lot of missing data for Aruba. If we delve deeper into the dataset we will find that this is also the case for many other countries in the 20th century. 

#Renaming the variables
df.rename(columns={'gdppc':'GDPPerCapita'}, inplace=True)
df.rename(columns={'csh_i':'Investment'}, inplace=True)
df.rename(columns={'popgr':'PopulationGrowth'}, inplace=True)
df.rename(columns={'csh_g':'GovernmentExpenditure'}, inplace=True)
df.rename(columns={'gdpgr':'GDPGrowth'}, inplace=True)
df.rename(columns={'pl_i':'PPI'}, inplace=True)


#We're focusing on the last 5 years of data. Most countries are covered and we don't have to constantly be aware that a financial crisis happened, even though the world was still in aftermath
#Dropping 1950-2010 due to inconsistent data on majority of the variables
indexNames = df[ (df['year'] >= 1950) & (df['year'] <= 2009) ].index   
reduced_df = df
reduced_df.drop(indexNames , inplace=True)
reduced_df.head()

#Swarm plots
def year_plot(y="Investment"): 
    
    fig = plt.figure(figsize=(9,3))
    fig1 = fig.add_subplot(1,1,1)
    fig1 = sns.swarmplot(x ="year", y = y, data=reduced_df)
    if y == "GovernmentExpenditure": 
        fig1.set_title("Figure 1: Government expenditure")
        fig1.set_ylabel("Secondary Education")
    elif y == "Investment": 
        fig1.set_title("Figure 2: Investment")
        fig1.set_ylabel("Investment")
    elif y == "PoliticalStability": 
        fig1.set_title("Figure 3: PoliticalStability")
        fig1.set_ylabel("Political Stability")
    fig1.set_xlabel("year")
    

year_plot("GovernmentExpenditure")
year_plot()
year_plot("PoliticalStability")
#Calculating means for 2010-2014
means = reduced_df.groupby('country')['GDPPerCapita', 'PopulationGrowth', 'Investment', 'GovernmentExpenditure', 'PPI', 'GDPGrowth', 'pri', 'sec', 'GovernmentEffectiveness', 'PoliticalStability'].mean()
means.head()

#Check maximums
means[means['Investment']==means['Investment'].max()]
means[means['GovernmentExpenditure']==means['GovernmentExpenditure'].max()]
#Dropping the large variables, as they're unnaturally high.
means = means.drop("Turks and Caicos Islands", axis=0)
means = means.drop('Cayman Islands', axis = 0)

#Send to excel so I can verify that there was no screwups
means.to_excel("2010-2014.xlsx")

#Jointplots
sns.jointplot(x="PoliticalStability", y="Investment", data=means, kind="reg")
#Working around position of plt.title
plt.subplots_adjust(top=0.9)
plt.suptitle('Figure 4: Political stability on investments', fontsize = 12)
plt.show()

sns.jointplot(x="PoliticalStability", y="sec", data=means, kind="reg")
plt.subplots_adjust(top=0.9)
plt.suptitle('Figure 5: Political stability on secondary education', fontsize = 12)
plt.show()

#Defining the title for our correlation matrices
def title_number(x):
    title=f"Figure {x}: Correlation of growth factors"
    return title
#Correlation Matrix
def corr_matrix(data):
    corr = data[['GDPPerCapita', 'PopulationGrowth', 'Investment', 'sec', 'pri', 'GDPGrowth', 'PoliticalStability']].corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, fmt='.2f')
    plt.show()

plt.title(title_number(6))
corr_matrix(means)

#OLS
result_basic = sm.ols(formula="GDPGrowth ~ PoliticalStability + GovernmentExpenditure + Investment + sec ", data=means).fit()
print(result_basic.summary())

#OLS
result_basic = sm.ols(formula="Investment ~  sec + PoliticalStability + GovernmentExpenditure", data=means).fit()
print(result_basic.summary())

#OLS
result_investment = sm.ols(formula = "sec ~ Investment  + PoliticalStability + GovernmentExpenditure", data=means).fit()
print(result_investment.summary())

#Dropping OPEC nations
means_without_oil = means.drop(['Algeria', 'Indonesia', 'Iran', 'Iraq', 'Kuwait', 'Venezuela', 'Ecuador', 'Congo, D.R.'])
#Gabon, Nigeria, Oman and Saudi Arabia are already dropped from dropping empty variables

#Correlation matrix
plt.title(title_number(7))
corr_matrix(means_without_oil)

#OLS
result_basic_without_oil = sm.ols(formula="GDPGrowth ~  Investment + sec + PoliticalStability + GovernmentExpenditure", data=means_without_oil).fit()
print(result_basic_without_oil.summary())

result_basic_without_oil = sm.ols(formula="Investment ~  sec + PoliticalStability + GovernmentExpenditure", data=means_without_oil).fit()
print(result_basic_without_oil.summary())

result_basic_without_oil = sm.ols(formula="sec ~  Investment + PoliticalStability + GovernmentExpenditure", data=means_without_oil).fit()
print(result_basic_without_oil.summary())
