
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def contvssale(contvar: str, df: pd.DataFrame , tarvar = "SalePrice"):
    """create subplot for continuous and target variable"""
    l = df[contvar].value_counts()
    s = df[tarvar].value_counts()

    fig, axes = plt.subplots(2,2,figsize=(10,10))
    #fig.tight_layout(pad=5,w_pad=5)
    fig.suptitle(f"{contvar} vs {tarvar}")

    axes[0,0].set_title(f"{contvar} histogram")
    axes[0,1].set_title(f"{tarvar} boxplot")
    axes[1,0].set_title(f"{contvar} boxplot")
    axes[1,1].set_title(f"{contvar} vs {tarvar} regplot")


    sb.histplot(ax = axes[0,0], data=df,x=contvar)
    sb.boxplot(ax = axes[0,1],data = df, x=tarvar)
    sb.boxplot(ax = axes[1,0],data = df,x=contvar)
    sb.regplot(ax = axes[1,1],data = df,x=contvar,y=tarvar,scatter_kws={"color": "red"}, line_kws={"color": "blue"})

def catgvssale(catgvar: str, df: pd.DataFrame , tarvar = "SalePrice"):
    """create subplot for categorical and target variable"""

    q = df[catgvar].value_counts()
    s = df[tarvar].value_counts()

    fig, axes = plt.subplots(1,3,figsize=(18,12), sharey = True)
    # fig.tight_layout(pad=3,h_pad=5)
    fig.suptitle(f"{catgvar} vs {tarvar}")


    axes[0].set_title(f"{catgvar}  bargraph")
    sb.countplot(ax = axes[0], y = catgvar, data = df)



    axes[1].set_title(f"{catgvar} & {tarvar} bargraph")
    sb.barplot(ax = axes[1],data = df, x=tarvar,y=catgvar, orient= "h", estimator=sum)

    axes[2].set_title( f"{tarvar} boxplot ")
    sb.boxplot(ax = axes[2],data = df, x=tarvar,y=catgvar, orient= "h")

def check_column_skewness(df, column_name):
    skewness = df[column_name].skew()
    return skewness      

def contvscont(contvar: str, df: pd.DataFrame , tarvar: str):
    """create subplot for continuous and target variable"""
    l = df[contvar].value_counts()
    s = df[tarvar].value_counts()

    fig, axes = plt.subplots(2,2,figsize=(10,10))
    #fig.tight_layout(pad=5,w_pad=5)
    fig.suptitle(f"{contvar} vs {tarvar}")

    axes[0,0].set_title(f"{contvar} histogram")
    axes[0,1].set_title(f"{tarvar} boxplot")
    axes[1,0].set_title(f"{contvar} boxplot")
    axes[1,1].set_title(f"{contvar} vs {tarvar} regplot")


    sb.histplot(ax = axes[0,0], data=df,x=contvar)
    sb.boxplot(ax = axes[0,1],data = df, x=tarvar)
    sb.boxplot(ax = axes[1,0],data = df,x=contvar)
    sb.regplot(ax = axes[1,1],data = df,x=contvar,y=tarvar,scatter_kws={"color": "red"}, line_kws={"color": "blue"}) 


def remove_skewness(data, column_name):
    
    column_values = data[column_name]
    skewness = column_values.skew()
    if abs(skewness) > 1:
        
        column_values = np.log1p(column_values)

    
    data[column_name] = column_values
    return data     

def plot_contv(contvar: str, df: pd.DataFrame):
    """create subplot for continuous variable"""
    
    

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    #fig.tight_layout(pad=5,w_pad=5)
    fig.suptitle(f"{contvar}")

    axes[0].set_title(f"{contvar} histogram")
    axes[1].set_title(f"{contvar} boxplot")
    


    sb.histplot(ax = axes[0], data=df,x=contvar)
    
    sb.boxplot(ax = axes[1],data = df,x=contvar)

