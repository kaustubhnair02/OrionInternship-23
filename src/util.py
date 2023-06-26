
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew


def contvssale(contvar: str, df: pd.DataFrame , tarvar = "SalePrice"):
    """create subplot for continuous and target variable"""
    l = df[contvar].value_counts()
    s = df[tarvar].value_counts()

    fig, axes = plt.subplots(2,2,figsize=(13,13))
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



def remove_ngskewness(data, column_name):
    column_values = data[column_name]
    skewness = column_values.skew()
    if skewness < -0.9:
        column_values = np.log1p(column_values)

    data[column_name] = column_values
    return data

def log_transform_continuous(dataset, continuous_vars):
    transformed_dataset = dataset.copy()

    for var in continuous_vars:
        # Compute skewness
        skewness = skew(transformed_dataset[var])

        if skewness > 1:
            # Perform log transform
            transformed_dataset[var] = np.log1p(transformed_dataset[var])

    return transformed_dataset

def one_hot_encode_nominal(dataset, nominal_vars):
    encoded_dataset = dataset.copy()

    for var in nominal_vars:
        # Perform one-hot encoding
        encoded_columns = pd.get_dummies(encoded_dataset[var], prefix=var, drop_first=True)
        encoded_columns = encoded_columns.astype(int)
        encoded_dataset = pd.concat([encoded_dataset, encoded_columns], axis=1)
        encoded_dataset.drop(columns=var,inplace=True)
        
    return encoded_dataset

def label_encode_dataset(dataset, ordinal_vars):
    label_encoder = LabelEncoder()

    for col in ordinal_vars:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    return dataset