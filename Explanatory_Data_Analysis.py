#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EDA(object):
    
    def __init__(self,data):
        self.data=data
        
    def missing_check(self,var_lst):
        miss=[]
        for i in range(len(var_lst)):
            missing_count=sum(self.data[var_lst[i]].isnull())
            miss.append([var_lst[i],missing_count])
        miss_tbl=pd.DataFrame(miss,columns=['','missing_counts'])
        miss_tbl=miss_tbl.set_index('')
        total=len(self.data)
        miss_tbl['missing_rate']=miss_tbl['missing_counts']/total
        miss_tbl['missing_rate']=pd.Series(['{0:.2f}%'.format(val*100) for val in  miss_tbl['missing_rate']],index=miss_tbl.index)
        print("####################  Variable Missing Check ####################")
        print(miss_tbl)
        
    def value_check(self,var_lst):
        miss=[]
        for i in range(len(var_lst)):
            print('')
            print(str(var_lst[i]))
            print(self.data[var_lst[i]].value_counts(dropna=False).sort_index())

    def uni_plot(self,uni_var):
        
        for feature in uni_var:
            ax = self.data[feature].plot.hist(bins=10, align='mid')
            ax.set_title(feature)
            plt.ylabel("Frequency")
            plt.title(feature)
            plt.show()
            
            # Create the table for the feature
            var = self.data[feature]
            varValue = var.value_counts().sort_index()
            print("{}: \n {}".format(feature,varValue))     
            
    
    def bi_pie_plot(self,bi_var,target):


        for feature in bi_var:
            subplot_num=int(np.ceil(np.sqrt(len(self.data[feature].unique()))))
            i=0
            j=0

            plt.figure(figsize=(10,10), dpi=100)
            fig, axs = plt.subplots(subplot_num,subplot_num)

            for value in list(np.sort(self.data[feature].unique())):
                if (j<subplot_num):

                    var_df = self.data[self.data[feature] == value]
                    value_df=pd.DataFrame(var_df[target].value_counts().rename_axis(target).reset_index(name='counts'))
                    value_df[target] = value_df[target].map({0:'No Survived',1:'Survived'})

                    axs[i,j].pie(value_df['counts'],labels=value_df[target], autopct = '%1.1f%%')
                    axs[i,j].set_xlabel(str(feature+'='+str(value)))
                    j+=1
                else:
                    i+=1
                    j=0
                    var_df = self.data[self.data[feature] == value]
                    value_df=pd.DataFrame(var_df[target].value_counts().rename_axis(target).reset_index(name='counts'))
                    value_df[target] = value_df[target].map({0:'No Survived',1:'Survived'})

                    axs[i,j].pie(value_df['counts'],labels=value_df[target], autopct = '%1.1f%%')
                    axs[i,j].set_xlabel(str(feature+'='+str(value)))
                    j+=1
            if i >=1:
                for k in range(j,subplot_num):
                    fig.delaxes(axs[i][k])
            else:
                for k in range(subplot_num):
                    fig.delaxes(axs[1][k])

            plt.tight_layout()
            plt.show()


            
    def corr_plot(self,corr_df):
        
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14,12))
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        sns.heatmap(corr_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
                    square=True, cmap=colormap, linecolor='white', annot=True)