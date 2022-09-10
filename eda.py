import numpy as np
import pandas as pd

class analyze():
    def __init__(self):
        pass
    def show_grp(self,df,col):
        self.grp=df.groupby(col)
        return self.grp.first()

    def perform_agregate(self,funcs):       
        agg1=[] 
        if "Mean" in funcs:
            agg1.append(np.mean)
        if "Median" in funcs:
            agg1.append(np.median)
        if "Standard Deviation" in funcs:
            agg1.append(np.std)
        if "Count" in funcs:
            agg1.append(np.count)
        if "Max" in funcs:
            agg1.append(np.max)
        if "Min" in funcs:
            agg1.append(np.min)
        if "Variance" in funcs:
            agg1.append(np.var)

        
        
        group_agg=self.grp.agg(agg1).reset_index()
        return group_agg





        
        

        

        