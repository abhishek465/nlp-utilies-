from Config import CFG
from create_fold import *
import pandas as pd

if __name__=="__main__":
    df=pd.read_csv(CFG.DATA_DIR+CFG.file_nm)
    print(df.shape)
    df=create_fold(df,tar_col='ACTION',n_folds=CFG.n_fold,task='classification')
    df.to_csv(CFG.OUTPUT_DIR+CFG.fold_file_nm,index=False)