import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

# from Config import CFG

def create_fold(df,tar_col,n_folds,task='classification'):
    df['kfold']=-1
    df=df.sample(frac=1).reset_index(drop=True)
    if task=='classification':
        y=df[tar_col].values
        skf=StratifiedKFold(n_splits=n_folds)
        for fold,(train_,valid_) in enumerate(skf.split(X=df,y=y)):
            df.loc[valid_,'kfold']=fold
    elif task=='regression':
        kf=KFold(n_splits=n_folds)
        for fold,(train_,valid_) in enumerate(kf.split(X=df)):
            df.loc[valid_,'kfold']=fold
    else:
        #do nothing
        pass
    return df

# if __name__=="__main__":
#     df=pd.read_csv(CFG.DATA_DIR+CFG.file_nm)
#     print(df.shape)
#     df=create_fold(df,tar_col='ACTION',n_folds=CFG.n_fold,task='classification')
#     df.to_csv(CFG.OUTPUT_DIR+CFG.fold_file_nm,index=False)

