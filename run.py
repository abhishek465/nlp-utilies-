from Config import CFG
import pandas as pd

from lm_train import *

def run(df,fold=0):
    train=df[df.kfold!=fold]
    valid=df[df.kfold==fold]

    # Fit language model
    data_lm, learn_lm = train_lm(df=train, output_name=CFG.OUTPUT_NAME, model_dir=CFG.LANGUAGE_MODEL_DIR)

    # Train lesson classifier
    learn_classifier = train_classifier(df_train=train,df_test=valid, data_lm=data_lm, output_name=CFG.OUTPUT_NAME+'_fold_'+str(fold), 
                                        n_epoch=CFG.NUM_EPOCH, lm_model_dir=CFG.LANGUAGE_MODEL_DIR, 
                                        class_model_dir=CFG.CLASSIFIER_MODEL_DIR)



if __name__=="__main__":
    df=pd.read_csv(CFG.DATA_DIR+CFG.fold_file_nm)
    for i in range(CFG.n_fold):run(df,i)
    