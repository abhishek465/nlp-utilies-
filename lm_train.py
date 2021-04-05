from fastai.text.all import *

LANGUAGE_MODEL_DIR = 'models/'
CLASSIFIER_MODEL_DIR = 'classifier_models/'
NUM_EPOCH = 30
OUTPUT_NAME = 'trained_model_23March_v1'


def train_lm(df, output_name=OUTPUT_NAME, lm_model_dir=LANGUAGE_MODEL_DIR):
    '''Fit language model.

    Encoder is saved at LANGUAGE_MODEL_DIR.

    Params
    ------
    df : pd.DataFrame
        Dataframe of paragraphs. Columns: paragraphs, isLesson
    
    output_name : str
        Name of language model to be saved
    
    lm_model_dir : str
        Directory where language model is saved. 

    Returns
    -------
    Text block used by fastai.
    language_model_learner fastai object.

    '''
    data_lm = (TextList.from_df(df)
                .split_by_rand_pct(0.15)
                .label_for_lm()
                .databunch(bs=32)
              )

    learn = language_model_learner(
                                    data_lm, 
                                    AWD_LSTM, 
                                    drop_mult=0.5,
                                    model_dir=lm_model_dir
                                    )

    learn.lr_find()
    learn.recorder.plot(suggestion = True)
    min_grad_lr = learn.recorder.min_grad_lr
    # To use fixed learning rate:
    # min_grad_lr = 7.59e-7

    # Save language model
    learn.fit_one_cycle(1, min_grad_lr)
    learn.save_encoder(output_name)
    
    return data_lm, learn


def train_classifier(df_train, df_test, data_lm, output_name=OUTPUT_NAME, n_epoch=NUM_EPOCH, lm_model_dir=LANGUAGE_MODEL_DIR, class_model_dir=CLASSIFIER_MODEL_DIR):
    '''Train lesson classifier from paragraphs using LSTMs.
    
    Trained model is saved as pickle at CLASSIFIER_MODEL_DIR.

    Params
    ------
    df_train, df_test : pd.DataFrame
        Dataframes with labels.

    data_lm : callable object
        language_model_learner fastai object from train_lm()

    output_name : str
        Name of trained model to be saved. Can be the same as train_lm()'s output_name since the models will be saved in different directories.

    n_epoch : int

    lm_model_dir : str
        Language model directory.

    class_model_dir : str
        Classifier model directory.


    Returns
    -------
    text_classifier_learner fastai object.

    '''
    data_clas = (TextList.from_df(df_train, vocab=data_lm.vocab)
             .split_by_rand_pct(0.1)
             .label_from_df('isLesson')
             .add_test(TextList.from_df(df_test[['paragraph']], vocab=data_lm.vocab))
             .databunch(bs=32))

    learn_classifier = text_classifier_learner(
        data_clas, 
        AWD_LSTM, 
        drop_mult=0.5, 
        metrics=[accuracy, Precision(), Recall(), FBeta(beta=1)], model_dir=class_model_dir)

    # Load encoder 
    learn_classifier.load_encoder(f"../{lm_model_dir}{output_name}")

    # Find optimal learning rate
    learn_classifier.lr_find()
    learn_classifier.recorder.plot(suggestion = True)
    min_grad_lr = learn_classifier.recorder.min_grad_lr
    # To use fixed learning rate:
    # min_grad_lr = 1.74e-3 # override

    # Train model
    learn_classifier.fit_one_cycle(n_epoch, min_grad_lr)
    
    # Save model
    model_name = f"{class_model_dir}{output_name}_{n_epoch}_epochs.pkl"
    learn_classifier.export(model_name) 
    print(f"{model_name} saved.")
    
    return learn_classifier