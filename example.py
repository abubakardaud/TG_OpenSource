from download import TemporalDataSets


if __name__ == "__main__":

    '''
    #! step 1: download dataset from url, select specific dataset using str name 
    '''
    input_list = "canparl" #"enron"
    enron_tgb = TemporalDataSets(data_list=input_list)
    
    #* to do, make a list dataset function --> Abu  DONE 
    #? should be called download_all, this will download all TG datasets  --> Abu DONE 
    #example_data.download_all()


    '''
    #! step 2: process the datasets in a TGL friendly way and act as input to ML methods
    '''
    enron_tgb.process()
    train_data = enron_tgb.train_data    
    test_data = enron_tgb.test_data
    val_data = enron_tgb.val_data
    




    '''
    #! step 3: able to retrieve train, validation, test data correctly
    the split should be deterministic if using the default split
    '''
    # training_data = enron_tgb.train_data
    # test_data = enron_tgb.test_data 
    # val_data = enron_tgb.val_data 



    
    



    
    '''
    #! step 4: Evaluate is able to run for Random, Historical, and Inductive setting 
    '''

    # evaluator = Evaluator(name = d_name)
    # print(evaluator.expected_input_format) 
    # print(evaluator.expected_output_format)
    # # input_dict = {"y_true": y_true, "y_pred": y_pred}
    # result_dict = evaluator.eval(input_dict) 

