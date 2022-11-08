from download import TemporalDataSets


if __name__ == "__main__":

    '''
    #! step 1: download dataset from url, select specific dataset using str name 
    '''
    input_list = "enron"
    example_data = TemporalDataSets(data_list=input_list)



    #* to do, make a list dataset function --> Abu 

    #? should be called download_all, this will download all TG datasets  --> Abu
    #example_data.redownload()



    '''
    #! step 2: process the datasets in a TGL friendly way and act as input to ML methods
    '''
    example_data.process()




    '''
    #! step 3: able to retrieve train, validation, test data correctly
    the split should be deterministic if using the default split
    '''
    # training_data = example_data.train_data 
    # test_data = example_data.test_data 
    # val_data = example_data.val_data 




    
    '''
    #! step 4: 
    '''
