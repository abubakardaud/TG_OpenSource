



def generate_split_index(dataset_name, val_ratio, test_ratio, different_new_nodes_between_val_and_test=False,
                         rnd_seed=2020, save_indices=False):
    """
    only generates the indices of the data in train, validation, and test split
    """
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values
    
    random.seed(rnd_seed)
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)
    
    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
    
    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    
    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    
    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
                      
    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set
    
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time
    
    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])
    
        edge_contains_new_val_node_mask = np.array(
                [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
                [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
            new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
            new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)
​
    if save_indices:
        print("INFO: Saving index files for {}...".format(dataset_name))
        np.save('./data/split_index/ml_{}_train_index.npy'.format(dataset_name), train_mask)
        np.save('./data/split_index/ml_{}_val_index.npy'.format(dataset_name), train_mask)
        np.save('./data/split_index/ml_{}_test_index.npy'.format(dataset_name), train_mask)
        np.save('./data/split_index/ml_{}_new_node_val_index.npy'.format(dataset_name), train_mask)
        np.save('./data/split_index/ml_{}_new_node_test_index.npy'.format(dataset_name), train_mask)
​
    return train_mask, val_mask, test_mask, new_node_val_mask, new_node_test_mask