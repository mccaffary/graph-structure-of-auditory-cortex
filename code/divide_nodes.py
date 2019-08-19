def remove_test_conns(myconns,idxs):

    """
    This function removes connections of nodes to be embedded
    
    Arguments:
    =====================
    
    myconns:    np.array
                connectivity matrix
                
    idxs:       list || np.array
                indices to remove nodes
    
    """

    conns_not_to_embed = myconns[idxs][:,idxs]
    
    # Connections of the nodes we will embed
    conn_to_embed = np.delete(np.delete(myconns, idxs, 0),idxs, 1)
    idxs2 = np.array([i for i in range(myconns.shape[0]) if i not in idxs])

    # Connections to the nodes we will embed
    conns_to = myconns[:,idxs]
    conns_to = conns_to[idxs2]
    
    # Connections from the nodes we will embed
    conns_from = myconns[idxs][:,idxs2]

    return conn_to_embed, conns_to, conns_from, conns_not_to_embed


def remove_test_feats(feat_data,idxs):

    """
    This function removes features of nodes and collects the features of the remaining nodes
    Arguments:
    =====================
    
    feat_data:    np.array
                  feature data of nodes
                
    idxs:         list || np.array
                  indices to remove nodes
    
    """

    feats_of_remaining_nodes = np.delete(feat_data, idxs, 1)
    feats_of_removed_nodes = feat_data[:,idxs]
    
    return feats_of_removed_nodes.T, feats_of_remaining_nodes.T


def separate_nodes(connMatrix,feat_data,fraction_to_remove=.2,seed=99):
    
    """ 
    This function separates the nodes into sets with embeddings and features
    
    Arguments:
    =====================
    
    feat_data:              np.array
                            feature data of nodes
                
    connMatrix:             np.array
                            connectivity matrix
                  
    fraction_to_remove:     fraction of nodes to remove (for the test set)
    seed:                   (pseudo)random seed initialisation for reproducibility
    
    """
    
    np.random.seed(seed)
    
    numNodes = connMatrix.shape[0]
    allN = np.random.permutation(np.arange(numNodes))
    
    num_to_remove = int(np.round(numNodes*fraction_to_remove))

    # Indices of the nodes to keep and those to remove
    idxs_to_remove = sorted(allN[:num_to_remove])
    idxs_to_keep = sorted(allN[num_to_remove:])
    
    conn_to_embed, conns_to, conns_from, conns_not_to_embed = remove_test_conns(connMatrix, idxs_to_remove)
    
    feats_of_removed_nodes, feats_of_remaining_nodes = remove_test_feats(feat_data, idxs_to_remove)

    return conn_to_embed, conns_to, conns_from, conns_not_to_embed, feats_of_removed_nodes, feats_of_remaining_nodes, (idxs_to_remove,idxs_to_keep)
    
