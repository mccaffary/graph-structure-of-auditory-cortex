# main graphsage-based algorithm

class graphsage_link_prediction(object):
    
    """ Extension of GraphSage to enable link prediction in neuronal population data """
    
    def __init__(self,BATCH_SIZE=2000):
        self.BATCH_SIZE = 2000
        # Auxiliary function for generating correlation matrices
        from get_corr_mats import get_corr_matrices_mu_rates
        # Auxiliary function for dividing data into sets with features + embeddings
        from divide_nodes import separate_nodes
            
    def mean_aggregator_tf(self,connMtx,embed_matrix):
        
        """
        
        Mean aggregator function - takes a list of nodes in a given batch + averages the feature representations
        of the node (neuron) + its neighbours to generate the embedding of these nodes
        
        Arguments:
        ===============================================
        connMtx:            (np.array) connectivity matrix of entire neuronal dataset
        embed_matrix:       (np.array) embedding matrix 
        
        """
        
        non_lin = lambda x: tf.tanh(x)
        next_embed = non_lin(tf.matmul(connMtx,embed_matrix))
        return next_embed
    
    
    def make_encoder(self,feats1,data,n_nodes=2):
        """
        
        First encoder (function which)
        Arguments:
        ===============================================
        feats1:
        data:
        n_nodes: 
        
        """
        activation = tf.nn.relu
        data = tf.concat([feats1,data],axis=1)
        # Fully connected neural network layer with tanh (hyperbolic tan) activation function
        x = tf.layers.dense(data,n_nodes,tf.nn.tanh,kernel_initializer=tf.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        return x
    
    
    def make_encoder2(self,feats2,data2):
        """
        
        Second encoder (function which)
        Arguments:
        ===============================================
        feats2:
        data2:
        
        """
        activation = tf.nn.relu
        data2 = tf.concat([feats2,data2],axis=1)
        # Fully connected neural network layer with relu (rectified linear unit) activation function
        x = tf.layers.dense(data2,19,tf.nn.relu,kernel_initializer=tf.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        return x
    
    
    def make_encoder3(self,feats2,data2):
        """
        
        Third encoder (function which)
        Arguments:
        ===============================================
        feats2:
        data2:
        
        """
        activation = tf.nn.relu
        data2 = tf.concat([feats2,data2],axis=1)
        # Fully connected neural network layer with relu (rectified linear unit) activation function
        x = tf.layers.dense(data2,2,tf.nn.relu,kernel_initializer=tf.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        return x
    
    
   def selector(self,data,embed_inds):

        """
        
        Selector function.
        Arguments:
        ===============================================
        data:
        embed_inds:
        
        """

        return tf.gather(data,embed_inds)
    
    def add_data(self,conns,features,xyPos):
        
        """
        
        Add data for algorithm.
        Arguments:
        ===============================================
        conns:          connectivity matrix (np.array)
        features:       features of the nodes (neurons); in this case, the tuning curves of the neurons (np.array)
        xyPos:          xy-relative (x,y) position of each neuron in the imaging field of view (measured in
                        micrometers)
        
        """

        self.conn_to_embed, self.conns_to, self.conns_from, self.conn_not_to_embed, self.feats_of_removed_nodes, self.feats_of_remaining_nodes, tmp = \
        separate_nodes(conns.copy(), features.copy().T, fraction_to_remove=0.2, seed=99) #nb fraction_remove
        self.idxs_to_remove, self.idxs_to_keep = tmp
        self.idxs_to_keep = np.array(self.idxs_to_keep)
        self.idxs_to_remove = np.array(self.idxs_to_remove)
        self.features = features
        self.conns = conns
        self.xyPos = xyPos
    
    def init_params(self,lr,search_depth=1):
        
        """
        
        Initialisation of parameters for GraphSage-based algorithm + setting of TensorFlow placeholders to be executed in run-time TensorFlow graph.
        
        Arguments:
        ===============================================
        
        search_depth:       integer (0,1,2) specifying the number of steps of neighbours through which the algorithm aggregates information
        lr:                 learning rate (usually set to 0.0005)
        
        """
        
        self.node_list1 = tf.placeholder(tf.int64,shape=self.BATCH_SIZE)
        self.node_list2 = tf.placeholder(tf.int64,shape=self.BATCH_SIZE)

        self.conn_to_embed_placeholder1 = tf.placeholder(tf.float32,shape=[None,self.conn_to_embed.shape[0]])
        self.conn_to_embed_placeholder2 = tf.placeholder(tf.float32,shape=[None,self.conn_not_to_embed.shape[0]])

        self.all_embed_feats1 = tf.placeholder(tf.float32,shape=[None,numFtrs]) #placeholder for nodes to use for embedding
        self.embed_node_feats1 = tf.placeholder(tf.float32,shape=[None,numFtrs]) #placeholder for nodes to use as features of embedded nodes

        self.all_embed_feats2 = tf.placeholder(tf.float32,shape=[None,numFtrs]) #placeholder for nodes to use for embedding
        self.embed_node_feats2 = tf.placeholder(tf.float32,shape=[None,numFtrs]) #placeholder for nodes to use as features of embedded nodes

        # x,y distance between nodes
        self.dists = tf.placeholder(tf.float32,shape=[None,2])

        self.test_conns_from = tf.placeholder(tf.int64,[self.BATCH_SIZE])
        self.test_conns_to = tf.placeholder(tf.int64,[self.BATCH_SIZE])

        # Make TensorFlow templates to facilitate variable sharing
        self.make_enc = tf.make_template('enc', self.make_encoder)
        self.make_enc2 = tf.make_template('enc2', self.make_encoder)
        
        
        
        # Search depth specification. A search depth == 0 utilises none of the GraphSage aggregation + encoding functionality, and uses node features only.
        # A search depth == 1 represents one degree of search depth from the specified node (that is, one 'leap' of neighbour aggregation + encoding)
        # A search depth == 2 represents two degrees of search depth from the specified node (that is, two 'leaps' of neighbour aggregation + encoding)
        if search_depth == 0:
            self.enc1_final = self.all_embed_feats1
            self.enc2_final = self.all_embed_feats2
            
            
            self.link_prediction = tf.layers.dense(tf.concat([
                                                  self.embed_node_feats1,
                                                  self.embed_node_feats2,
                                                  self.dists],axis=1),1,
                                       activation=tf.nn.tanh,
                                       kernel_initializer=tf.random_normal_initializer())
        else:
            if search_depth == 1:

                self.agg1 = self.mean_aggregator_tf(self.conn_to_embed_placeholder1,self.all_embed_feats1)
                self.enc1 = self.make_enc(self.all_embed_feats1,self.agg1)
                self.agg2 = self.mean_aggregator_tf(self.conn_to_embed_placeholder2,self.all_embed_feats2)
                self.enc2 = self.make_enc(self.all_embed_feats2,self.agg2)
                self.enc1_final = self.enc1
                self.enc2_final = self.enc2

            elif search_depth==2:

                self.agg1 = self.mean_aggregator_tf(self.conn_to_embed_placeholder1,self.all_embed_feats1)
                self.enc1 = self.make_enc(self.all_embed_feats1,self.agg1)
                self.agg1_ = self.mean_aggregator_tf(self.conn_to_embed_placeholder1,self.enc1)
                self.enc1_ = self.make_enc2(self.all_embed_feats1,self.agg1_)

                self.agg2 = self.mean_aggregator_tf(self.conn_to_embed_placeholder2,self.all_embed_feats2)
                self.enc2 = self.make_enc(self.all_embed_feats2,self.agg2)
                self.agg2_ = self.mean_aggregator_tf(self.conn_to_embed_placeholder2,self.enc2)
                self.enc2_ = self.make_enc2(self.all_embed_feats2,self.agg2_)

                self.enc1_final = self.enc1_
                self.enc2_final = self.enc2_
            else:
                raise("Search Depth needs to be (0,1,2)")

           # Removing enc_sel1,enc_sel2 removes GraphSage functionality 
            self.enc_sel1 = self.selector(self.enc1_final,self.node_list1)
            self.enc_sel2 = self.selector(self.enc2_final,self.node_list2)




           

            self.link_prediction = tf.layers.dense(tf.concat([self.enc_sel1,
                                                              self.enc_sel2,
                                                              self.embed_node_feats1,
                                                              self.embed_node_feats2,
                                                              self.dists],axis=1),1,
                                                   activation=tf.nn.tanh,
                                                   kernel_initializer=tf.random_normal_initializer())

    

        self.real_link = tf.placeholder(tf.float32,shape=[None,1])
        
        # Cost function
        self.cost_lp = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.real_link,predictions=self.link_prediction))
        # Optimisation using Adam
        self.optimizer_lp = tf.train.AdamOptimizer(lr).minimize(self.cost_lp)

        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        
    def batches(self,SEED=99):
        
        """
        
        Function for generating training + test batches from the complete dataset, in accordance with
        BATCH_SIZE specified above.
        
        Arguments:
        ===============================================
        SEED:       (pseudo)random seed initialisation for reproducibility
        
        """
        
        np.random.seed(SEED)
        tmp = np.array(list(itertools.product(np.random.permutation(np.arange(self.feats_of_remaining_nodes.shape[0])),
                             np.random.permutation(np.arange(self.feats_of_removed_nodes.shape[0])))))
        tmp = np.random.permutation(np.array([i for i in tmp]))

        # Generation of training batch
        nRnds = int(np.floor(tmp.shape[0]/float(self.BATCH_SIZE)))
        conns_train = tmp[:self.BATCH_SIZE*nRnds]
        self.conns_train_batch = conns_train.reshape(nRnds,self.BATCH_SIZE,2)

        # Generation of test batch
        self.conns_test = self.conns_train_batch.reshape(nRnds,self.BATCH_SIZE,2)[-1:]
        self.conns_train_batch = self.conns_train_batch[:-1]

        self.conn_to_embed2 = self.conn_to_embed.copy()
        self.conn_to_embed2 = self.conn_to_embed2/np.sum(np.abs(self.conn_to_embed2),axis=0)[:,None]
        self.conn_not_to_embed2 = self.conn_not_to_embed.copy()
        self.conn_not_to_embed2 = self.conn_not_to_embed2/np.sum(np.abs(self.conn_not_to_embed2),axis=0)[:,None]
        
    def run_graphsage(self):
        
        """
        Function which runs the GraphSage-based link prediction algorithm.
        
        """
        
        # Global variable initialisation for TensorFlow graph
        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        kk = 0

        for iterN in range(20000):
            for i in self.conns_train_batch:
                i = np.array(i)
                self.conn_test_from = i[:,0]
                self.conn_test_to = i[:,1] 
                sess.run(self.optimizer_lp, feed_dict={self.conn_to_embed_placeholder1: self.conn_to_embed2,
                                                  self.conn_to_embed_placeholder2: self.conn_not_to_embed2,
                                                  self.embed_node_feats1: self.feats_of_remaining_nodes[i[:,0],:],
                                                  self.embed_node_feats2: self.feats_of_removed_nodes[i[:,1],:],
                                                  self.all_embed_feats1: self.feats_of_remaining_nodes,
                                                  self.all_embed_feats2: self.feats_of_removed_nodes,
                                                  self.node_list1: self.conn_test_from,
                                                  self.node_list2: i[:,1],
                                                  self.dists: np.abs(self.xyPos[self.idxs_to_keep[i[:,0]]] - self.xyPos[self.idxs_to_remove[i[:,1]]])/256.,
                                                  self.real_link: self.conns_to[i[:,0],i[:,1]].reshape(self.BATCH_SIZE,1)})



            if np.remainder(iterN,20)==0:
                i = np.concatenate(self.conns_test)
                real_links = self.conns_to[i[:,0],i[:,1]].reshape(i.shape[0],1).flatten()
                sel_nodes = np.zeros(self.conn_to_embed.shape[0])
                sel_nodes[i[:,0]] = 1

                predictions = sess.run(self.link_prediction,feed_dict={self.conn_to_embed_placeholder1: self.conn_to_embed2,
                                                                  self.conn_to_embed_placeholder2: self.conn_not_to_embed2,
                                                                  self.embed_node_feats1: self.feats_of_remaining_nodes[i[:,0],:],
                                                                  self.embed_node_feats2: self.feats_of_removed_nodes[i[:,1],:],
                                                                  self.all_embed_feats1: self.feats_of_remaining_nodes,
                                                                  self.all_embed_feats2: self.feats_of_removed_nodes,
                                                                  self.node_list1: i[:,0],
                                                                  self.node_list2: i[:,1],
                                                                  self.dists: np.abs(self.xyPos[self.idxs_to_keep[i[:,0]]] - self.xyPos[self.idxs_to_remove[i[:,1]]])/256.,
                                                                  self.real_link: self.conns_to[i[:,0],i[:,1]].reshape(self.BATCH_SIZE,1)})



                print('CC %s' %np.corrcoef(predictions.flatten(),real_links.flatten())[0,1])
                print('MSE %s' %np.sum(np.abs(predictions.flatten()-real_links.flatten())))

            kk += 1
