import tensorflow as tf
import numpy as np
import networkx as nx


from utils.GraphReader import Greader #For pkl
from utils.txtGraphReader import txtGreader #For txt

def main():
    mode = "train" #"test" or "train"
    filename = "./datasets/blogcatalog/blogcatalogedge.txt" 
    FOSO = "second_order" #"first_order" or "second_order"
    lr = 0.025 #learning rate
    batch_size = 128
    batch_num = 10000
    K = 5 #negative edge number
    dim = 128 #embedding dimension
    
    if mode == "train":
        train(filename, FOSO, lr, batch_size, batch_num, K, dim)
    elif mode == "test":
        test()

def train(filename, FOSO, lr, batch_size, batch_num, K, dim):
    data = txtGreader(filename, direct = False, weighted = True)
    

    nodenum = data.node_num
    model = DataFlowModel(FOSO, batch_size, K, dim, nodenum)
    
    with tf.Session() as sess:

        
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        init_learn_rate = lr
        learn_rate = lr
        
        for batchid in range(batch_num):
            #get sample
            vertex_i, vertex_j, weight = data.getbatch(batch_size, K)
            train_mat = {model.v_i: vertex_i, model.v_j: vertex_j, model.weight: weight, model.lr: learn_rate}
            
            
            #train
            if batchid % 100 != 0:
                sess.run(model.target, feed_dict = train_mat)
            
            
                #lr update
                learn_rate = max(init_learn_rate * (1 - batchid / batch_num), init_learn_rate * 0.001)
            
            #plot
            else:
                loss = sess.run(model.loss, feed_dict = train_mat)
                print("Iterations:", batchid," Loss: ", loss)
                
                
            #save
            if batchid % 1000 == 0 or batchid == (batch_num - 1):
                embedding = sess.run(model.embedding)
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                
                embeddingdict = {}
                for node in data.node:
                    index = data.indexdict[node]
                    embeddingdict[node] = normalized_embedding[index]
                
                #write
                
                f = open("Line1embd_%s.txt" % FOSO, "w")
                embdkeylist = []
                
                for key in embeddingdict.keys():
                    embdkeylist.append(int(key))
                
                embdkeylist.sort()
                
                for key in embdkeylist:
                    
                    embdwrite = embeddingdict[str(key)]
                    f.write(str(key)+" ")
                    for item in embdwrite:
                        f.write(str(item)+" ")
                    f.write("\n")

def test():
    pass
    
class DataFlowModel:
    #"persudo code from Network_Embedding_with_TensorFlow.pdf"
    def __init__(self, FOSO, batch_size, K, dim, nodenum):
        self.v_i = tf.placeholder(name="v_i", dtype=tf.int32, shape=[batch_size * (K + 1)])
        self.v_j = tf.placeholder(name="v_j", dtype=tf.int32, shape=[batch_size * (K + 1)])
        self.weight = tf.placeholder(name='weight', dtype=tf.float32, shape=[batch_size * (K + 1)])
        self.lr = tf.placeholder(name="lr", dtype=tf.float32)
        
        self.embedding = tf.get_variable('target_embedding', [nodenum, dim], initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.u_i = tf.matmul(tf.one_hot(self.v_i, depth=nodenum), self.embedding)
        
        if FOSO == "first_order":
            self.u_j = tf.matmul(tf.one_hot(self.v_j, depth=nodenum), self.embedding)
            
        elif FOSO == "second_order":
            self.context_embedding = tf.get_variable('context_embedding', [nodenum, dim], initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_j = tf.matmul(tf.one_hot(self.v_j, depth=nodenum), self.context_embedding)

        self.prod = tf.reduce_sum(self.u_i * self.u_j, axis=1)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.weight * self.prod))
        
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        self.target = self.optimizer.minimize(self.loss)
        

        
if __name__ == '__main__':
    main()