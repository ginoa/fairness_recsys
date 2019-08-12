import tensorflow as tf
import numpy as np

class Autoencoder:
    def __init__(self, input_size, hidden_layer_size):
        # Input placeholder, "None" here means any size e.g. (13,input_size), (420,input_size), etc.
        self.X = tf.placeholder(tf.float32, shape=(None, input_size))
        
        # Input to hidden (input_size -> hidden_layer_size)
        self.W1 = tf.Variable(tf.random_normal(shape=(input_size,hidden_layer_size)))
        self.b1 = tf.Variable(np.zeros(hidden_layer_size).astype(np.float32))
        
        # Hidden -> output (hidden_layer_size -> input_size)
        self.W2 = tf.Variable(tf.random_normal(shape=(hidden_layer_size,input_size)))
        self.b2 = tf.Variable(np.zeros(input_size).astype(np.float32))
        
        # hidden layer
        self.Z = tf.nn.relu( tf.matmul(self.X, self.W1) + self.b1 )
        
        # output layer
        self.X_hat = tf.nn.sigmoid(tf.matmul(self.Z, self.W2) + self.b2)
        
        # Define loss function
        self.loss = tf.losses.mean_squared_error(
            labels=self.X,
            predictions=self.X_hat
        )
                
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.005).minimize(self.loss)
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.get_default_session() 
        if(self.sess == None):
            self.sess = tf.Session()
        self.sess.run(self.init_op)
        
        
    def fit(self, X, epochs=10, bs=64):
        n_batches = len(X) // bs
        print("Training {} batches".format(n_batches))
        
        for i in range(epochs):
            print("Epoch: ", i)
            X_perm = np.random.permutation(X)
            for j in range(n_batches):
                batch = X_perm[j*bs:(j+1)*bs]
                _, _ = self.sess.run((self.optimizer, self.loss),
                                    feed_dict={self.X: batch})


    def predict(self, X):
        return self.sess.run(self.X_hat, feed_dict={self.X: X})
        
    def encode(self, X):
        return self.sess.run(self.Z, feed_dict={self.X: X})
    
    def decode(self, Z):
        return self.sess.run(self.X_hat, feed_dict={self.Z: Z})
    
    def terminate(self):
        self.sess.close()
        del self.sess