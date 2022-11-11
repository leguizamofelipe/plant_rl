import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Reshape, Lambda, Dropout
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
from keras import backend as K

Adam = tf.keras.optimizers.Adam(learning_rate = 0.01)

############################################# ICM Stuff #################################################

class ForwardModel(tf.keras.Model):
    ## NN
    ## I: f_t (feature vector of state t), a_t
    ## O: s_t+1 (next state)

    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.forward1 = Dense(input_shape[0], activation='relu')
        self.forward2 = Dense(50, activation='relu')
        self.forward3 = Dense(output_shape[0], activation='linear')

    def call(self, f_t, a_t):
        input=concatenate([f_t, a_t])
        x = self.forward1(input)
        x = self.forward2(x)
        return self.forward3(x)

class InverseModel(tf.keras.Model):
    ## NN
    ## I: s_t, s_t+1
    ## O: a_t

    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.forward1 = Dense(input_shape[0]*2, activation='relu')
        self.forward2 = Dense(output_shape[0], activation='sigmoid')

    def call(self, s_t, s_t1):
        input=concatenate([s_t, s_t1])
        x = self.forward1(input)
        return self.forward2(x)

class FeatureExtractor(tf.keras.Model):
    ## NN
    ## I: s_t
    ## O: f_t

    def __init__(self, input_shape) -> None:
        super().__init__()
        self.feat1 = Dense(input_shape[0], activation="relu")
        self.feat2 = Dense(12, activation="relu")
        self.feat3 = Dense(input_shape[0], activation="linear", name = 'feature')

    def call(self, s_t):
        x = self.feat1(s_t)
        x = self.feat2(x)
        return self.feat3(x)

class ICMModel():
    def __init__(self, state_shape, action_shape) -> None:
        # Declare all NNs
        self.forward_net = ForwardModel((state_shape[0]+action_shape[0],), state_shape)
        self.inverse_net = InverseModel(state_shape, action_shape)
        self.feature_extractor = FeatureExtractor(state_shape)

        self.forward_net.compile(optimizer=Adam, loss='mse')
        self.inverse_net.compile(optimizer=Adam, loss='mse')
        self.feature_extractor.compile(optimizer=Adam, loss='mse')

        s_t=Input(shape=state_shape, name="state_t") # (2,)
        s_t1=Input(shape=state_shape, name="state_t1") # (2,)
        a_t=Input(shape=action_shape, name="action") # (3,)

        reshape=Reshape(target_shape= state_shape)

        self.lmd = 0.99
        self.beta = 0.01

        fv_t=self.feature_extractor(reshape(s_t))
        fv_t1=self.feature_extractor(reshape(s_t1))

        a_t_hat=self.inverse_net(fv_t, fv_t1)
        fv_t1_hat=self.forward_net(fv_t, a_t)

        # the intrinsic reward refelcts the diffrence between
        # the next state versus the predicted next state
        # $r^i_t = \frac{\nu}{2}\abs{\hat{s}_{t+1}-s_{t+1})}^2$
        int_reward=Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1),
                     output_shape=(1,),
                     name="reward_intrinsic")([fv_t1, fv_t1_hat])

        #inverse model loss
        inv_loss=Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), 
                                         axis=-1),
                    output_shape=(1,))([a_t, a_t_hat])

        # combined model loss - beta weighs the inverse loss against the
        # rwd (generate from the forward model)
        loss=Lambda(lambda x: self.beta * x[0] + (1.0 - self.beta) * x[1],
                    output_shape=(1,))([int_reward, inv_loss])
        #
        # lmd is lambda, the param the weights the importance of the policy
        # gradient loss against the intrinsic reward
        rwd=Input(shape=(1,))
        loss=Lambda(lambda x: (-self.lmd * x[0] + x[1]), 
                    output_shape=(1,))([rwd, loss])

        self.built_model = Model([s_t, s_t1, a_t, rwd], loss)

    

