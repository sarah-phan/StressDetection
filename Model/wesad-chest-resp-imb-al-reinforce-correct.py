# import stumpy
import os
os.environ['PYTHONHASHSEED']= '0'
import numpy as np
np.random.seed(1)
import random as rn
rn.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras import callbacks
#from tensorflow.keras import optimizers
#%matplotlib inline
import pandas as pd
import argparse
import seaborn as sns
import csv
from numpy import save,load
import time
#import coremltools
from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import preprocessing as pre
from scipy.stats import dirichlet
from matplotlib import pyplot as plt
import math
import sys
import glob
from keras.layers import *
from keras.models import *
import keras.backend as K
#from keras.callbacks import ModelCheckpoint
#from layers import AttentionWithContext, Addition
from collections import deque
import random
#from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
#import mass_ts as mts


#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, Reshape, LSTM
#from keras.utils import np_utils
#from keras.callbacks import ModelCheckpoint
#from keras import optimizers


import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#=========================================#
#                                         #
#             BEGIN                       #
#                                         #
#=========================================#
# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
#sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
#print('keras version ', keras.__version__)
# Same labels will be reused throughout the program

LABELS = [1, 2, 3]
#LABELS = ["Desk Work","Eating/Drinking","Movement","Sport","unknown"]
# The number of steps within one time segment
TIME_PERIODS = 40
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40
N_FEATURES = 1
RESAMPLE = 175
BATCH_SIZE = 64
EPOCHS = 4
AL = 5
EPISODES = 1
amount_sample = 1
ALPHA = 1e-6
#threshold = 60
iter = 5
subject = 16

path = "/results/wesad/test-S" + str(subject) + "/10s/chest/RESP/lstm_bias_imb_al_rein_correct/lstm_imb_al_rein_correct_" + \
       str(iter) + "_" + str(AL)+ "_" + str(amount_sample) + "_" + str(ALPHA) + ".txt"
cwd = os.getcwd()
report_dir = cwd + path
WEIGHT_CLASS = []
num_classes = len(LABELS) - 1
print(num_classes)

def write_list_to_file(guest_list, filename):
    """Write the list to csv file."""
    with open(filename, "w") as output:
        writer = csv.writer(output, delimiter = ',', lineterminator='\n')
        for row in enumerate(guest_list):
            writer.writerows([row])

def read_data(cwd, filepath):
    #print('====loading data====')
    #df = pd.concat(map(pd.read_pickle, glob.glob(os.path.join('', filepath))))
    # Parse paths
    print(filepath)
    os.chdir(cwd + "/"+ filepath)
    extension = 'pkl'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    print(all_filenames)
    # combine all files in the list
    frames = []
    label = []
    ECG = []
    EDA = []
    BVP = []
    EMG = []
    RESP = []
    TEMP = []
    for f in all_filenames:
        df = pd.read_pickle(f)
        print(df)
        #np.set_printoptions(threshold=sys.maxsize)
        list_label = df['label'].tolist()
        #print('labels = ', list_label)
        #list_chest_ECG = df['signal']['chest']['ECG'].tolist()
        #list_chest_EDA = df['signal']['chest']['EDA'].tolist()
        #list_chest_EMG = df['signal']['chest']['EMG'].tolist()
        list_chest_RESP = df['signal']['chest']['Resp'].tolist()
        #list_chest_TEMP = df['signal']['chest']['Temp'].tolist()
        #list_wrist_EDA = df['signal']['wrist']['EDA'].tolist()
        #list_wrist_BVP = df['signal']['wrist']['BVP'].tolist()
        #print(f)
        #print('length of BVP: ', len(list_wrist_BVP))
        #print('length of EDA: ', len(list_wrist_EDA))
        #print('length of EDA: ', len(list_wrist_EDA))

        print('length of label: ', len(list_label))
        for i in range(0, int(len(list_chest_RESP)/RESAMPLE)):
            #ECG.append(list_chest_ECG[i*RESAMPLE][0])
            RESP.append(list_chest_RESP[i*RESAMPLE][0])
            #EDA.append(list_chest_EDA[i*RESAMPLE][0])
            #EMG.append(list_chest_EMG[i*RESAMPLE][0])
            #TEMP.append(list_chest_TEMP[i*RESAMPLE][0])
            #EDA.append(list_wrist_EDA[i][0])
            #BVP.append(list_wrist_BVP[i][0])
        print(int(len(list_label)/RESAMPLE))
        #print(list_label)
        count_label = [0, 0]
        for j in range(0, int(len(list_label)/RESAMPLE)):
            if list_label[j*RESAMPLE] == 2:
                count_label[1] = count_label[1] + 1
            else:
                count_label[0] = count_label[0] + 1

            label.append(list_label[j*RESAMPLE])
        print(count_label)
        print('length of RESP: ', len(RESP))
        print('length of label: ', len(label))
        #plt.plot(ECG)
        #plt.show()
        df_fn = pd.DataFrame(list(zip(label,
                                      #ECG,
                                      RESP,
                                      #EDA,
                                      #BVP
                                      #EMG,
                                      #TEMP
                                      )),
                             columns=['label',
                                      #'ECG',
                                      'RESP',
                                      #'EDA',
                                      #'BVP'
                                      #'EMG',
                                      #'TEMP'
                                      ])
        frames.append(df_fn)
    df_frames = pd.concat(frames)

    #df = pd.read_pickle(filepath)
    #
    #print(df['label'])
    #print(df['signal']['chest']['ECG'])
    #print(df['signal']['chest']['EDA'])
    #print(df['signal']['chest']['EMG'])


    #print(len(df['signal']['chest']['RESP']))
    #print(len(df['signal']['chest']['TEMP']))
    #print(df['label'])

    #df = pd.read_csv(filepath, skip_blank_lines=True, na_filter=True).dropna()
    #df = df.drop(labels=['timestamp'], axis=1)
    #df.dropna(how='any', inplace=True)
    #cols_to_norm = ['mean_x_p', 'mean_y_p', 'mean_z_p', 'var_x_p', 'var_y_p', 'var_z_p',
    #                'mean_x_w', 'mean_y_w', 'mean_z_w', 'var_x_w', 'var_y_w', 'var_z_w']
    #df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    # Round numbers
    #df = df.round({'attr_x': 4, 'attr_y': 4, 'attr_z': 4})
    print('finished loading data...')
    return df_frames

def create_segments_and_labels(df, time_steps, step):
    print('=====starting to segment====')
    # x, y, z acceleration as features

    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    count_label = [0, 0]
    for i in range(0, len(df) - time_steps, step):
        #x = df['ECG'].values[i: i + time_steps]
        #eda = df['EDA'].values[i: i + time_steps]
        resp = df['RESP'].values[i: i + time_steps]
        #bvp = df['BVP'].values[i: i + time_steps]
        #z = df['EMG'].values[i: i + time_steps]
        #t = df['Temp'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        #print(df['label'][i: i + time_steps])[0][0]
        #print(df['label'][i: i + time_steps])[0][i + time_steps]
        label = stats.mode(df['label'][i: i + time_steps])[0]
        if label in LABELS:
            segments.append([#x,
                             #eda,
                             resp,
                             #bvp
                             #z,
                             #t
                            ])
            if (label == 1) or (label == 3):
                labels.append("non-stress")
            else:
                labels.append("stress")
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)
    print('=====end segment====')
    return reshaped_segments, labels

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


def show_basic_dataframe_info(dataframe):
    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))

def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix ACTIVITY')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def calculate_non_weighted_entropy(array):
    list_entropy = []
    total_rows = array.shape[1]
    for i in range(total_rows):
        entropy = 0
        total_columns = array.shape[0]
        for j in range(total_columns):
            if (array[j][i] != 0):
                entropy += (array[j][i] * math.log(array[j][i]))
        list_entropy.append(-entropy)

    return list_entropy


def concatenate_delete(train_xs, train_ys, validation_xs, validation_ys, delete_indices):
    train_xs = np.concatenate((train_xs, validation_xs[delete_indices]))
    train_ys = np.concatenate((train_ys, validation_ys[delete_indices]))
    #f_train.extend(np.array(f_validation)[delete_indices].tolist())

    validation_xs = np.delete(validation_xs, delete_indices, 0)
    validation_ys = np.delete(validation_ys, delete_indices, 0)
    #f_validation = np.delete(f_validation, delete_indices).tolist()

    return train_xs, train_ys, validation_xs, validation_ys

def concatenate(train_xs, train_ys, validation_xs, validation_ys, delete_indices):

    train_xs = np.concatenate((train_xs, validation_xs[delete_indices]))
    train_ys = np.concatenate((train_ys, validation_ys[delete_indices]))

    return train_xs, train_ys

def indicesNotMatches(listA, listB):
    # index variable
    idx = 0

    # Result list
    res = []

    # With iteration
    for i in listA:
        if i != listB[idx]:
            res.append(idx)
        idx = idx + 1

    # Result
    print("The index positions with mismatched values:\n", res)
    return res


def remove_common(a, b):
    a = a.tolist()
    #b = b.tolist()

    for i in a[:]:
        if i in b:
            a.remove(i)
            b.remove(i)

    print("list1 : ", a)
    print("list2 : ", b)
    return a

def count_classes_function(array_inputs):
    weight_classes = [0, 0]
    ratio_classes = [0, 0]
    count_classes_train_set = [0, 0]
    highest_class = 0
    count_classes_train_set_all = 0
    for index_, value_ in enumerate(array_inputs):
        if (np.argwhere(value_)) == 0:
            count_classes_train_set[0] = count_classes_train_set[0] + 1
        elif (np.argwhere(value_)) == 1:
            count_classes_train_set[1] = count_classes_train_set[1] + 1

    for ide, v in enumerate(count_classes_train_set):
        # output_file.write("class %s %s \n" % (ide, count_classes_train_set[ide]))
        print("class %s %s \n" % (ide, count_classes_train_set[ide]))
        count_classes_train_set_all = count_classes_train_set_all + count_classes_train_set[ide]
        if count_classes_train_set[ide] > highest_class:
            highest_class = count_classes_train_set[ide]
    print("highest class:", highest_class)
    # output_file.write("total classes: %s  \n" % (count_classes_train_set_all))

    for ide, v in enumerate(count_classes_train_set):
        weight_classes[ide] = (v/ count_classes_train_set_all * 1.0)
        ratio_classes[ide] = int(highest_class / (v))
    # (2) weight_classes[ide] = 1 - (v/(count_classes_train_set_all*1.0))
    # print "weight classes: ", weight_classes

    # ===find the minority class and return the weight of all classes==
    # ======stratgegy (1) weighted with formular true ratio = TP/all_classes ==========
    index_class = np.argsort(weight_classes)[:1]
    smallest_weight = min(weight_classes)
    # print "smallest weight: ", smallest_weight

    # ========stratgegy (2) weighted with formular 1 - true ratio==========
    # index_class = np.argsort(weight_classes)[-1:]
    # largest_weight = max(weight_classes)
    # print "largest weight", largest_weight

    lenght_act = len(weight_classes)
    index_class = lenght_act - index_class - 1
    onehot_minority_class = "%0*d" % (lenght_act, 10 ** index_class)
    onehot_minority_class = [int(l) for l in onehot_minority_class]
    onehot_minority_class = np.hstack([np.expand_dims(x, 0) for x in onehot_minority_class])
    onehot_minority_class = np.asarray(onehot_minority_class)
    print("one hot minority class: ", onehot_minority_class)

    return weight_classes, smallest_weight, onehot_minority_class, ratio_classes

def calculate_weighted_entropy(array, ratio_classes):
    list_entropy = []
    total_rows = array.shape[1]
    # print "total_rows", total_rows

    for i in range(total_rows):
        entropy = 0
        sum_ratio_classes = 0
        total_columns = array.shape[0]
        # print "total_columns",total_columns
        #for l in range(total_columns):
        #	sum_ratio_classes += list_ratio_classes[l] * array[l][i]
        #print('sum ratio:',sum_ratio_classes)
        for j in range(total_columns):
            if (array[j][i] != 0):
                #x = float(array[j][i] * list_ratio_classes[j]) / sum_ratio_classes
                #x = float(array[j][i] * list_ratio_classes[j])
                #entropy += x * math.log(x)
                print('ratio_classes[j]: ', ratio_classes[j])
                entropy += (array[j][i]* math.log(array[j][i]))* ratio_classes[j]
        list_entropy.append(-entropy)

    return list_entropy

class REINFORCE:
    def __init__(self, state_size, action):
        self.state_size = state_size
        self.action = action
        #self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # decay rate of past observations
        self.alpha = ALPHA  # learning rate in the policy gradient
        self.learning_rate = 0.01  # learning rate in deep learning
        self.hidden_units = 128
        self.dense_units = 2
        self.model = self.build_model()
        # record observations
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []
        self.total_rewards = []


    def build_model(self):

        model_m = keras.Sequential()

        w_c, s_w, min_c, r_c = count_classes_function(self.action)

        print('weighted classes before compute with reward: ', w_c)

        #print('self.reward: ', self.reward)

        input_shape = TIME_PERIODS * 1

        model_m.add(layers.Reshape((TIME_PERIODS, N_FEATURES),
                                   input_shape=(input_shape,))
                    )

        model_m.add(layers.LSTM(self.hidden_units,
                                activation='tanh',
                                input_shape=(TIME_PERIODS, N_FEATURES)
                                )
                    )

        model_m.add(layers.Dropout(0.5))

        model_m.add(layers.Dense(self.dense_units,
                                 use_bias=True,
                                 bias_initializer=keras.initializers.Constant(w_c),
                                 #kernel_initializer=keras.initializers.Constant(self.reward[:self.hidden_units]),
                                 activation='softmax')
                    )

        model_m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model_m.summary()

        return model_m

    def get_action(self, state):
        '''samples the next action based on the policy probability distribution
        of the actions'''
        # transform state
        #state = state.reshape([1, state.shape[0] * state.shape[1]])
        # get action probably
        action_probability_distribution = self.model.predict(state)

        print('action_probability_distribution: ', action_probability_distribution)
        # norm action probability distribution
        #action_probability_distribution /= np.sum(action_probability_distribution)

        #print('action_probability_distribution after using np.sum: ', action_probability_distribution)

        action = np.argmax(action_probability_distribution, axis=1)

        action = to_categorical(action, num_classes)

        print('action: ', action)

        # sample action
        #action = np.random.choice(action, state.shape[0], p=action_probability_distribution)

        #print('sample action: ', action)

        return action, action_probability_distribution

    def get_action_al(self, state):
        '''samples the next action based on the policy probability distribution
                of the actions'''
        # transform state
        # state = state.reshape([1, state.shape[0] * state.shape[1]])
        # get action probably
        action_probability_distribution = self.model.predict(state)

        print('action_probability_distribution before using np.sum: ', action_probability_distribution)
        # norm action probability distribution
        #action_probability_distribution /= np.sum(action_probability_distribution)

        #print('action_probability_distribution after using np.sum: ', action_probability_distribution)

        action = np.argmax(action_probability_distribution, axis=1)

        print('action after agrmax: ', action)

        action = to_categorical(action, num_classes)

        #print('action shape[0]: ', action.shape[0])

        #print('action shape[1]: ', action.shape[1])

        #action = action.tolist()

        #print('action to list: ', action)

        print('action proba: ', action_probability_distribution[0].tolist() )

        probs = action_probability_distribution[0].tolist()

        probs = [round(x, 2) for x in probs]

        print('probs ', probs)

        # sample action
        #for i in range(action.shape[0]):
        action = np.random.choice(action.shape[1], amount_sample, p=probs)

        action = to_categorical(action, num_classes)

        print('sample action: ', action)

        print('state', state)

        return state, action

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def get_discounted_rewards(self, rewards):

        '''Use gamma to calculate the total reward discounting for rewards
        Following - \gamma ^ t * Gt'''

        discounted_rewards = []
        cumulative_total_return = 0
        # iterate the rewards backwards and and calc the total return
        for reward in rewards[::-1]:
            cumulative_total_return = (cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards -
                                   mean_rewards) / (std_rewards + 1e-7)  # avoiding zero div

        return norm_discounted_rewards

    def remember(self, state, action, action_prob, reward):
        '''stores observations'''
        #encoded_action = self.hot_encode_action(action)
        self.gradients.append(action - action_prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)

    def update_policy(self):
        '''Updates the network.'''
        # get X
        current_state = len(self.states) - 1
        states = self.states[current_state]
        print('states: ', states)
        # get Y
        gradients = self.gradients[current_state]
        print('self.gradients before discount reward: ', gradients)
        rewards = self.rewards
        print('rewards: ', rewards)
        #print('rewards[0]: ', rewards[0])
        discounted_rewards = self.get_discounted_rewards(rewards)
        print('discounted rewards: ', discounted_rewards)

        gradients *= discounted_rewards[current_state]
        print('self.gradients after discount reward: ', gradients)

        #print('self.probs: ', self.probs)
        #gradients = gradients +  self.probs
        gradients = self.alpha * gradients + self.probs[current_state]

        #gradients = self.probs

        gradients = np.reshape(gradients, (-1, 2))

        print('self.gradients after multiple learning rate and plus the previous action proba: ', gradients)

        history = self.model.train_on_batch(states, gradients)

        #print('history: ', history)

        #self.states, self.probs, self.gradients, self.rewards = [], [], [], []

        return history

    def train(self, callbacks_list):

        history = self.model.fit(x_train, y_train,
                               batch_size=BATCH_SIZE,
                               epochs=EPOCHS,
                               #class_weight=class_weight_train,
                               #validation_split=0.1,
                               callbacks=callbacks_list,
                               verbose=1)

        acc = history.history['accuracy']

        print("!!!!!! training acc, previous acc: ", acc[EPOCHS-1], pre_acc)

        train_acc = float(acc[EPOCHS-1])

        #action = self.model.predict(x_train)

        return train_acc, train_acc
        #if train_acc - pre_acc <= 0.01:
        #    return action, train_acc, train_acc
        #else:
        #    return action, 0.01, train_acc



argument_parser = argparse.ArgumentParser(description="CLI for training and testing Sequential Neural Network Model")
argument_parser.add_argument("--train_file", type=str, help="Train file (CSV). Required for training.")
argument_parser.add_argument("--validation_file", type=str, help="Validation file (CSV). Required for validation.")
argument_parser.add_argument("--test_file", type=str, help="Test file (CSV). Required for testing.")
args = argument_parser.parse_args()

# Load data set containing all the data from csv
# Define column name of the label vector
#LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = pre.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
start_time = time.time()

train = read_data(cwd, args.train_file)

x_train, y_train = create_segments_and_labels(train, TIME_PERIODS, STEP_DISTANCE)

# Set input & output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
# Set input_shape / reshape for Keras
i_shape = (num_time_periods * num_sensors)

print('input_shape: ', i_shape)

x_train = x_train.reshape(x_train.shape[0], i_shape)

print('x_train shape:', x_train.shape)

x_train = x_train.astype('float64')

#x_train = x_train.reshape(x_train.shape + (1,))

x_train = np.asarray(x_train)

print('x_train: ', x_train)

print('x_train shape: ', x_train.shape)

y_train = le.fit_transform(y_train)

y_train = to_categorical(y_train, num_classes)

#y_train = np.argmax(y_train, axis =1)

#y_train = np.asarray(y_train)

#y_train = y_train.tolist()

#print('y_train: ', y_train)

print('y_train length: ', len(y_train))

#=================Read Validation Data===========================#
# val = read_data(cwd, args.validation_file)
#
# x_val, y_val = create_segments_and_labels(val, TIME_PERIODS, STEP_DISTANCE)
#
# x_val = x_val.reshape(x_val.shape[0], i_shape)
#
# x_val = x_val.astype('float32')
#
# x_val = np.asarray(x_val)
#
# print('x_test: ', x_val)
#
# y_val = le.fit_transform(y_val)
#
# y_val = to_categorical(y_val, num_classes)

#=================Read Test Data===========================#
test = read_data(cwd, args.test_file)

x_test, y_test = create_segments_and_labels(test, TIME_PERIODS, STEP_DISTANCE)

x_test = x_test.reshape(x_test.shape[0], i_shape)

x_test = x_test.astype('float64')

#x_test = x_test.reshape(x_test.shape + (1,))

x_test = np.asarray(x_test)

print('x_test: ', x_test)

y_test = le.fit_transform(y_test)

y_test = to_categorical(y_test, num_classes)

#============Training Phase=============#
state_size = x_train.shape[0]
action_size = y_train
agent = REINFORCE(state_size, action_size)
# Create the model with attention, train and evaluate
model_m = agent.build_model()

# Hyper-parameters
# serialize model to JSON
filepath="weights_al_rein_correct"+str(TIME_PERIODS)+str(STEP_DISTANCE)+str(EPOCHS)+str(BATCH_SIZE)+str(AL)+"_activity.best.keras"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

total_rewards=np.zeros(EPISODES)
step = 0
pre_acc = 0.0
#batch_size = 64

with open(report_dir, "a") as text_file:

#text_file = open(report_dir, "a")

    explored_indices = []

    for e in range(EPISODES):

        episode_reward = 0  # record episode reward

        total_sample_relabel = 0

        for l in range(iter):

            state = x_train

            print('x_train.shape: ', x_train.shape)

            action, prob = agent.get_action(state) #return predicted labels (action) and posterior probability (prob) on training data

            print("Starting to reinforcement learning iteration: ", l)

            reward, pre_acc = agent.train(callbacks_list) #return reward and previous accuracy on training data

            print("reward, pre_acc =====>", reward, pre_acc)

            episode_reward += reward

            agent.remember(state, action, prob, reward)

            print("================Test phase==================", l)

            model_m.load_weights(
                "weights_al_rein_correct" + str(TIME_PERIODS) + str(STEP_DISTANCE) + str(EPOCHS) + str(BATCH_SIZE)+str(AL)+"_activity.best.keras")

            print("Loaded model from disk")

            #model_m.compile(optimizer='adam',
            #                loss='categorical_crossentropy',
            #                metrics=['accuracy'])

            #score = model_m.evaluate(x_test, y_test, verbose=1)

            #print('\nAccuracy on test data: %0.2f' % score[1])

            #print('\nLoss on test data: %0.2f' % score[0])

            y_pred_test = model_m.predict(x_test)

            max_y_test = np.argmax(y_test, axis=1)

            max_y_pred_test = np.argmax(y_pred_test, axis=1)

            #show_confusion_matrix(max_y_test, max_y_pred_test)

            list_mismatch = indicesNotMatches(max_y_test,  max_y_pred_test)

            text_file.write('list mismatch: ' + str( list_mismatch ) + "\n")

            text_file.write("size of this list is: " + str(len(list_mismatch)) + "\n")

            print('size of this list is:', len(list_mismatch))

            print(classification_report(max_y_test, max_y_pred_test))

            report = classification_report(max_y_test, max_y_pred_test, digits=2, output_dict=False)

            text_file.write(report + "\n")

            y_pred_test = le.fit_transform(max_y_pred_test)

            one_hot_y_pred_test = to_categorical(y_pred_test, num_classes)

            print("===========Active Learning Phase==============:", l)

            #w_c, s_w, min_c, r_c = count_classes_function(y_train)

            #print('WEIGHT_CLASS: ', WEIGHT_CLASS)
            #
            y_prob_test = model_m.predict(x_test)
            #
            pred_entropies = calculate_non_weighted_entropy(y_prob_test.T)

            #pred_entropies = calculate_weighted_entropy(y_prob_test.T, w_c)
            #
            uncertainty_index_oversampling = np.argsort(pred_entropies)[::-1]

            #uncertainty_index_oversampling = np.argsort(pred_entropies)

            delete_indices = np.array([])

            delete_indices = uncertainty_index_oversampling[:AL]

            miss = list(set(list_mismatch).intersection(delete_indices))  # returns common elements in the two lists

            print('list of mismatch labels in query active learning: ', miss)

            text_file.write('list of mismatch labels in query active learning: ' + str(miss) + "\n")

            text_file.write("size of this list is: " + str(len(miss) ) + "\n")

            print('size of this list is:', len(miss))

            for i in delete_indices:
                #print(x_test[i])
                x_concat, pred_labels = agent.get_action_al(x_test[[i]])
                x_train = np.vstack((x_train, x_concat))
                y_train = np.vstack((y_train, pred_labels))

            #x_train, y_train = concatenate(x_train, y_train, x_test, one_hot_y_pred_test, delete_indices)

            text_file.write(" Samples of Training Data: " + str(len(x_train)) + "\n")

            #text_file.write(" ============================================= " + "\n")

            print("=======update the agent action and model===========")

            agent.build_model()

            history = agent.update_policy()

        total_rewards[e] = episode_reward

        #text_file.write("====END OF EPISODE " + str(e) + " has total sample relabels: " + str(total_sample_relabel) + "\n")

    agent.total_rewards = total_rewards


#numpy.set_printoptions(threshold=sys.maxsize)

#======Print confusion matrix for training data========
y_pred_train = model_m.predict(x_train)
#print('y_pred_train: ', y_pred_train)
#Take the class with the highest probability from the train_0_6 predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
max_train = np.argmax(y_train, axis=1)
print(classification_report(max_train, max_y_pred_train))



# later...
#=====load weights into new model====
model_m.load_weights("weights_al_rein_correct"+str(TIME_PERIODS)+str(STEP_DISTANCE)+str(EPOCHS)+str(BATCH_SIZE)+str(AL)+"_activity.best.keras")
print("Loaded model from disk")
model_m.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
score = model_m.evaluate(x_test, y_test, verbose=1)

print('\nAccuracy on test data: %0.2f' % score[1])
print('\nLoss on test data: %0.2f' % score[0])

y_pred_test = model_m.predict(x_test)
max_y_test = np.argmax(y_test, axis=1)

max_y_pred_test = np.argmax(y_pred_test, axis=1)
#print('F1 score:', classification.f1_score(max_y_test, max_y_pred_test, average='macro'))
#show_confusion_matrix(max_y_test, max_y_pred_test)
print(classification_report(max_y_test, max_y_pred_test))

#====write predicted activity list to file====
#act_list_predict_test = le.inverse_transform(max_y_pred_test)
#act_list_predict_test = le.inverse_transform(max_y_pred_calibration)
#write_list_to_file(act_list_predict_test, 'semantic_concept_generation/activity_prediction_test.csv')

#====write actual activity list to file======
#act_list_actual_test = le.inverse_transform(max_y_test)
#write_list_to_file(act_list_actual_test, 'semantic_concept_generation/activity_actual_test.csv')

#=======================================================================================================
end_time = time.time()
total_time_in_seconds = end_time - start_time
print("Completion time took %.2f seconds" % total_time_in_seconds)

#save model
os.chdir(cwd)
model_path = './model/model_full_save.keras'
directory = os.path.dirname(model_path)
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)
model_m.save(model_path)
