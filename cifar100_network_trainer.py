import argparse
import numpy as np
import tensorflow as tf
import tqdm
from data_providers import CIFAR100DataProvider
from network_builder import ClassifierNetworkGraph
from utils.parser_utils import ParserClass
from utils.storage import build_experiment_folder, save_statistics

# Resets any previous graphs to clear memory =====================================================================================
tf.reset_default_graph()  

# Parser =========================================================================================================================
parser = argparse.ArgumentParser(description='Welcome to CNN experiments script')  # generates an argument parser
parser_extractor = ParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, tensorboard_enable, batch_norm, \
strided_dim_reduction, experiment_prefix, dropout_rate_value, classifier_type = parser_extractor.get_argument_variables()



# Setup exp directories ============================================================================================================
experiment_name = "Experiment_{}_batch_size_{}_bn_{}_mp_{}".format(experiment_prefix,
                                                                   batch_size, batch_norm,
                                                                   strided_dim_reduction) # generate exp name

saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)  # generate experiment dir


# Data inputs ======================================================================================================================
rng = np.random.RandomState(seed=seed)  # set seed

train_data = CIFAR100DataProvider(which_set="train", batch_size=batch_size, rng=rng, random_sampling=True)
val_data = CIFAR100DataProvider(which_set="valid", batch_size=batch_size, rng=rng)
test_data = CIFAR100DataProvider(which_set="test", batch_size=batch_size, rng=rng)
#  setup our data providers

data_inputs = tf.placeholder(tf.float32, [batch_size, train_data.inputs.shape[1], train_data.inputs.shape[2],
                                          train_data.inputs.shape[3]], 'data-inputs')

# Define classification task placeholders ============================================================================
#Multi task--------------------------------------------------------------------------

#Main Task
data_targets = tf.placeholder(tf.int32, [batch_size], 'data-targets')

#Aux Task1
data_targets1 = tf.placeholder(tf.int32, [batch_size], 'data-targets1')
#------------------------------------------------------------------------------------

training_phase = tf.placeholder(tf.bool, name='training-flag')
rotate_data = tf.placeholder(tf.bool, name='rotate-flag')
dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')

classifier_network = ClassifierNetworkGraph(input_x=data_inputs, target_placeholder=data_targets, target_placeholder1=data_targets1,
                                            dropout_rate=dropout_rate, batch_size=batch_size,
                                            num_channels=train_data.inputs.shape[3], n_classes=train_data.num_classes,
                                            is_training=training_phase, augment_rotate_flag=rotate_data,
                                            strided_dim_reduction=strided_dim_reduction,
                                            use_batch_normalization=batch_norm, network_name=classifier_type)  # initialize network. 


# Setup Computational Graph =========================================================================================================

if continue_from_epoch == -1:  # if this is a new experiment and not continuation of a previous one then generate a new
    # statistics file
    save_statistics(logs_filepath, "result_summary_statistics", ["epoch", "train_c_loss", "train_c_accuracy",
                                                                 "val_c_loss", "val_c_accuracy",
                                                                 "test_c_loss", "test_c_accuracy"], create=True)

start_epoch = continue_from_epoch if continue_from_epoch != -1 else 0  # if new experiment start from 0 otherwise
# continue where left off

#Multi task-----------------------------------------------------------------------------------------------------
losses_ops, (c_error_opt_op,c_error_opt_op1) = classifier_network.init_train()  # get graph operations (ops)
#---------------------------------------------------------------------------------------------------------------

total_train_batches = train_data.num_batches
total_val_batches = val_data.num_batches
total_test_batches = test_data.num_batches

best_epoch = 0

init = tf.global_variables_initializer()  # initialization op for the graph


# Run Graph ========================================================================================================================
with tf.Session() as sess:
    sess.run(init)  # actually running the initialization op
    train_saver = tf.train.Saver()  # saver object that will save our graph so we can reload it later for continuation of
    val_saver = tf.train.Saver()
    #  training or inference

    if continue_from_epoch != -1:
        train_saver.restore(sess, "{}/{}_{}.ckpt".format(saved_models_filepath, experiment_name,
                                                   continue_from_epoch))  # restore previous graph to continue operations

    best_val_accuracy = 0.
    for e in range(start_epoch, epochs):
        
        # TRAINING
        #####################################################################################################################
        #Multi task-----------------------------------------------------------------
        
        #Main Task
        total_c_loss = 0.
        total_accuracy = 0.
        
        #Aux Task1
        total_c_loss1 = 0.
        total_accuracy1 = 0.
        #--------------------------------------------------------------------------
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_data):
            iter_id = e * total_train_batches + batch_idx
            #Multi task---------------------------------------------------------------------------------
            
            #Main Task
            _, c_loss_value, acc = sess.run(
            [c_error_opt_op, losses_ops["crossentropy_losses"], losses_ops["accuracy"]],
            feed_dict={dropout_rate: dropout_rate_value, data_inputs: x_batch,
            data_targets: y_batch, training_phase: True, rotate_data: False})
                        
            total_c_loss += c_loss_value  # add loss of current iter to sum
            total_accuracy += acc # add acc of current iter to sum
                    
            #Aux Task1
            _, c_loss_value1, acc1 = sess.run(
            [c_error_opt_op1, losses_ops["crossentropy_losses1"], losses_ops["accuracy1"]],
            feed_dict={dropout_rate: dropout_rate_value, data_inputs: x_batch,
            data_targets1: y_batch, training_phase: True, rotate_data: False})
                    
            total_c_loss1 += c_loss_value1  # add loss of current iter to sum
            total_accuracy1 += acc1 # add acc of current iter to sum
            #---------------------------------------------------------------------------------------------        
                            
        #Multi task-------------------------------------------------------------
        #Main Task
        total_c_loss /= total_train_batches  # compute mean of loss
        total_accuracy /= total_train_batches # compute mean of accuracy

        #Aux Task1
        total_c_loss1 /= total_train_batches  # compute mean of loss
        total_accuracy1 /= total_train_batches # compute mean of accuracy
        #-----------------------------------------------------------------------
        
        # save graph and weights
        save_path = train_saver.save(sess, "{}/{}_{}.ckpt".format(saved_models_filepath, experiment_name, e))
        
        # VALIDATION 
        ############################################################################################################################
        
        #Multi task-----------------------------------------------------
        
        #Main Task
        total_val_c_loss = 0.
        total_val_accuracy = 0.
        
        #Aux Task1
        total_val_c_loss1 = 0.
        total_val_accuracy1 = 0.
        #--------------------------------------------------------------
        
        for batch_idx, (x_batch, y_batch) in enumerate(val_data):
        
            #Multi task--------------------------------------------------------------------
            
            #Main Task
            c_loss_value, acc = sess.run(
            [losses_ops["crossentropy_losses"], losses_ops["accuracy"]],
            feed_dict={dropout_rate: dropout_rate_value, data_inputs: x_batch,
            data_targets: y_batch, training_phase: False, rotate_data: False})
        
            total_val_c_loss += c_loss_value
            total_val_accuracy += acc
            
            #Aux Task1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! replace y batch with target values.
            c_loss_value1, acc1 = sess.run(
            [losses_ops["crossentropy_losses1"], losses_ops["accuracy1"]],
            feed_dict={dropout_rate: dropout_rate_value, data_inputs: x_batch,
            data_targets1: y_batch, training_phase: False, rotate_data: False})
        
            total_val_c_loss1 += c_loss_value1
            total_val_accuracy1 += acc1
            #-----------------------------------------------------------------------------
            
        #Multi task------------------------------------
        
        #Main Task
        total_val_c_loss /= total_val_batches
        total_val_accuracy /= total_val_batches
        
        #Task2
        total_val_c_loss1 /= total_val_batches
        total_val_accuracy1 /= total_val_batches
        #----------------------------------------------
        
        #Only concerns Main Task
        if best_val_accuracy < total_val_accuracy:  # check if val acc better than the previous best and if
            # so save current as best and save the model as the best validation model to be used on the test set
            #  after the final epoch
            best_val_accuracy = total_val_accuracy
            best_epoch = e
            save_path = val_saver.save(sess, "{}/best_validation_{}_{}.ckpt".format(saved_models_filepath, experiment_name, e))
            print("Saved best validation score model at", save_path)

        # save statistics of this epoch, train and val without test set performance
        save_statistics(logs_filepath, "result_summary_statistics",
                       [e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy,
                        -1, -1])

        # TESTING - Only concerns Main Task
        ##########################################################################################################################
        val_saver.restore(sess, "{}/best_validation_{}_{}.ckpt".format(saved_models_filepath, experiment_name, best_epoch))
        # restore model with best performance on validation set
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        # compute test loss and accuracy and save
    for batch_id, (x_batch, y_batch) in enumerate(test_data):
        c_loss_value, acc = sess.run(
        [losses_ops["crossentropy_losses"], losses_ops["accuracy"]],
        feed_dict={dropout_rate: dropout_rate_value, data_inputs: x_batch,
        data_targets: y_batch, training_phase: False, rotate_data: False})
            
        total_test_c_loss += c_loss_value
        total_test_accuracy += acc
        iter_out = "test_loss: {}, test_accuracy: {}".format(total_test_c_loss / (batch_idx + 1),
                                                                     acc / (batch_idx + 1))

    total_test_c_loss /= total_test_batches
    total_test_accuracy /= total_test_batches

    save_statistics(logs_filepath, "result_summary_statistics",
                     ["test set performance", -1, -1, -1, -1,
                     total_test_c_loss, total_test_accuracy])
