import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def validate(dist_array, top_k, input_data, index_offset=0):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[1]):
        gt_indexes = input_data.neighbor_gt_val[str(i+index_offset)]
        gt_dist = np.min(dist_array[gt_indexes, i])
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top_k:
            accuracy += 1.0
        data_amount += 1.0
    
    accuracy /= data_amount
    
    return accuracy


def validate_local(dist_array, top_k, input_data, index_offset=0):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[1]):
        nearby_indexes = input_data.neighbor_negative_samples_val[str(i+index_offset)]
        gt_indexes = input_data.neighbor_gt_val[str(i+index_offset)]
        gt_dist = np.min(dist_array[gt_indexes, i])
        list_withincircle = [dist_array[j, i] for j in nearby_indexes]
        prediction = np.sum(list_withincircle<gt_dist)
        if prediction < top_k:
            accuracy += 1.0
        data_amount += 1.0
    
    accuracy /= data_amount
    
    return accuracy

def compute_loss(sat_global, grd_global, utms_x, UTMthres, useful_pairs_s2g, useful_pairs_g2s):
    
    with tf.name_scope('weighted_soft_margin_triplet_loss'):
        loss_weight = 10.0
        tfd = tfp.distributions
        zeros = tf.fill(tf.shape(useful_pairs_s2g), 0.0)
        sig = tf.fill(tf.shape(useful_pairs_s2g), tf.constant(UTMthres,dtype=tf.float32)) 
        dist = tfd.Normal(loc=zeros, scale=sig)
        Distance_weights = (-dist.prob(utms_x)+dist.prob(zeros))/dist.prob(zeros)

       
        batch_size, channels = sat_global.get_shape().as_list()
        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True) #[S1G1, S1G2; S2G1, S2G2]
        pos_dist = tf.diag_part(dist_array)
        # ground to satellite
        pair_n_g2s = tf.reduce_sum(useful_pairs_g2s) + 0.001 # avoid zero division
        triplet_dist_g2s = (pos_dist - dist_array)
        loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))*\
                                 tf.multiply(Distance_weights, useful_pairs_g2s)) / pair_n_g2s

        # satellite to ground
        pair_n_s2g = tf.reduce_sum(useful_pairs_s2g) + 0.001
        triplet_dist_s2g = (tf.expand_dims(pos_dist, 1) - dist_array)
        loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))*\
                                 tf.multiply(Distance_weights, useful_pairs_s2g)) / pair_n_s2g

        loss = (loss_g2s + loss_s2g) / 2.0
        
    return loss