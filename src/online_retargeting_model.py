import os

import numpy as np
import tensorflow as tf

from forward_kinematics import FK
from ops import gaussian_noise as gnoise
from ops import batch_norm
from ops import instance_norm
from ops import conv1d
from ops import lrelu
from ops import linear
from ops import qlinear
from tensorflow import atan2
from tensorflow import asin


class EncoderDecoderGRU(object):
  def __init__(self,
               batch_size,
               alpha,
               gamma,
               omega,
               euler_ord,
               n_joints,
               layers_units,
               max_len,
               dmean,
               dstd,
               omean,
               ostd,
               parents,
               keep_prob,
               logs_dir,
               learning_rate,
               optim_name,
               is_train=True):

    self.n_joints = n_joints
    self.batch_size = batch_size
    self.alpha = alpha
    self.gamma = gamma
    self.omega = omega
    self.euler_ord = euler_ord
    self.kp = keep_prob
    self.max_len = max_len
    self.learning_rate = learning_rate
    self.fk = FK()

    self.seqA_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len, 3 * n_joints + 4], name="seqA")
    self.seqB_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len, 3 * n_joints + 4], name="seqB")
    self.skelA_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len, 3 * n_joints], name="skelA")
    self.skelB_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len, 3 * n_joints], name="skelB")
    self.mask_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len], name="mask")

    enc_gru = self.gru_model(layers_units)
    dec_gru = self.gru_model(layers_units)

    b_local = []
    b_global = []
    b_quats = []
    a_local = []
    a_global = []
    a_quats = []
    reuse = False

    statesA_AB = ()
    statesB_AB = ()
    statesA_BA = ()
    statesB_BA = ()

    for units in layers_units:
      statesA_AB += (tf.zeros([batch_size, units]), )
      statesB_AB += (tf.zeros([batch_size, units]), )
      statesA_BA += (tf.zeros([batch_size, units]), )
      statesB_BA += (tf.zeros([batch_size, units]), )

    for t in range(max_len):
      """ Retarget A to B """
      with tf.variable_scope("Encoder", reuse=reuse):
        ptA_in = self.seqA_[:, t, :]

        _, statesA_AB = tf.contrib.rnn.static_rnn(
            enc_gru, [ptA_in], initial_state=statesA_AB, dtype=tf.float32)

      with tf.variable_scope("Decoder", reuse=reuse):
        if t == 0:
          ptB_in = tf.zeros([batch_size, 3 * n_joints + 4])
        else:
          ptB_in = tf.concat([b_local[-1], b_global[-1]], axis=-1)

        ptcombined = tf.concat(
            values=[self.skelB_[:, 0, 3:], ptB_in, statesA_AB[-1]], axis=1)
        _, statesB_AB = tf.contrib.rnn.static_rnn(
            dec_gru, [ptcombined], initial_state=statesB_AB, dtype=tf.float32)
        angles_n_offset = self.mlp_out(statesB_AB[-1])
        output_angles = tf.reshape(angles_n_offset[:, :-4],
                                   [batch_size, n_joints, 4])
        b_global.append(angles_n_offset[:, -4:])
        b_quats.append(self.normalized(output_angles))

        skel_in = tf.reshape(self.skelB_[:, 0, :], [batch_size, n_joints, 3])
        skel_in = skel_in * dstd + dmean

        output = (self.fk.run(parents, skel_in, output_angles) - dmean) / dstd
        output = tf.reshape(output, [batch_size, -1])
        b_local.append(output)

        reuse = True

    self.localB = tf.stack(b_local, axis=1)
    self.globalB = tf.stack(b_global, axis=1)
    self.quatB = tf.stack(b_quats, axis=1)

    if is_train:
      dmean = dmean.reshape([1, 1, -1])
      dstd = dstd.reshape([1, 1, -1])

      output_localB = self.localB
      target_seqB = self.seqB_[:, :, :-4]
      self.local_loss = tf.reduce_sum(
          tf.square(
              tf.multiply(self.mask_[:, :, None],
                          tf.subtract(output_localB, target_seqB))))
      self.local_loss = tf.divide(self.local_loss,
                                  tf.maximum(tf.reduce_sum(self.mask_), 1))

      self.global_loss = tf.reduce_sum(
          tf.square(
              tf.multiply(self.mask_[:, :, None],
                          tf.subtract(self.seqB_[:, :, -4:], self.globalB))))
      self.global_loss = tf.divide(self.global_loss,
                                   tf.maximum(tf.reduce_sum(self.mask_), 1))

      dnorm_offB_ = self.globalB * ostd + omean
      self.smoothness = tf.reduce_sum(
          tf.square(
              tf.multiply(self.mask_[:, 1:, None],
                          dnorm_offB_[:, 1:] - dnorm_offB_[:, :-1])))
      self.smoothness = tf.divide(self.smoothness,
                                  tf.maximum(tf.reduce_sum(self.mask_), 1))

      rads = self.alpha / 180.0
      self.twist_loss = tf.reduce_mean(
          tf.square(
              tf.maximum(
                  0.0,
                  tf.abs(self.euler(self.quatB, euler_ord)) - rads * np.pi)))

      self.L = (self.local_loss + self.global_loss +
                self.gamma * self.twist_loss + self.omega * self.smoothness)

      local_sum = tf.summary.scalar("losses/local_loss", self.local_loss)
      global_sum = tf.summary.scalar("losses/global_loss", self.global_loss)
      twist_sum = tf.summary.scalar("losses/twist_loss", self.twist_loss)
      smooth_sum = tf.summary.scalar("losses/smoothness", self.smoothness)

      self.sum = tf.summary.merge(
          [local_sum, global_sum, twist_sum, smooth_sum])
      self.writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

      self.allvars = tf.trainable_variables()
      self.gvars = [v for v in self.allvars if "DIS" not in v.name]

      if optim_name == "rmsprop":
        goptimizer = tf.train.RMSPropOptimizer(
            self.learning_rate, name="goptimizer")
      elif optim_name == "adam":
        goptimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1=0.5, name="goptimizer")
      else:
        raise Exception("Unknown optimizer")

      ggradients, gg = zip(
          *goptimizer.compute_gradients(self.L, var_list=self.gvars))

      ggradients, _ = tf.clip_by_global_norm(ggradients, 25)

      self.goptim = goptimizer.apply_gradients(zip(ggradients, gg))

      num_param = 0
      for var in self.gvars:
        num_param += int(np.prod(var.get_shape()))
      print "NUMBER OF G PARAMETERS: " + str(num_param)

    self.saver = tf.train.Saver()

  def mlp_out(self, input_, reuse=False, name="mlp_out"):
    out = qlinear(input_, 4 * (self.n_joints + 1), name="dec_fc")
    return out

  def gru_model(self, layers_units, rnn_type="GRU"):
    gru_cells = [tf.contrib.rnn.GRUCell(units) for units in layers_units]
    gru_cells = [
        tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.kp)
        for cell in gru_cells
    ]
    stacked_gru = tf.contrib.rnn.MultiRNNCell(gru_cells)
    return stacked_gru

  def train(self, sess, seqA_, skelA_, seqB_, skelB_, mask_, step):
    feed_dict = dict()

    feed_dict[self.seqA_] = seqA_
    feed_dict[self.skelA_] = skelA_
    feed_dict[self.seqB_] = seqB_
    feed_dict[self.skelB_] = skelB_
    feed_dict[self.mask_] = mask_

    _, summary_str = sess.run([self.goptim, self.sum], feed_dict=feed_dict)

    self.writer.add_summary(summary_str, step)
    l = self.L.eval(feed_dict=feed_dict)

    return l

  def predict(self, sess, seqA_, skelB_, mask_):
    feed_dict = dict()
    feed_dict[self.seqA_] = seqA_
    feed_dict[self.skelB_] = skelB_
    feed_dict[self.mask_] = mask_
    SL = self.localB.eval(feed_dict=feed_dict)
    SG = self.globalB.eval(feed_dict=feed_dict)
    output = np.concatenate((SL, SG), axis=-1)
    quats = self.quatB.eval(feed_dict=feed_dict)
    return output, quats

  def normalized(self, angles):
    lengths = tf.sqrt(tf.reduce_sum(tf.square(angles), axis=-1))
    return angles / lengths[..., None]

  def euler(self, angles, order="yzx"):
    q = self.normalized(angles)
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    if order == "xyz":
      ex = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
      ey = asin(tf.clip_by_value(2 * (q0 * q2 - q3 * q1), -1, 1))
      ez = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
      return tf.stack(values=[ex, ez], axis=-1)[:, :, 1:]
    elif order == "yzx":
      ex = atan2(2 * (q1 * q0 - q2 * q3),
                 -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
      ey = atan2(2 * (q2 * q0 - q1 * q3),
                 q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
      ez = asin(tf.clip_by_value(2 * (q1 * q2 + q3 * q0), -1, 1))
      return ey[:, :, 1:]
    else:
      raise Exception("Unknown Euler order!")

  def save(self, sess, checkpoint_dir, step):
    model_name = "EncoderDecoderGRU.model"

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(
        sess, os.path.join(checkpoint_dir, model_name), global_step=step)

  def load(self, sess, checkpoint_dir, model_name=None):
    print("[*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if "paper_models" in checkpoint_dir:
      self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
      print("     Loaded model: " + str(model_name))
      return True, model_name
    elif ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      if model_name is None: model_name = ckpt_name
      self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
      print("     Loaded model: " + str(model_name))
      return True, model_name
    else:
      return False, None
