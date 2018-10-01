import os

import numpy as np
import tensorflow as tf

from forward_kinematics import FK
from ops import gaussian_noise as gnoise
from ops import conv1d
from ops import lrelu
from ops import linear
from ops import qlinear
from ops import relu
from tensorflow import atan2
from tensorflow import asin

layer_norm = tf.contrib.layers.layer_norm

#python2 だから，親から継承してないクラスの引数に　object　が必要
#python3 では，object　()　は必要ない
class EncoderDecoderGRU(object):
  #このクラスのインスタンスが使われる時はいつでも
  #先に、ネットワークの構造の設計等を決める
  #__init__　の中身を「コンストラクタ」という
  #__del__ としたら「デストラクタ」という
  def __init__(self,
               batch_size,
               alpha,
               beta,
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
               margin,
               d_arch,
               d_rand,
               norm_type,
               is_train=True):

    self.n_joints = n_joints
    self.batch_size = batch_size
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.omega = omega
    self.euler_ord = euler_ord
    #これだけ名前変更
    self.kp = keep_prob
    self.max_len = max_len
    self.learning_rate = learning_rate
    self.fk = FK()
    self.d_arch = d_arch
    self.d_rand = d_rand
    self.margin = margin
    #クラスの引数ではない
    self.fake_score = 0.5
    self.norm_type = norm_type

    #____________________RNNに渡す各入力の型を作成_________________________________
    self.realSeq_ = tf.placeholder(
        tf.float32,
        shape=[batch_size, max_len, 3 * n_joints + 4],
        name="realSeq")
        #batch_size　1学習につかうアニメーションセットの数
        #max_len あるアニメーションのフレーム数
        #3N →pの成分　4→vの成分
    self.realSkel_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len, 3 * n_joints], name="realSkel")

    #アニメーションが入る?
    self.seqA_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len, 3 * n_joints + 4], name="seqA")
    self.seqB_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len, 3 * n_joints + 4], name="seqB")

    #T-poseが入る?
    self.skelA_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len, 3 * n_joints], name="skelA")
    self.skelB_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len, 3 * n_joints], name="skelB")

    self.aeReg_ = tf.placeholder(
        tf.float32, shape=[batch_size, 1], name="aeReg")
    self.mask_ = tf.placeholder(
        tf.float32, shape=[batch_size, max_len], name="mask")
    self.kp_ = tf.placeholder(tf.float32, shape=[], name="kp")
    #____________________RNNに渡す各入力の型を作成_________________________________

    #モデルの呼び出し
    #RNNのセル配置を作成
    enc_gru = self.gru_model(layers_units)
    dec_gru = self.gru_model(layers_units)

    #出力結果を入れる箱?
    #ネットのA→B部分
    b_local = []   #p^B　が入る
    b_global = []  #v^B　が入る
    b_quats = []   #q^B　が入る
    #ネットのB→A部分
    a_local = []
    a_global = []
    a_quats = []
    reuse = False

    #RNNのグラフを入れる箱　　()でタプル(集合)を定義
    statesA_AB = ()
    statesB_AB = ()
    statesA_BA = ()
    statesB_BA = ()

    for units in layers_units:
      #タプルに対する+=は集合的な和算を意味する
      #A→BへのエンコーダRNNに
      statesA_AB += (tf.zeros([batch_size, units]), )
      #A→BへのデコーダRNNに
      statesB_AB += (tf.zeros([batch_size, units]), )
      #B→AへのエンコーダRNNに
      statesA_BA += (tf.zeros([batch_size, units]), )
      #B→AへのデコーダRNNに
      statesB_BA += (tf.zeros([batch_size, units]), )

    for t in range(max_len):

      """Retarget A to B"""
      #名前空間(変数たちをまとめる袋)を作る
      #Encoderは袋のタグ　reuse=Falseなら，袋の中の各変数は異なるキーワードを持つ
      with tf.variable_scope("Encoder", reuse=reuse):

        #すべてのアニメーションセットの中で時刻tのフレームだけ取り出す
        #どのtでもEncoderと識別子がついてる
        ptA_in = self.seqA_[:, t,:]

        #RNNの構造が決まる　static_rnn(cell,inputs,)
        _, statesA_AB = tf.contrib.rnn.static_rnn(
            enc_gru, [ptA_in], initial_state=statesA_AB, dtype=tf.float32)

      with tf.variable_scope("Decoder", reuse=reuse):
        if t == 0:
          #エンコーダを通った結果が入り，これがデコーダに渡される
          #h_enc+Sバー　
          ptB_in = tf.zeros([batch_size, 3 * n_joints + 4])
        else:
          #t=>1なら，デコーダの入力にp^,v^が入るはず
          #tf.concat 配列を軸を指定して結合させる
          ptB_in = tf.concat([b_local[-1], b_global[-1]], axis=-1)

        #デコーダのRNNのグラフの入力の型を決定
        ptcombined = tf.concat(
            values=[self.skelB_[:, 0, 3:], ptB_in, statesA_AB[-1]], axis=1)
        #デコーダのRNNのグラフを定義してstatesB_ABに入れる
        _, statesB_AB = tf.contrib.rnn.static_rnn(
            dec_gru, [ptcombined], initial_state=statesB_AB, dtype=tf.float32)

        angles_n_offset = self.mlp_out(statesB_AB[-1])
        output_angles = tf.reshape(angles_n_offset[:, :-4],
                                   [batch_size, n_joints, 4])
        #v^Bが入る
        b_global.append(angles_n_offset[:, -4:])
        #q^Bが入る
        b_quats.append(self.normalized(output_angles))

        skel_in = tf.reshape(self.skelB_[:, 0, :], [batch_size, n_joints, 3])
        skel_in = skel_in * dstd + dmean

        #A→Bデコーダからの出力をFKに入力するグラフの連結
        output = (self.fk.run(parents, skel_in, output_angles) - dmean) / dstd
        output = tf.reshape(output, [batch_size, -1])
        b_local.append(output)

      """Retarget B back to A"""
      with tf.variable_scope("Encoder", reuse=True):
        ptB_in = tf.concat([b_local[-1], b_global[-1]], axis=-1)

        _, statesB_BA = tf.contrib.rnn.static_rnn(
            enc_gru, [ptB_in], initial_state=statesB_BA, dtype=tf.float32)

      with tf.variable_scope("Decoder", reuse=True):
        if t == 0:
          ptA_in = tf.zeros([batch_size, 3 * n_joints + 4])
        else:
          ptA_in = tf.concat([a_local[-1], a_global[-1]], axis=-1)

        ptcombined = tf.concat(
            values=[self.skelA_[:, 0, 3:], ptA_in, statesB_BA[-1]], axis=1)
        _, statesA_BA = tf.contrib.rnn.static_rnn(
            dec_gru, [ptcombined], initial_state=statesA_BA, dtype=tf.float32)
        angles_n_offset = self.mlp_out(statesA_BA[-1])
        output_angles = tf.reshape(angles_n_offset[:, :-4],
                                   [batch_size, n_joints, 4])
        a_global.append(angles_n_offset[:, -4:])
        a_quats.append(self.normalized(output_angles))

        skel_in = tf.reshape(self.skelA_[:, 0, :], [batch_size, n_joints, 3])
        skel_in = skel_in * dstd + dmean

        #B→Aデコーダからの出力をFKに入力するグラフの連結
        output = (self.fk.run(parents, skel_in, output_angles) - dmean) / dstd
        output = tf.reshape(output, [batch_size, -1])
        a_local.append(output)

        reuse = True

    #tf.stack(T= 3階のテンソル, axis=1)で，テンソルの各成分(行列)のある行(axis=1)だけを
    #取り出して1つの行列にまとめている
    self.localB = tf.stack(b_local, axis=1)
    self.globalB = tf.stack(b_global, axis=1)
    #q^Bを入れる
    self.quatB = tf.stack(b_quats, axis=1)
    self.localA = tf.stack(a_local, axis=1)
    self.globalA = tf.stack(a_global, axis=1)
    #q^Aを入れる
    self.quatA = tf.stack(a_quats, axis=1)
    #3階のテンソル axis=0,1,2 　T[a,b,c] a番目の行列(axis=0方向)のb番目の行(axis=1方向)
    #のc番目の列(axis=2方向)にある要素を取り出せる

    #_________________________ネットワークのグラフ終了_____________________________

    if is_train:
      dmean = dmean.reshape([1, 1, -1])
      dstd = dstd.reshape([1, 1, -1])

      with tf.variable_scope("DIS", reuse=False):
        if self.d_rand:
          dnorm_seq = self.realSeq_[:, :, :-4] * dstd + dmean
          dnorm_off = self.realSeq_[:, :, -4:] * ostd + omean
          #t=0(各行列の1行目)の3列目〜(root以外の関節すべて)取り出す
          skel_ = self.realSkel_[:, 0:1, 3:]
        else:
          dnorm_seq = self.seqA_[:, :, :-4] * dstd + dmean
          dnorm_off = self.seqA_[:, :, -4:] * ostd + omean
          skel_ = self.skelA_[:, 0:1, 3:]

        diff_seq = dnorm_seq[:, 1:, :] - dnorm_seq[:, :-1, :]
        real_data = tf.concat([diff_seq, dnorm_off[:, :-1, :]], axis=-1)

        self.D_logits = self.discriminator(real_data, skel_)
        #r^Aを計算
        self.D = tf.nn.sigmoid(self.D_logits)

      self.seqB = tf.concat([self.localB, self.globalB], axis=-1)
      with tf.variable_scope("DIS", reuse=True):
        dnorm_seqB = self.seqB[:, :, :-4] * dstd + dmean
        dnorm_offB = self.seqB[:, :, -4:] * ostd + omean
        diff_seqB = dnorm_seqB[:, 1:, :] - dnorm_seqB[:, :-1, :]
        fake_data = tf.concat([diff_seqB, dnorm_offB[:, :-1, :]], axis=-1)

        self.D_logits_ = self.discriminator(fake_data, self.skelB_[:, 0:1, 3:])
        #r^Bを計算
        self.D_ = tf.nn.sigmoid(self.D_logits_)

      """ CYCLE OBJECTIVE """
      output_localA = self.localA
      target_seqA = self.seqA_[:, :, :-4]
      self.cycle_local_loss = tf.reduce_sum(
          tf.square(
              tf.multiply(self.mask_[:, :, None],
                          tf.subtract(output_localA, target_seqA))))
      self.cycle_local_loss = tf.divide(self.cycle_local_loss,
                                        tf.reduce_sum(self.mask_))

      self.cycle_global_loss = tf.reduce_sum(
          tf.square(
              tf.multiply(self.mask_[:, :, None],
                          tf.subtract(self.seqA_[:, :, -4:], self.globalA))))
      self.cycle_global_loss = tf.divide(self.cycle_global_loss,
                                         tf.reduce_sum(self.mask_))

      dnorm_offA_ = self.globalA * ostd + omean
      self.cycle_smooth = tf.reduce_sum(
          tf.square(
              tf.multiply(self.mask_[:, 1:, None],
                          dnorm_offA_[:, 1:] - dnorm_offA_[:, :-1])))
      self.cycle_smooth = tf.divide(self.cycle_smooth,
                                    tf.reduce_sum(self.mask_))
      """ INTERMEDIATE OBJECTIVE FOR REGULARIZATION """
      output_localB = self.localB
      target_seqB = self.seqB_[:, :, :-4]
      self.interm_local_loss = tf.reduce_sum(
          tf.square(
              tf.multiply(self.aeReg_[:, :, None] * self.mask_[:, :, None],
                          tf.subtract(output_localB, target_seqB))))
      self.interm_local_loss = tf.divide(
          self.interm_local_loss,
          tf.maximum(tf.reduce_sum(self.aeReg_ * self.mask_), 1))

      self.interm_global_loss = tf.reduce_sum(
          tf.square(
              tf.multiply(self.aeReg_[:, :, None] * self.mask_[:, :, None],
                          tf.subtract(self.seqB_[:, :, -4:], self.globalB))))
      self.interm_global_loss = tf.divide(
          self.interm_global_loss,
          tf.maximum(tf.reduce_sum(self.aeReg_ * self.mask_), 1))

      dnorm_offB_ = self.globalB * ostd + omean
      self.interm_smooth = tf.reduce_sum(
          tf.square(
              tf.multiply(self.mask_[:, 1:, None],
                          dnorm_offB_[:, 1:] - dnorm_offB_[:, :-1])))
      self.interm_smooth = tf.divide(self.interm_smooth,
                                     tf.maximum(tf.reduce_sum(self.mask_), 1))
      #°をラジアンに
      rads = self.alpha / 180.0
      self.twist_loss1 = tf.reduce_mean(
          tf.square(
              tf.maximum(
                  0.0,
                  tf.abs(self.euler(self.quatB, euler_ord)) - rads * np.pi)))
      self.twist_loss2 = tf.reduce_mean(
          tf.square(
              tf.maximum(
                  0.0,
                  tf.abs(self.euler(self.quatA, euler_ord)) - rads * np.pi)))

      """Twist loss"""
      self.twist_loss = 0.5 * (self.twist_loss1 + self.twist_loss2)

      """Acceleration smoothness loss"""
      self.smoothness = 0.5 * (self.interm_smooth + self.cycle_smooth)

      self.overall_loss = (
          self.cycle_local_loss + self.cycle_global_loss +
          self.interm_local_loss + self.interm_global_loss +
          self.gamma * self.twist_loss + self.omega * self.smoothness)

      self.L_disc_real = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.D_logits, labels=tf.ones_like(self.D_logits)))

      self.L_disc_fake = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.D_logits_, labels=tf.zeros_like(self.D_logits_)))

      self.L_disc = self.L_disc_real + self.L_disc_fake

      self.L_gen = tf.reduce_sum(
          tf.multiply((1 - self.aeReg_),
                      tf.nn.sigmoid_cross_entropy_with_logits(
                          logits=self.D_logits_,
                          labels=tf.ones_like(self.D_logits_))))
      self.L_gen = tf.divide(self.L_gen,
                             tf.maximum(tf.reduce_sum(1 - self.aeReg_), 1))

      #最終的な誤差関数
      #beta はAdversial loss のβ?
      self.L = self.beta * self.L_gen + self.overall_loss

      cycle_local_sum = tf.summary.scalar("losses/cycle_local_loss",
                                          self.cycle_local_loss)
      cycle_global_sum = tf.summary.scalar("losses/cycle_global_loss",
                                           self.cycle_global_loss)
      interm_local_sum = tf.summary.scalar("losses/interm_local_loss",
                                           self.interm_local_loss)
      interm_global_sum = tf.summary.scalar("losses/interm_global_loss",
                                            self.interm_global_loss)
      twist_sum = tf.summary.scalar("losses/twist_loss", self.twist_loss)
      smooth_sum = tf.summary.scalar("losses/smoothness", self.smoothness)

      Ldiscreal_sum = tf.summary.scalar("losses/disc_real", self.L_disc_real)
      Ldiscfake_sum = tf.summary.scalar("losses/disc_fake", self.L_disc_fake)
      Lgen_sum = tf.summary.scalar("losses/disc_gen", self.L_gen)

      self.sum = tf.summary.merge([
          cycle_local_sum, cycle_global_sum, interm_local_sum,
          interm_global_sum, Lgen_sum, Ldiscreal_sum, Ldiscfake_sum, twist_sum,
          smooth_sum
      ])
      self.writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

      self.allvars = tf.trainable_variables()
      self.gvars = [v for v in self.allvars if "DIS" not in v.name]
      self.dvars = [v for v in self.allvars if "DIS" in v.name]

      #最適化手法を用意　(本プログラムでは2種のみ)
      if optim_name == "rmsprop":
        goptimizer = tf.train.RMSPropOptimizer(
            self.learning_rate, name="goptimizer")

        doptimizer = tf.train.RMSPropOptimizer(
            self.learning_rate, name="doptimizer")
      elif optim_name == "adam":
        goptimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1=0.5, name="goptimizer")

        doptimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1=0.5, name="doptimizer")
      else:
        raise Exception("Unknown optimizer")

      #compute_gradients　第一引数　誤差関数E(w)　第二引数　誤差関数の引数となる変数
      #gg 目的変数wの各点の値　ggradients　その各点に置けるE(w)の傾きの値　を返す
      ggradients, gg = zip(
          *goptimizer.compute_gradients(self.L, var_list=self.gvars))

      dgradients, dg = zip(
          *doptimizer.compute_gradients(self.L_disc, var_list=self.dvars))

      #＜クリッピング＞
      #各点の傾きを最大25に制限 それより大きければ　×(25/max{25, 勾配ノルムの分散})
      #降下方向は変えずに移動量だけを制限して
      #RNNは勾配ががくんと下がる時があるから? →RNNの非線形性
      #がけの付近で学習率小さくすればよいのでは？
      ggradients, _ = tf.clip_by_global_norm(ggradients, 25)
      dgradients, _ = tf.clip_by_global_norm(dgradients, 25)

      #再びオプティマイザーに発注してクラス内変数にセット
      self.goptim = goptimizer.apply_gradients(zip(ggradients, gg))
      self.doptim = doptimizer.apply_gradients(zip(dgradients, dg))

      num_param = 0
      for var in self.gvars:
        num_param += int(np.prod(var.get_shape()))
      print("NUMBER OF G PARAMETERS: " + str(num_param))

      num_param = 0
      for var in self.dvars:
        num_param += int(np.prod(var.get_shape()))
      print("NUMBER OF D PARAMETERS: " + str(num_param))

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

  def discriminator(self, input_, cond_):
    if self.norm_type == "batch_norm":
      from ops import batch_norm as norm
    elif self.norm_type == "instance_norm":
      from ops import instance_norm as norm
    else:
      raise Exception("Unknown normalization layer!!!")

    if not self.norm_type == "instance_norm":
      input_ = tf.nn.dropout(input_, self.kp_)

    #conv1Dはopsに定義
    if self.d_arch == 0:
      h0 = lrelu(conv1d(input_, 128, k_w=4, name="conv1d_h0"))
      h1 = lrelu(norm(conv1d(h0, 256, k_w=4, name="conv1d_h1"), "bn1"))
      h1 = tf.concat([h1, tf.tile(cond_, [1, int(h1.shape[1]), 1])], axis=-1)
      h2 = lrelu(norm(conv1d(h1, 512, k_w=4, name="conv1d_h2"), "bn2"))
      h2 = tf.concat([h2, tf.tile(cond_, [1, int(h2.shape[1]), 1])], axis=-1)
      h3 = lrelu(norm(conv1d(h2, 1024, k_w=4, name="conv1d_h3"), "bn3"))
      h3 = tf.concat([h3, tf.tile(cond_, [1, int(h3.shape[1]), 1])], axis=-1)
      logits = conv1d(h3, 1, k_w=4, name="logits", padding="VALID")
    elif self.d_arch == 1:
      h0 = lrelu(conv1d(input_, 64, k_w=4, name="conv1d_h0"))
      h1 = lrelu(norm(conv1d(h0, 128, k_w=4, name="conv1d_h1"), "bn1"))
      h1 = tf.concat([h1, tf.tile(cond_, [1, int(h1.shape[1]), 1])], axis=-1)
      h2 = lrelu(norm(conv1d(h1, 256, k_w=4, name="conv1d_h2"), "bn2"))
      h2 = tf.concat([h2, tf.tile(cond_, [1, int(h2.shape[1]), 1])], axis=-1)
      h3 = lrelu(norm(conv1d(h2, 512, k_w=4, name="conv1d_h3"), "bn3"))
      h3 = tf.concat([h3, tf.tile(cond_, [1, int(h3.shape[1]), 1])], axis=-1)
      logits = conv1d(h3, 1, k_w=4, name="logits", padding="VALID")
    elif self.d_arch == 2:
      h0 = lrelu(conv1d(input_, 32, k_w=4, name="conv1d_h0"))
      h1 = lrelu(norm(conv1d(h0, 64, k_w=4, name="conv1d_h1"), "bn1"))
      h1 = tf.concat([h1, tf.tile(cond_, [1, int(h1.shape[1]), 1])], axis=-1)
      h2 = lrelu(norm(conv1d(h1, 128, k_w=4, name="conv1d_h2"), "bn2"))
      h2 = tf.concat([h2, tf.tile(cond_, [1, int(h2.shape[1]), 1])], axis=-1)
      h3 = lrelu(norm(conv1d(h2, 256, k_w=4, name="conv1d_h3"), "bn3"))
      h3 = tf.concat([h3, tf.tile(cond_, [1, int(h3.shape[1]), 1])], axis=-1)
      logits = conv1d(h3, 1, k_w=4, name="logits", padding="VALID")
    elif self.d_arch == 3:
      h0 = lrelu(conv1d(input_, 16, k_w=4, name="conv1d_h0"))
      h1 = lrelu(norm(conv1d(h0, 32, k_w=4, name="conv1d_h1"), "bn1"))
      h1 = tf.concat([h1, tf.tile(cond_, [1, int(h1.shape[1]), 1])], axis=-1)
      h2 = lrelu(norm(conv1d(h1, 64, k_w=4, name="conv1d_h2"), "bn2"))
      h2 = tf.concat([h2, tf.tile(cond_, [1, int(h2.shape[1]), 1])], axis=-1)
      h3 = lrelu(norm(conv1d(h2, 128, k_w=4, name="conv1d_h3"), "bn3"))
      h3 = tf.concat([h3, tf.tile(cond_, [1, int(h3.shape[1]), 1])], axis=-1)
      logits = conv1d(h3, 1, k_w=4, name="logits", padding="VALID")
    else:
      raise Exception("Unknown discriminator architecture!!!")

    return tf.reshape(logits, [self.batch_size, 1])

  def train(self, sess, realSeq_, realSkel_, seqA_, skelA_, seqB_, skelB_,
            aeReg_, mask_, step):

    #辞書を作成
    feed_dict = dict()

    feed_dict[self.realSeq_] = realSeq_
    feed_dict[self.realSkel_] = realSkel_
    feed_dict[self.seqA_] = seqA_
    feed_dict[self.skelA_] = skelA_
    feed_dict[self.seqB_] = seqB_
    feed_dict[self.skelB_] = skelB_
    feed_dict[self.aeReg_] = aeReg_
    feed_dict[self.mask_] = mask_

    if self.fake_score > self.margin:
      print("update D")
      feed_dict[self.kp_] = 0.7
      cur_score = self.D_.eval(feed_dict=feed_dict).mean()
      self.fake_score = 0.99 * self.fake_score + 0.01 * cur_score
      _, summary_str = sess.run([self.doptim, self.sum], feed_dict=feed_dict)

    print("update G")
    feed_dict[self.kp_] = 1.0
    cur_score = self.D_.eval(feed_dict=feed_dict).mean()
    self.fake_score = 0.99 * self.fake_score + 0.01 * cur_score
    _, summary_str = sess.run([self.goptim, self.sum], feed_dict=feed_dict)

    self.writer.add_summary(summary_str, step)
    dlf, dlr, gl, lc = sess.run(
        [self.L_disc_fake, self.L_disc_real, self.L_gen, self.overall_loss],
        feed_dict=feed_dict)

    return dlf, dlr, gl, lc

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
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      if model_name is None: model_name = ckpt_name
      self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
      print("     Loaded model: " + str(model_name))
      return True, model_name
    else:
      return False, None
