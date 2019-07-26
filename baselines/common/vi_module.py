import numpy as np
import tensorflow as tf

from baselines.a2c.utils import ortho_init
from baselines.common.tf_util import huber_loss


class VI_trans(object):
    def __init__(self, n_actions, state_shape, hidden_dim=None, type='fc'):
        self.type = type
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.hidden_dim = state_shape[-1] if hidden_dim is None else hidden_dim
        if self.type == 'fc':
            self.fc1 = tf.keras.layers.Dense(self.n_actions * self.hidden_dim, activation='relu', name='vi/trans/fc1')
            self.fc2 = tf.keras.layers.Dense(self.hidden_dim, activation='relu', name='vi/trans/fc2')
            self.attention = tf.keras.layers.Dense(self.hidden_dim, activation='sigmoid',
                                                   name='vi/trans/attention')
            self.fc3 = tf.keras.layers.Dense(self.state_shape[-1], activation='relu', name='vi/trans/fc3')
        elif self.type == 'conv':
            self.conv1 = tf.keras.layers.Conv2D(self.n_actions * self.hidden_dim, 3, 1, activation='relu',
                                                name='vi/trans/conv1')
            self.conv2 = tf.keras.layers.Conv2D(self.hidden_dim, 3, 1, activation='relu', name='vi/trans/conv2')
            self.attention = tf.keras.layers.Conv2D(self.hidden_dim, 1, 1, activation='sigmoid',
                                                    name='vi/trans/attention')
            self.conv3 = tf.keras.layers.Conv2D(self.state_shape[-1], 3, 1, activation='relu', name='vi/trans/conv3')
        else:
            raise Exception('only support fc & conv for state transition')

    def __call__(self, s):
        if self.type == 'fc':
            # n_batch, n_action, dim -> n_batch*n_action, dim
            x = tf.reshape(tf.reshape(self.fc1(s), [-1, self.n_actions, self.hidden_dim]), [-1, self.hidden_dim])
            x = self.fc2(x)
            att = self.attention(x)
            x = self.fc3(x)
            # new_state = bypass + attention * state_trans
            x = tf.transpose(tf.reshape(x, [-1, self.n_actions, self.hidden_dim]), [1, 0, 2])
            att = tf.transpose(tf.reshape(att, [-1, self.n_actions, self.hidden_dim]), [1, 0, 2])
            x = x * att + s
            x = tf.reshape(tf.transpose(x, [1, 0, 2]), [-1, self.hidden_dim])
        elif self.type == 'conv':
            # n_batch, n_action, h, w, dim -> n_batch*n_action, h, w, dim
            [h, w] = self.state_shape[0:2]
            x = tf.reshape(
                tf.transpose(tf.reshape(self.conv1(s), [-1, h, w, self.n_actions, self.hidden_dim]), [0, 3, 1, 2, 4]),
                [-1, h, w, self.hidden_dim])
            x = self.conv2(x)
            att = self.attention(x)
            x = self.conv3(x)
            # new_state = bypass + attention * state_trans
            x = tf.transpose(tf.reshape(x, [-1, self.n_actions, h, w, self.hidden_dim]), [1, 0, 2, 3, 4])
            att = tf.transpose(tf.reshape(att, [-1, self.n_actions, h, w, self.hidden_dim]), [1, 0, 2, 3, 4])
            x = x * att + s
            x = tf.reshape(tf.transpose(x, [1, 0, 2, 3, 4]), [-1, h, w, self.hidden_dim])
        else:
            raise Exception('only support fc & conv for state transition')
        return x


class VI_module(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, state, expand_depth=1, expand_breadth=4, lookahead_depth=1, gamma=0.99):
        self.n_actions = env.action_space.n
        self.state_shape = state.shape.as_list()[1:]
        self.expand_depth = expand_depth
        self.expand_breadth = min(expand_breadth, self.n_actions)
        self.lookahead_depth = max(lookahead_depth, expand_depth)
        self.gamma = gamma

        # value iteration: expand for all possible action
        self.vi_trans = VI_trans(self.n_actions, self.state_shape)
        self.vi_value = tf.keras.layers.Dense(1, activation=None, name='vi/value', kernel_initializer=ortho_init(1))
        self.vi_reward = tf.keras.layers.Dense(self.n_actions, activation=None, name='vi/reward')
        self.vi_pi = tf.keras.layers.Dense(self.n_actions, activation=None, name='pi',
                                           kernel_initializer=ortho_init(0.01))

        # batch norm layer for vi_loss computation
        self.r_bn = tf.keras.layers.BatchNormalization()
        self.v_bn = tf.keras.layers.BatchNormalization()

        # vi latent layer
        self.vi_latent = tf.keras.layers.Dense(64, activation='relu', name='vi/latent')

    def rollout(self, s, a=None):
        r = self.vi_reward(s)
        s = self.vi_trans(s)
        v = tf.reshape(self.vi_value(s), [-1, self.n_actions])
        if a is not None:
            l = tf.expand_dims(tf.range(0, tf.shape(r)[0]), 1)
            l = tf.concat([l, tf.dtypes.cast(tf.reshape(a, [-1, 1]), tf.int32)], axis=1)
            r = tf.gather_nd(r, l)
            s = tf.gather_nd(tf.reshape(s, [-1, self.n_actions] + self.state_shape), l)
            v = tf.gather_nd(v, l)
        return r, v, s

    def training_loss(self, s_history, a_history, v_history, r_history, nenvs, nstep):
        # for each envs, carry out same action for nstep
        r_vi, v_vi, s_vi = [], [], []
        l = tf.expand_dims(tf.range(0, nenvs), 1)
        l = tf.concat([l, tf.tile([[0]], [nenvs, 1])], axis=1)
        s = tf.gather_nd(tf.reshape(s_history, [nenvs, nstep, -1]), l)
        a = tf.gather_nd(tf.reshape(a_history, [nenvs, nstep]), l)
        for i in range(nstep):
            s_vi.append(s)
            r, v, s = self.rollout(s, a)
            r_vi.append(r)
            v_vi.append(v)
        r_vi = tf.stack(r_vi, axis=1)
        v_vi = tf.stack(v_vi, axis=1)
        s_vi = tf.stack(s_vi, axis=1)

        s_history = tf.reshape(s_history, [nenvs, 1, nstep, -1])
        v_history = tf.reshape(v_history, [nenvs, 1, nstep])
        r_history = tf.reshape(r_history, [nenvs, 1, nstep])
        s_vi = tf.reshape(s_vi, [nenvs, nstep, 1, -1])
        v_vi = tf.reshape(v_vi, [nenvs, nstep, 1])
        r_vi = tf.reshape(r_vi, [nenvs, nstep, 1])
        # use the upper triangular part
        idx = np.flip(np.triu(np.ones([nstep, nstep])), 1)
        idx[self.lookahead_depth:, :] = 0  # iterate at most self.lookahead_depth times
        idx = np.where(idx.reshape([-1]))[0]
        l = np.repeat(np.arange(nenvs), idx.size)
        l = np.stack([l, np.tile(idx, nenvs)], axis=1)

        s_mat = tf.gather_nd(tf.reshape(s_history - s_vi, [nenvs, nstep * nstep, self.state_shape[-1]]), l)
        r_mat = tf.gather_nd(tf.reshape(r_history - r_vi, [nenvs, -1]), l)
        v_mat = tf.gather_nd(tf.reshape(v_history - v_vi, [nenvs, -1]), l)
        # # bn before loss
        # r_mat = self.r_bn(r_mat)
        # v_mat = self.v_bn(v_mat)
        # compute loss
        s_loss = tf.math.reduce_sum(huber_loss(s_mat))
        r_loss = tf.math.reduce_sum(huber_loss(r_mat))
        v_loss = tf.math.reduce_sum(huber_loss(v_mat))
        return r_loss + v_loss  # + s_loss

    def __call__(self, s, **kwargs):
        # tree expansion
        q_rollout = []
        r_rollout = []
        v_rollout = []
        idx_rollout = []
        s_rollout = []
        n_envs = s.shape.as_list()[0]

        for i in range(self.lookahead_depth):
            # forward
            r, v, s = self.rollout(s)
            q = r + self.gamma * v

            # only record top k terms with k=expand_breadth or 1
            b = self.expand_breadth if i < self.expand_depth else 1
            _, idx = tf.nn.top_k(q, k=b)
            l = tf.tile(tf.expand_dims(tf.range(0, tf.shape(idx)[0]), 1), [1, b])
            l = tf.concat([tf.reshape(l, [-1, 1]), tf.reshape(idx, [-1, 1])], axis=1)

            v = tf.gather_nd(v, l)
            s = tf.gather_nd(tf.reshape(s, [-1, self.n_actions] + self.state_shape), l)
            r = tf.gather_nd(r, l)
            q = tf.gather_nd(q, l)

            r_rollout.append(r)
            v_rollout.append(v)
            s_rollout.append(s)
            q_rollout.append(q)
            idx_rollout.append(idx)

        # tree aggregation
        v_plan = [None] * self.lookahead_depth
        q_plan = [None] * self.lookahead_depth

        v_plan[-1] = v_rollout[-1]
        for i in reversed(range(self.lookahead_depth)):
            q_plan[i] = r_rollout[i] + self.gamma * v_plan[i]
            if i > 0:
                b = self.expand_breadth if i < self.expand_depth else 1
                q_max = tf.reduce_max(tf.reshape(q_plan[i], [-1, b]), axis=1)
                n = float(self.lookahead_depth - i)
                v_plan[i - 1] = (v_rollout[i - 1] + q_max * n) / (n + 1)

        for i in range(self.lookahead_depth):
            q_plan[i] = tf.reshape(q_plan[i], [n_envs, -1])
            v_plan[i] = tf.reshape(v_plan[i], [n_envs, -1])

        q_latent = self.vi_latent(q_plan[0])

        return q_latent, q_plan, v_plan
