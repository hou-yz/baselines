import tensorflow as tf


class VI_module(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, state, expand_depth=2, expand_breadth=4, lookahead_depth=2, gamma=0.99, sess=None):

        self.n_actions = env.action_space.n
        self.state_dim = state.shape.as_list()[1]
        self.expand_depth = expand_depth
        self.expand_breadth = expand_breadth
        self.lookahead_depth = lookahead_depth
        self.gamma = gamma
        self.sess = sess

        # value iteration: expand for all possible action
        self.vi_trans = tf.keras.layers.Dense(self.n_actions * self.state_dim, activation='relu',
                                              input_shape=(self.state_dim,), name='vi/trans')
        self.vi_value = tf.keras.layers.Dense(1, activation=None, input_shape=(self.state_dim,),
                                              name='vi/value')
        self.vi_reward = tf.keras.layers.Dense(self.n_actions, activation=None, input_shape=(self.state_dim,),
                                               name='vi/reward')

    def rollout(self, s, a=None):
        r = self.vi_reward(s)
        s = self.vi_trans(s)
        v = tf.reshape(self.vi_value(tf.reshape(s, [-1, self.state_dim])), [-1, self.n_actions])
        if a is not None:
            l = tf.expand_dims(tf.range(0, tf.shape(r)[0]), 1)
            l = tf.concat([l, tf.dtypes.cast(tf.reshape(a, [-1, 1]), tf.int32)], axis=1)
            r = tf.gather_nd(r, l)
            s = tf.gather_nd(tf.reshape(s, [-1, self.n_actions, self.state_dim]), l)
            v = tf.gather_nd(v, l)
        return r, v, s

    def training_fwd(self, s_history, a_history, v_history, r_history, nenvs, nstep):
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
        r_mat = tf.matrix_band_part(tf.reverse(r_history - r_vi, [-1]), 0, -1)
        v_mat = tf.matrix_band_part(tf.reverse(v_history - v_vi, [-1]), 0, -1)
        r_loss = tf.reduce_sum(tf.math.pow(r_mat, 2))
        v_loss = tf.reduce_sum(tf.math.pow(v_mat, 2))
        return r_loss + v_loss

    def __call__(self, s, **kwargs):
        # tree expansion
        q_rollout = []
        r_rollout = []
        v_rollout = []
        idx_rollout = []
        s_rollout = []

        for i in range(self.lookahead_depth):
            # forward
            r, v, s = self.rollout(s)
            q = r + self.gamma * v

            # only record top k terms with k=expand_breadth or 1
            b = min(self.expand_breadth, self.n_actions) if i < self.expand_depth else 1
            _, idx = tf.nn.top_k(q, k=b)
            l = tf.tile(tf.expand_dims(tf.range(0, tf.shape(idx)[0]), 1), [1, b])
            l = tf.concat([tf.reshape(l, [-1, 1]), tf.reshape(idx, [-1, 1])], axis=1)

            v = tf.gather_nd(v, l)
            s = tf.gather_nd(tf.reshape(s, [-1, self.n_actions, self.state_dim]), l)
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

        return q_plan, v_plan
