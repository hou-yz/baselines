import tensorflow as tf


class VI_module(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, state, expand_depth=2, expand_breadth=4, lookahead_depth=5, gamma=0.98):

        self.n_actions = env.action_space.n
        self.state_dim = state.shape.as_list()[1]
        self.expand_depth = expand_depth
        self.expand_breadth = expand_breadth
        self.lookahead_depth = lookahead_depth
        self.gamma = gamma

        # value iteration: expand for all possible action
        self.vi_value = tf.keras.layers.Dense(self.n_actions, activation='relu', input_shape=(self.state_dim,),
                                              name='vi/value')
        self.vi_trans = tf.keras.layers.Dense(self.n_actions * self.state_dim, activation='relu',
                                              input_shape=(self.state_dim,), name='vi/trans')
        self.vi_reward = tf.keras.layers.Dense(self.n_actions, activation='relu', input_shape=(self.state_dim,),
                                               name='vi/reward')
        self.vi_timestep = tf.keras.layers.Dense(1, activation='relu', input_shape=(self.state_dim,),
                                                 name='vi/timestep')

    def __call__(self, state, **kwargs):
        batch_size = state.shape.as_list()[0]

        # tree expansion
        q_rollout = []
        r_rollout = []
        g_rollout = []
        v_rollout = []
        idx_rollout = []
        s_rollout = []

        s = state
        for i in range(self.lookahead_depth):
            # forward
            v = self.vi_value(s)
            s = self.vi_trans(s)
            r = self.vi_reward(s)
            t = self.vi_timestep(s)
            g = tf.pow(tf.constant(self.gamma), t)
            q = r + g * v

            # only record top k terms with k=expand_breadth or 1
            b = min(self.expand_breadth, self.n_actions) if i < self.expand_depth else 1
            _, idx = tf.nn.top_k(q, k=b)
            l = tf.tile(tf.expand_dims(tf.range(0, tf.shape(idx)[0]), 1), [1, b])
            l = tf.concat([tf.reshape(l, [-1, 1]), tf.reshape(idx, [-1, 1])], axis=1)

            v = tf.gather_nd(v, l)
            s = tf.gather_nd(tf.reshape(s, [-1, self.n_actions, self.state_dim]), l)
            r = tf.gather_nd(r, l)
            g = tf.gather_nd(g, l)
            q = tf.gather_nd(q, l)

            r_rollout.append(r)
            v_rollout.append(v)
            s_rollout.append(s)
            g_rollout.append(g)
            q_rollout.append(q)
            idx_rollout.append(idx)

        # tree aggregation
        v_plan = [None] * self.lookahead_depth
        q_plan = [None] * self.lookahead_depth

        v_plan[-1] = v_rollout[-1]
        for i in reversed(range(self.lookahead_depth)):
            q_plan[i] = r_rollout[i] + g_rollout[i] * v_plan[i]
            if i > 0:
                b = self.expand_breadth if i < self.expand_depth else 1
                q_max = tf.reduce_max(tf.reshape(q_plan[i], [-1, b]), axis=1)
                n = float(self.lookahead_depth - i)
                v_plan[i - 1] = (v_rollout[i - 1] + q_max * n) / (n + 1)

        return q_plan, v_plan
