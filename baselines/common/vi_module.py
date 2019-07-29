import tensorflow as tf
import numpy as np
from baselines.common.tf_util import huber_loss
from baselines.a2c.utils import ortho_init


def vi_trans(s, n_actions, hidden_dim=None):
    state_shape = s.shape.as_list()[1:]
    hidden_dim = state_shape[-1] if hidden_dim is None else hidden_dim

    fc1 = tf.layers.dense(s, n_actions * hidden_dim, activation='relu', name='fc1')
    fc1 = tf.reshape(tf.reshape(fc1, [-1, n_actions, hidden_dim]), [-1, hidden_dim])
    fc2 = tf.layers.dense(fc1, hidden_dim, activation='relu', name='fc2')
    attention = tf.layers.dense(fc2, hidden_dim, activation='sigmoid',
                                name='attention')
    fc3 = tf.layers.dense(fc2, state_shape[-1], activation='relu', name='fc3')
    # new_state = bypass + attention * state_trans
    fc3 = tf.transpose(tf.reshape(fc3, [-1, n_actions, hidden_dim]), [1, 0, 2])
    attention = tf.transpose(tf.reshape(attention, [-1, n_actions, hidden_dim]), [1, 0, 2])
    x = fc3 * attention + s
    x = tf.reshape(tf.transpose(x, [1, 0, 2]), [-1, hidden_dim])
    return x


def rollout_step(s, a, n_actions):
    state_shape = s.shape.as_list()[1:]
    op_trans_state = tf.make_template('trans_state', vi_trans, n_actions=n_actions)
    r = tf.layers.dense(s, n_actions, activation=None, name='reward', kernel_initializer=ortho_init(1))
    s = op_trans_state(s)
    v = tf.reshape(tf.layers.dense(s, 1, activation=None, name='value', kernel_initializer=ortho_init(1)),
                   [-1, n_actions])
    if a is not None:
        l = tf.expand_dims(tf.range(0, tf.shape(r)[0]), 1)
        l = tf.concat([l, tf.dtypes.cast(tf.reshape(a, [-1, 1]), tf.int32)], axis=1)
        r = tf.gather_nd(r, l)
        s = tf.gather_nd(tf.reshape(s, [-1, n_actions] + state_shape), l)
        v = tf.gather_nd(v, l)
    return r, v, s

    # Action-conditional rollout over time for training


def training_loss(op_rollout, s_history, a_history, v_history, r_history, nenvs, nstep, training_depth=1):
    state_shape = s_history.shape.as_list()[1:]
    # for each envs, carry out same action for nstep
    r_vi, v_vi, s_vi = [], [], []

    # value iteration: expand for all possible action
    l = tf.expand_dims(tf.range(0, nenvs), 1)
    l = tf.concat([l, tf.tile([[0]], [nenvs, 1])], axis=1)
    s = tf.gather_nd(tf.reshape(s_history, [nenvs, nstep, -1]), l)
    a = tf.gather_nd(tf.reshape(a_history, [nenvs, nstep]), l)
    for i in range(training_depth):
        s_vi.append(s)
        r, v, s = op_rollout(s, a)
        r_vi.append(r)
        v_vi.append(v)
    r_vi = tf.stack(r_vi, axis=1)
    v_vi = tf.stack(v_vi, axis=1)
    s_vi = tf.stack(s_vi, axis=1)

    s_history = tf.reshape(s_history, [nenvs, 1, nstep, -1])
    v_history = tf.reshape(v_history, [nenvs, 1, nstep])
    r_history = tf.reshape(r_history, [nenvs, 1, nstep])
    s_vi = tf.reshape(s_vi, [nenvs, training_depth, 1, -1])
    v_vi = tf.reshape(v_vi, [nenvs, training_depth, 1])
    r_vi = tf.reshape(r_vi, [nenvs, training_depth, 1])
    # use the upper triangular part
    idx = np.flip(np.triu(np.ones([training_depth, nstep])), 1)
    idx = np.where(idx.reshape([-1]))[0]
    l = np.repeat(np.arange(nenvs), idx.size)
    l = np.stack([l, np.tile(idx, nenvs)], axis=1)

    s_mat = tf.gather_nd(tf.reshape(s_history - s_vi, [nenvs, training_depth * nstep, state_shape[-1]]), l)
    r_mat = tf.gather_nd(tf.reshape(r_history - r_vi, [nenvs, -1]), l)
    v_mat = tf.gather_nd(tf.reshape(v_history - v_vi, [nenvs, -1]), l)
    # # bn before loss
    # r_mat = r_bn(r_mat)
    # v_mat = v_bn(v_mat)
    # compute loss
    s_loss = tf.math.reduce_mean(huber_loss(s_mat))
    r_loss = tf.math.reduce_mean(huber_loss(r_mat))
    v_loss = tf.math.reduce_mean(huber_loss(v_mat))
    return r_loss + v_loss  # + s_loss


def VI_module(env, state, op_rollout, expand_depth=1, expand_breadth=4, lookahead_depth=1, gamma=0.99):
    # value iteration: expand for all possible action
    n_actions = env.action_space.n
    n_envs = state.shape.as_list()[0]
    state_shape = state.shape.as_list()[1:]
    expand_breadth = min(expand_breadth, n_actions)
    lookahead_depth = max(lookahead_depth, expand_depth)

    # tree expansion for inference
    q_rollout = []
    r_rollout = []
    v_rollout = []
    idx_rollout = []
    s_rollout = []
    s = state
    for i in range(lookahead_depth):
        # forward
        r, v, s = op_rollout(s, None)
        q = r + gamma * v

        # only record top k terms with k=expand_breadth or 1
        b = expand_breadth if i < expand_depth else 1
        _, idx = tf.nn.top_k(q, k=b)

        l = tf.tile(tf.expand_dims(tf.range(0, tf.shape(idx)[0]), 1), [1, b])
        l = tf.concat([tf.reshape(l, [-1, 1]), tf.reshape(idx, [-1, 1])], axis=1)

        v = tf.gather_nd(v, l)
        s = tf.gather_nd(tf.reshape(s, [-1, n_actions] + state_shape), l)
        r = tf.gather_nd(r, l)
        q = tf.gather_nd(q, l)

        r_rollout.append(r)
        v_rollout.append(v)
        s_rollout.append(s)
        q_rollout.append(q)
        idx_rollout.append(idx)

    # Backup
    # tree aggregation
    v_plan = [None] * lookahead_depth
    q_plan = [None] * lookahead_depth

    v_plan[-1] = v_rollout[-1]
    for i in reversed(range(lookahead_depth)):
        q_plan[i] = r_rollout[i] + gamma * v_plan[i]
        if i > 0:
            b = expand_breadth if i < expand_depth else 1
            q_max = tf.reduce_max(tf.reshape(q_plan[i], [-1, b]), axis=1)
            n = float(lookahead_depth - i)
            v_plan[i - 1] = (v_rollout[i - 1] + q_max * n) / (n + 1)

    for i in range(lookahead_depth):
        q_plan[i] = tf.reshape(q_plan[i], [n_envs, -1])
        v_plan[i] = tf.reshape(v_plan[i], [n_envs, -1])

    return q_plan, v_plan  # , op_rollout
