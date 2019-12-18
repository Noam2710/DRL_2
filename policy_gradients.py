import gym
import numpy as np
import tensorflow as tf
import collections
from ModifiedTensorBoard import ModifiedTensorBoard
from datetime import datetime


env = gym.make('CartPole-v1')

np.random.seed(1)

# Define hyperparameters
state_size = 4
action_size = env.action_space.n
max_episodes = 100
max_steps = 501
discount_factor = 0.99
learning_rate_policy_network = 0.0001
learning_rate_value_network = 0.0001
neurones_policy = [64, 64, 64, 64, 64]
neurones_value = [64, 64, 64, 64, 64]
kernel_initializer = tf.contrib.layers.xavier_initializer()
render = False
algorithm = 2  # 1 for REINFORCE , 2 for REINFORCE WITH BASELINE , 3 for ACTOR-CRITIC


class PolicyActorNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            layer = tf.layers.dense(units=neurones_policy[0], inputs=self.state,
                                    kernel_initializer=kernel_initializer,
                                    activation=tf.nn.relu)

            for idx in range(1, len(neurones_policy)-1):
                layer = tf.layers.dense(units=neurones_policy[idx], inputs=layer,
                                        kernel_initializer=kernel_initializer,
                                        activation=tf.nn.relu)

            self.output = tf.layers.dense(units=action_size, inputs=layer,
                                          kernel_initializer=kernel_initializer,
                                          activation=None)

            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueCriticNetwork:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.value = tf.placeholder(dtype=tf.float32, name="value")

            layer = tf.layers.dense(units=neurones_value[0], inputs=self.state,
                                kernel_initializer=kernel_initializer,
                                activation=tf.nn.relu)

            for idx in range(1, len(neurones_value)-1):
                layer = tf.layers.dense(units=neurones_value[idx], inputs=layer,
                                    kernel_initializer=kernel_initializer,
                                    activation=tf.nn.relu)

            self.output = tf.layers.dense(units=action_size, inputs=layer,
                                          kernel_initializer=kernel_initializer,
                                          activation=None)

            self.value_estimate = tf.squeeze(self.output)
            self.loss = tf.squared_difference(self.value_estimate, self.value)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def print_tests_in_tensorboard(path_for_file_or_name_of_file=None, read_from_file=False, data_holder=None):
    if read_from_file:
        data_holder_to_visualize = np.load(path_for_file_or_name_of_file)
        name_of_log_dir = '{}-{}'.format(path_for_file_or_name_of_file.split("/")[3],
                                         path_for_file_or_name_of_file.split("/")[4])
        name_of_log_dir = name_of_log_dir.split('.')[0]
        name_of_log_dir += datetime.now().strftime("-%m-%d-%Y-%H-%M-%S")
    else:
        data_holder_to_visualize = data_holder
        name_of_log_dir = path_for_file_or_name_of_file

    tensorboard = ModifiedTensorBoard(log_dir="logs/{}".format(name_of_log_dir))

    for data_of_episode in data_holder_to_visualize:
        tensorboard.step = data_of_episode[0]
        tensorboard.update_stats(Number_of_steps=data_of_episode[1],
                                 average_rewards=data_of_episode[2])


def reinforce():
    while True:
        data_holder = []
        # Initialize the policy network
        tf.reset_default_graph()
        policy = PolicyActorNetwork(state_size, action_size, learning_rate_policy_network)
        Value = ValueCriticNetwork(state_size, learning_rate_value_network)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            solved = False
            Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
            episode_rewards = np.zeros(max_episodes)

            for episode in range(max_episodes):
                state = env.reset()
                state = state.reshape([1, state_size])
                episode_transitions = []

                for step in range(max_steps):
                    actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                    action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                    next_state, reward, done, _ = env.step(action)
                    next_state = next_state.reshape([1, state_size])

                    if render:
                        env.render()

                    action_one_hot = np.zeros(action_size)
                    action_one_hot[action] = 1
                    episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                    episode_rewards[episode] += reward

                    if done:
                        if episode > 98:
                            # Check if solved
                            average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                        else:
                            average_rewards = 0
                        data_holder.append([episode, step, average_rewards])
                        print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                        if average_rewards > 475 and episode > 786:
                            time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S-episode-break-{}".format(episode))
                            print_tests_in_tensorboard(
                                path_for_file_or_name_of_file="REINFORCE_{}_NEW_{}_{}".format(algorithm, episode, time),
                                data_holder=data_holder)
                            print(' Solved at episode: ' + str(episode))
                            solved = True
                        break
                    state = next_state

                if solved:
                    break

                # Compute Rt for each time-step t and update the network's weights
                for t, transition in enumerate(episode_transitions):
                    total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt

                    # REINFORCE WITH BASELINE
                    if algorithm == 2:
                        baseline = sess.run(Value.value_estimate, {Value.state: transition.state})

                        advantage = total_discounted_return - baseline

                        feed_dict = {Value.state: transition.state, Value.value: total_discounted_return}
                        _, loss = sess.run([Value.optimizer, Value.loss], feed_dict)

                        feed_dict = {policy.state: transition.state, policy.R_t: advantage, policy.action: transition.action}
                        _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)

                    # REINFORCE
                    else:
                        feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return, policy.action: transition.action}
                        _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)


def actor_critic():
    while True:
        data_holder = []
        # Initialize the networks
        tf.reset_default_graph()
        policy = PolicyActorNetwork(state_size, action_size, learning_rate_policy_network)
        Value = ValueCriticNetwork(state_size, learning_rate_value_network)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            solved = False
            Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
            episode_rewards = np.zeros(max_episodes)

            for episode in range(max_episodes):
                state = env.reset()
                state = state.reshape([1, state_size])
                episode_transitions = []

                for step in range(max_steps):
                    actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                    action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                    next_state, reward, done, _ = env.step(action)
                    next_state = next_state.reshape([1, state_size])

                    if render:
                        env.render()

                    action_one_hot = np.zeros(action_size)
                    action_one_hot[action] = 1
                    episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                    episode_rewards[episode] += reward

                    value_nex_state = 0 if done else sess.run(Value.value_estimate, {Value.state: next_state})
                    td_target = reward + discount_factor * value_nex_state
                    td_error = td_target - sess.run(Value.value_estimate, {Value.state: state})

                    feed_dict = {Value.state: state, Value.value: td_target}
                    _, loss = sess.run([Value.optimizer, Value.loss], feed_dict)

                    feed_dict = {policy.state: state, policy.R_t: td_error,
                                 policy.action: action_one_hot}
                    _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)

                    if done:
                        if episode > 98:
                            # Check if solved
                            average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                        else:
                            average_rewards = 0
                        data_holder.append([episode, step, average_rewards])
                        print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                        if average_rewards > 475:
                            time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S-episode-break-{}".format(episode))
                            print_tests_in_tensorboard(
                                path_for_file_or_name_of_file="ACTOR_CRITIC_{}_{}".format(episode, time),
                                data_holder=data_holder)
                            print(' Solved at episode: ' + str(episode))
                            solved = True
                        break
                    state = next_state

                if solved:
                    break


# Algorithm execution
if algorithm in [1, 2]:
    reinforce()
else:
    actor_critic()