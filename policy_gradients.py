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

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate_policy_network = 0.0005
learning_rate_value_network = 0.0005


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 24], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [24], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [24, self.action_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.value = tf.placeholder(dtype=tf.float32, name="value")
            neurons = 10
            self.W1 = tf.get_variable("W1", [self.state_size, neurons], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            self.b1 = tf.get_variable("b1", [neurons], initializer=tf.zeros_initializer())

            # self.W2 = tf.get_variable("W2", [24, 1], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            # self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(self.A1, self.b1)

            self.value_estimate = tf.squeeze(self.output)
            # Loss with negative log probability
            self.loss = tf.squared_difference(self.value_estimate, self.value)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)



def print_tests_in_TensorBoard(path_for_file_or_name_of_file=None, read_from_file=False, data_holder=None):
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


render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate_policy_network)
Value = ValueNetwork(state_size, learning_rate_value_network)


# Start training the agent with REINFORCE algorithm

def reinforce():
    while True:
        data_holder = []
        # Initialize the policy network
        tf.reset_default_graph()
        policy = PolicyNetwork(state_size, action_size, learning_rate_policy_network)
        Value = ValueNetwork(state_size, learning_rate_value_network)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            solved = False
            Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
            episode_rewards = np.zeros(max_episodes)
            average_rewards = 0.0

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
                        if average_rewards > 475:
                            time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S-episode-break-{}".format(episode))
                            print_tests_in_TensorBoard(
                                path_for_file_or_name_of_file="REINFORCE_WITH_BASELINE_{}_{}".format(episode, time),
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
                    baseline = sess.run(Value.value_estimate, {Value.state: transition.state})

                    advantage = total_discounted_return - baseline

                    feed_dict = {Value.state: transition.state, Value.value: total_discounted_return}
                    _, loss = sess.run([Value.optimizer, Value.loss], feed_dict)

                    feed_dict = {policy.state: transition.state, policy.R_t: advantage, policy.action: transition.action}
                    _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)


def actor_critic():
    while True:
        data_holder = []
        # Initialize the policy network
        tf.reset_default_graph()
        policy = PolicyNetwork(state_size, action_size, learning_rate_policy_network)
        Value = ValueNetwork(state_size, learning_rate_value_network)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            solved = False
            Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
            episode_rewards = np.zeros(max_episodes)
            average_rewards = 0.0

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

                    value_nex_state = sess.run(Value.value_estimate, {Value.state: next_state})
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
                            print_tests_in_TensorBoard(
                                path_for_file_or_name_of_file="REINFORCE_WITH_BASELINE_{}_{}".format(episode, time),
                                data_holder=data_holder)
                            print(' Solved at episode: ' + str(episode))
                            solved = True
                        break
                    state = next_state

                if solved:
                    break


actor_critic()