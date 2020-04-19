
import tqdm
import pickle5 as pickle
import random
import pyspiel
import numpy as np
import tensorflow as tf
import copy

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent

import parameters

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(2)
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                time_step = env.step([agent_output.action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1
    return wins / num_episodes


if __name__ == '__main__' :
    game = parameters.game
    num_players = 2
    env = rl_environment.Environment(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = parameters.hidden_layers_sizes
    replay_buffer_capacity = int(1e4)
    train_episodes = 500000
    loss_report_interval = 1000
    save_every_n_step = 1000

    win_rates_history = []

    with tf.Session() as sess:
        dqn_agents = [dqn.DQN(
            sess,
            player_id=idx,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity)
            for idx in range(num_players)]
        #si tu veux restorer les agents pour continuer de l'entrainer, tu peux décommenter les 2 lignes en dessous
        #for i in range(num_players) :
        #    dqn_agents[i].restore("agents/better_puissance4/", str(i), "99")
        rand_agents = [
                random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
                for idx in range(num_players)
                ]
        sess.run(tf.compat.v1.global_variables_initializer())


        #trianing
        for ep in range(train_episodes) :
            if (ep + 1) % save_every_n_step == 0 :
                win_rates = eval_against_random_bots(env, dqn_agents, rand_agents, 100)
                win_rates_history.append(win_rates)
                print("Episode {} of {}, win {}".format(ep+1, train_episodes, win_rates))
                for i, agent in enumerate(dqn_agents) :
                    agent.save(parameters.agent_path, str(i), str(int(ep/save_every_n_step)))
            time_step = env.reset()
            while not time_step.last() :
                player_id = time_step.observations["current_player"]
                agent_output = dqn_agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            for agent in dqn_agents :
                agent.step(time_step)

        #si tu veux save et visualiser qui a gagné quoi pendant le train
        #with open("win_history.pkl", "wb") as f :
        #    pickle.dump(win_rates_history, f)



