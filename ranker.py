
import sys
import tqdm
import random
import pyspiel
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle5 as pickle

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent

import parameters

from parameters import JJ


#effectue le softmax des q_values
def create_prob_vector(q_values) :
    q_values = q_values.copy()
    sum_of_prob = sum(np.e**(value) for value in q_values)
    for i in range(len(q_values)) :
        q_values[i] = np.e**(q_values[i])/sum_of_prob
    return q_values

def knowledge(probs, alpha) :
    for i in range(len(probs)) :
        probs[i] = probs[i] + (1-alpha) * ((1/len(probs)) - probs[i])
    return probs

def rationality(probs, beta) :
    if beta == 1 :
        new_probs = np.zeros(len(probs))
        new_probs[np.argmax(probs)] = 1.0
        return new_probs
    for i in range(len(probs)) :
        probs[i] = max(0, probs[i] - beta * np.amax(probs))
    probs = probs / sum(probs)
    return probs

#ce qui permet de dégrader le vecteur de décision
def better_rationality(probs, beta) :
    probs = probs.copy() # ?
    threshold = max(probs) * beta
    for i in range(len(probs)) :
        if probs[i] < threshold :
            probs[i] = 0;

    sum_of_remaining_probs = sum(probs)
    for i in range(len(probs)) :
        probs[i] = probs[i] / sum_of_remaining_probs

    return probs

#smurf est la mesure de la somme des probabilité moins la probabilité maximale
def smurf(probs_before, probs_after) :
    max_prob = max(probs_before)
    numerateur = sum(probs_before[i] if probs_after[i] else 0 for i in range(len(probs_before))) - max_prob
    denomintateur = sum(probs_before) - max_prob
    if max_prob == 1 :
        return 0.0
    else :
        return numerateur/denomintateur

#la fonction d'opposition des agents et qui retourne les victoires des bots
def eval_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    wins = np.zeros(2)
    jjs = [trained_agents, random_agents]
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents.agents[0], random_agents.agents[1]]
        else:
            cur_agents = [random_agents.agents[0], trained_agents.agents[1]]
        for _ in range(num_episodes) :
            time_step = env.reset()
            games = [[], []]
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                info_state = time_step.observations["info_state"][0]
                legal_actions = time_step.observations["legal_actions"][player_id]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True, q_distrib=True)
                probs = create_prob_vector(agent_output.probs)


                #probs = knowledge(probs, jjs[player_id].alpha)
                probs_rational = better_rationality(probs, jjs[player_id].beta)
                jjs[player_id].smurf_values.append(smurf(probs, probs_rational))

                action = legal_actions[int(np.random.choice(len(probs_rational), 1, p=probs_rational)[0])]

                games[player_id].append((info_state, probs, legal_actions, action))

                time_step = env.step([action])
            if time_step.rewards[player_pos] > 0:
                wins[player_pos] += 1

        trained_agents.games.append(games[0])
        random_agents.games.append(games[1])


    return wins, wins[0] == wins[1]

#cherche les indices des joueurs dans le delta_rank specifié
def find_player_range(jjs, delta_rank, jj_index) :
    max_index = jj_index
    min_index = jj_index
    for i in range(jj_index, len(jjs)) :
        if jjs[i].rank() > jjs[jj_index].rank() + delta_rank :
            break
        else :
            max_index = i
    for i in range(jj_index, -1, -1) :
        if jjs[i].rank() < jjs[jj_index].rank() - delta_rank :
            break
        else :
            min_index = i

    return min_index, max_index

if __name__ == '__main__' :
    game = parameters.game
    num_players = 2
    env = rl_environment.Environment(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = parameters.hidden_layers_sizes
    replay_buffer_capacity = int(1e4)
    train_episodes = 50000
    loss_report_interval = 1000
    delta_rank = 10
    max_rank = 20
    min_rank = -20

    with tf.Session() as sess :
        sess.run(tf.compat.v1.global_variables_initializer())
        dqn_agents = [dqn.DQN(
            sess,
            player_id=idx,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity) for idx in range(num_players)]

        stds = []
        #cette boucle for permet d'effectuer un ladder sur les versions de l'agent voulu : si on ne veut rank qu'une seule version de l'agent, utiliser la premier ligne, sinon, la 2nd
        for version in [159] :
        #for version in range(1, 160, 5) :
            jjs = []
            for i in range(num_players) :
                dqn_agents[i].restore(parameters.agent_path, str(i), str(version))

            #creation de la population de jean jacques, avec son
            beta_min = 0.7
            beta_max = 1
            nb_jjs = 500
            for beta, rank in zip(np.linspace(beta_min, beta_max, num=nb_jjs, endpoint=True), np.linspace(min_rank, max_rank, num=500, endpoint=True)) :
                jjs.append(JJ(dqn_agents, 1, beta))
                jjs[-1].set_rank(int(rank))

            #la boucle qui construit le ladder
            for fight in tqdm.tqdm(range(50)) :
                for jj_index in range(len(jjs)) :
                    min_index, max_index = find_player_range(jjs, delta_rank, jj_index)
                    if min_index == max_index :
                        break
                    else :
                        while True :
                            challenger_index = random.randint(min_index, max_index)
                            if challenger_index != jj_index :
                                break
                    j1 = jjs[jj_index]
                    j2 = jjs[challenger_index]
                    result, draw = eval_bots(env, j1, j2, 1)
                    if not draw :
                        winner = np.argmax(result)
                        if [j1, j2][winner].rank() != max_rank :
                            [j1, j2][winner].update_rank(1)
                        if [j1, j2][1 - winner].rank() != min_rank :
                            [j1, j2][1 - winner].update_rank(-1)
                    jjs.sort(key=lambda x: x.rank())

            #calcul de déviation standart
            x, y = [i for i in range(len(jjs))], [joueur.beta for joueur in jjs]
            deg1, deg0 = np.polyfit(x, y, 1)
            line = [i*deg1 + deg0 for i in range(len(jjs))]
            canard = [y[i]-line[i] for i in range(len(jjs))]
            stds.append((version, np.std(canard)))

    #sauvegarde de la liste stds, qui contient les déviations standart d'un ladder à differente version de l'agent
    with open('data/stds.stat', 'wb') as f :
        pickle.dump(stds, f)

    for joueur in jjs :
        del joueur.agents
    with open(parameters.jjs_path, 'wb') as f :
        pickle.dump(jjs, f)


