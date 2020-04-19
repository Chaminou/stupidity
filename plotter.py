
import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import tqdm
from scipy.optimize import curve_fit

import parameters
from parameters import JJ

def pretty_board(info_state):
    """Returns the board in `time_step` in a human readable format."""
    x_locations = np.nonzero(info_state[:42])[0]
    o_locations = np.nonzero(info_state[42:84])[0]
    board = np.full(6 * 7, ".")
    board[x_locations] = "X"
    board[o_locations] = "0"
    board = np.reshape(board, (6, 7))
    return board

def normalise_rank(jjs) :
    min_rank = min(joueur.rank() for joueur in jjs)
    for joueur in jjs :
        joueur.set_rank(joueur.rank() - min_rank)
    max_rank = max(joueur.rank() for joueur in jjs)
    for joueur in jjs :
        joueur.set_rank(joueur.rank()/max_rank)
    return jjs

def rank_by_beta(jjs) :
    plt.scatter([joueur.beta for joueur in jjs], [joueur.rank() for joueur in jjs])

def rank(jjs) :
    plt.plot([joueur.rank() for joueur in jjs])

def beta(jjs) :
    plt.scatter([i for i in range(len(jjs))], [joueur.beta for joueur in jjs])

def beta_by_rank(jjs) :
    plt.scatter([joueur.rank() for joueur in jjs], [joueur.beta for joueur in jjs])

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

if __name__ == "__main__" :
    with open(parameters.jjs_path, 'rb') as f :
        jjs = pickle.load(f)

    #jjs.sort(key=lambda x: x.rank())

    jjs = normalise_rank(jjs)

    #calcul le rank moyen sur une fenetre (on exclue les nouvelles valeurs de rank normalisées)
    '''
    for joueur in jjs :
        joueur.rank_moyen = sum(joueur.rank_history[-10:-2])/8

    #retrie les jjs en fonction du rank moyen
    #jjs.sort(key=lambda x: x.rank_moyen)

    plt.scatter(list(range(len(jjs))), [joueur.rank_moyen for joueur in jjs])
    '''



    #Quelques experiences mais pas necessairement utiles
    #print("max:", max(joueur.rank() for joueur in jjs),
    #    "\nmin:", min(joueur.rank() for joueur in jjs))

    #plt.scatter(list(range(len(jjs))), [joueur.beta for joueur in jjs])
    #plt.plot([joueur.rank() for joueur in jjs], [len(joueur.rank_history) for joueur in jjs])


    #plt.scatter([joueur.beta for joueur in jjs], [sum(joueur.smurf_values)/len(joueur.smurf_values) for joueur in jjs])
    #plt.scatter([sum(joueur.smurf_values)/len(joueur.smurf_values) for joueur in jjs], [joueur.rank() for joueur in jjs])
    #plt.scatter([sum(joueur.smurf_values)/len(joueur.smurf_values) for joueur in jjs], [joueur.beta() for joueur in jjs])


    #les fonctions de plot de bases
    #rank(jjs)
    #beta(jjs)
    #rank_by_beta(jjs)
    #beta_by_rank(jjs)



    #permet de visualiser une partie (à l'envers pour le puissance 4)
    '''
    joueur = jjs[-1]
    print(joueur.rank())
    for coup in joueur.games[-1] :
        print(pretty_board(coup[0]))
        print(coup[1:])
    '''

    #permet de tracer les paramettres de la loi normale et la plot
    x, y = [i for i in range(len(jjs))], [joueur.beta for joueur in jjs]
    deg1, deg0 = np.polyfit(x, y, 1)
    line = [i*deg1 + deg0 for i in range(len(jjs))]
    #plt.plot(line)

    canard = [y[i]-line[i] for i in range(len(jjs))]
    #plt.scatter(x, canard)

    print(np.std(canard))
    #plt.hist(canard, bins=20)

    hist, bin_edges = np.histogram(canard, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

    p0 = [1., 0., 1.]

    coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

    # Get the fitted curve
    hist_fit = gauss(bin_centres, *coeff)

    plt.plot(bin_centres, hist, label='Test data')
    plt.plot(bin_centres, hist_fit, label='Fitted data')

    print('mean: ', coeff[1])
    print('std: ', coeff[2])



    #nb_of_jjs = 20
    #plt.plot([np.std(canard[i:i+nb_of_jjs]) for i in range(int(len(canard)/nb_of_jjs)-1)])


    #for i in range(8) :
    #    plt.plot(jjs[i].rank_history)

    '''
    prob_histo = [0]*100
    total_count = 0
    for joueur in tqdm.tqdm(jjs) :
        for game in joueur.games :
            for coup in game :
                for prob in coup[1] :
                    prob_histo[int(prob*100)-1] += 1
                    total_count += 1

    print(total_count)
    plt.scatter(np.linspace(0, 1, num=100, endpoint=True), prob_histo)
    '''
    plt.show()
