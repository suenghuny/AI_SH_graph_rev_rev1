from Components.Modeler_Component import *
from Components.Adapter_Component import *
from Components.Policy import *
from collections import deque
from cfg import get_cfg
from GDN import Agent
import numpy as np
import torch, random
import scipy



import pygad

def fitness_func(ga_instance, solution, solution_idx):
    data = preprocessing(scenarios)
    t = 0
    env = modeler(data,
                  visualize=visualize,
                  size=size,
                  detection_by_height=detection_by_height,
                  tick=tick,
                  simtime_per_framerate=simtime_per_frame,
                  ciws_threshold=ciws_threshold,
                  action_history_step=cfg.action_history_step)
    anneal_episode = cfg.anneal_episode
    anneal_step = (cfg.per_beta - 1) / anneal_episode
    epsilon = 1
    min_epsilon = 0.01
    reward_list = list()
    agent = None
    non_lose = 0
    action_availability_distribution = [[0] * env.get_env_info()["n_actions"] for _ in range(500)]
    score = 0
    n = 10
    for e in range(n):
        seed = 2 * e
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        start = time.time()
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step)
        epi_reward, epsilon, t, eval, win_tag, action_availability_distribution = evaluation(agent, env, e, t, train_start=cfg.train_start, epsilon=epsilon,
                                                 min_epsilon=min_epsilon, anneal_step=anneal_step, initializer=False,
                                                 output_dir=None, vdn=True, n_step=n_step, action_availability_distribution=action_availability_distribution,
                                                                                             solution = solution)


        # import pandas as pd
        #
        # df = pd.DataFrame(arr)
        # df.to_csv("distribution.csv")
        # print(score)
        if win_tag != 'lose':
            score += 1/n
        else:
            score += 0
    print(score)
    return score

def constraints_func(solution, solution_idx, action_size):
    # Verify the constraints for each element of the solution encoding
    for element in solution:
        if (element < 0) or (element >= action_size) or (type(element) is not int):
            return False
    return True

def create_population():
    initial_population = np.random.randint(low=0, high=env.get_env_info()["n_actions"], size=(population_size, T))
    return initial_population


def action_changer(action, avail_actions):
    d = False
    #print("전", action, avail_actions[0])
    while d == False:
        if avail_actions[0][action] == True:
            d = True
            action = action
        else:
            action -= 1
    #print("후", action, avail_actions[0])
    action = [action]
    return action


def preprocessing(scenarios):
    scenario = scenarios[0]
    if mode == 'txt':
        if vessl_on == True:
            input_path = ["/root/AI_SH/Data/{}/ship.txt".format(scenario),
                          "/root/AI_SH/Data/{}/patrol_aircraft.txt".format(scenario),
                          "/root/AI_SH/Data/{}/SAM.txt".format(scenario),
                          "/root/AI_SH/Data/{}/SSM.txt".format(scenario),
                          "/root/AI_SH/Data/{}/inception.txt".format(scenario)]
        else:
            input_path = ["Data/{}/ship.txt".format(scenario),
                          "Data/{}/patrol_aircraft.txt".format(scenario),
                          "Data/{}/SAM.txt".format(scenario),
                          "Data/{}/SSM.txt".format(scenario),
                          "Data/{}/inception.txt".format(scenario)]
    else:
        input_path = "Data\input_data.xlsx"

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data


def evaluation(agent, env, e, t, train_start, epsilon, min_epsilon, anneal_step, initializer, output_dir, vdn, n_step, action_availability_distribution, solution = None):
    interval_min_blue = cfg.interval_min_blue
    interval_constant_blue = cfg.interval_constant_blue
    temp = random.uniform(0, 50)
    agent_blue = Policy(env, rule='rule2', temperatures=[cfg.temperature, cfg.temperature])
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0
    step = 0
    losses = []
    epi_r = list()
    eval = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    step_checker = 0
    if random.uniform(0, 1) > 0.5:
        interval_min = True
    else:
        interval_min = False
    interval_constant = random.uniform(2,4)

    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(interval_min_blue,
                                                                                                 interval_constant_blue,
                                                                                                 side='blue')

            t = int(env.now / 4)
            action = solution[t]
            action_blue = action_changer(action, avail_action_blue)

            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(interval_min, interval_constant, side='yellow')

            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done, leaker = env.step(action_blue, action_yellow, rl = False)
            episode_reward += reward
            status = None
            step_checker += 1
            if e >= train_start:
                t += 1
        else:
            pass_transition = True
            env.step(action_blue=friendly_action_for_transition,
                     action_yellow=enemy_action_for_transition, pass_transition=pass_transition, rl = False)

        if done == True:
            break
    return episode_reward, epsilon, t, eval, win_tag, action_availability_distribution


if __name__ == "__main__":
    cfg = get_cfg()
    vessl_on = cfg.vessl
    if vessl_on == True:
        import vessl

        vessl.init()
        output_dir = "/output/"
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        from torch.utils.tensorboard import SummaryWriter

        output_dir = "../output_susceptibility_heuristic/"
        writer = SummaryWriter('./logs2')
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    import time

    """
    환경 시스템 관련 변수들
    """
    visualize = False  # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
    size = [600, 600]  # 화면 size / 600, 600 pixel
    tick = 500  # 가시화 기능 사용 시 빠르기
    n_step = cfg.n_step
    simtime_per_frame = cfg.simtime_per_frame
    decision_timestep = cfg.decision_timestep
    detection_by_height = False  # 고도에 의한
    num_iteration = cfg.num_episode  # 시뮬레이션 반복횟수
    mode = 'txt'  # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'  # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
    temperature = [10,
                   20]  # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
    ciws_threshold = 0.5
    polar_chart_visualize = False
    scenarios = ['scenario1', 'scenario2', 'scenario3']
    lose_ratio = list()
    remains_ratio = list()
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용
    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    episode_polar_chart = polar_chart[0]
    records = list()

    population_size = 50
    num_generations = 100
    T = 500  # Define the length of the solution encoding







    data = preprocessing(scenarios)
    env = modeler(data,
                  visualize=visualize,
                  size=size,
                  detection_by_height=detection_by_height,
                  tick=tick,
                  simtime_per_framerate=simtime_per_frame,
                  ciws_threshold=ciws_threshold,
                  action_history_step=cfg.action_history_step)
    df = pd.read_csv("distribution.csv")

    #print([[i for i in range(env.get_env_info()["n_actions"]) if df.iloc[j, i] > 0] for j in range(T)])
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=1,
                           fitness_func=fitness_func,
                           sol_per_pop=population_size,
                           num_genes=T,
                           gene_type=int,
                           gene_space=[[i for i in range(env.get_env_info()["n_actions"]) if df.iloc[j, i] > 0] for j in range(T)])
    # initial_population = create_population()
    # ga_instance.population_init(population=initial_population)


    ga_instance.run()
    best_solution = ga_instance.best_solution()
    best_fitness = ga_instance.best_solution()[1]
