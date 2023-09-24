import numpy as np
import pandas as pd
import torch, random
from cfg import get_cfg


from Components.Modeler_Component_test import *
from Components.Adapter_Component import *
from Components.Policy import *
from collections import deque
from GDN import Agent
from scipy.optimize import minimize, dual_annealing
from scipy.optimize import Bounds
from functools import partial
fix_l = 0
fix_u = 17

 # 현재 까지 5가 최고임

# 5에서 0.9
# 15에서 0.74




def objective_function(ga_instance, solution, solution_idx):
    n_eval = 1
    episode_reward = 0
    temperature1 = solution[0]
    interval_constant_blue1 = solution[1]
    temperature2 = solution[2]
    interval_constant_blue2 = solution[3]
    air_alert_distance = solution[4]
    warning_distance = solution[5]
    for _ in range(n_eval):
        env_copy = deepcopy(env)
        env_copy.interval_constant_blue = [interval_constant_blue1, interval_constant_blue2]
        env_copy.air_alert_distance_blue = air_alert_distance


        temp = random.uniform(fix_l, fix_u)
        agent_yellow = Policy(env_copy, rule='rule2', temperatures=[temp, temp])
        agent_blue = Policy(env_copy, rule='rule3', temperatures=[temperature1, temperature2 ])

        enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
        for t in range(prediction_horizon):
            if env_copy.now % (decision_timestep) <= 0.00001:
                avail_action_blue, target_distance_blue, air_alert_blue = env_copy.get_avail_actions_temp(side='blue')
                avail_action_yellow, target_distance_yellow, air_alert_yellow = env_copy.get_avail_actions_temp(side='yellow')

                action_blue = agent_blue.get_action(avail_action_blue, target_distance_blue, air_alert_blue, open_fire_distance = warning_distance)
                action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)

                reward, win_tag, done, leakers = env_copy.step(action_blue, action_yellow, rl=False)
                episode_reward += reward
                if (done == True):
                    if ((win_tag == 'draw') or (win_tag == 'win')):
                        episode_reward += 1000
                    else:
                        episode_reward -= 1000
                    break

            else:
                pass_transition = True
                actions_blue = list()
                for i in range(len(env.friendlies_fixed_list)):
                    actions_blue.append([0, 0, 0, 0, 0, 0, 0, 0])
                env_copy.step(action_blue=actions_blue, action_yellow=enemy_action_for_transition,pass_transition=pass_transition)
    #print(-episode_reward)
    return episode_reward  # Minimize the negative total reward (maximize reward)


def SA_optimizer(env):
    x0 = [15,15,15,15,100,200]
    result=minimize(objective_function,
                    x0=x0,
                    method='Nelder-Mead',
                    bounds=[[0,50],[0,50],[0,50],[0,50],[0,200],[0,300]],args=(env, ),
                    options={'xtol': 2, 'disp': True}
                    )
    #print("?????slslslsdjfkdlsdkfj")
    optimal_control_sequence = result.x

    return optimal_control_sequence

def GA_optimizer(env):
    solution_space = [[i for i in range(0, 20)], [i for i in range(0, 50)],
                      [i for i in range(0, 20)], [i for i in range(0, 50)], [i for i in range(0, 200)],
                      [i for i in range(0, 300)]]
    num_genes = len(solution_space)

    initial_population = []
    sol_per_pop = 10
    for _ in range(sol_per_pop):
        new_solution = [np.random.choice(space) for space in solution_space]
        initial_population.append(new_solution)

    num_generations = 10  # 세대 수
    num_parents_mating = 6  # 각 세대에서 선택할 부모 수
    parent_selection_type = "sss"
    keep_parents = 2
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10
    #objective_func_partial = partial(objective_function,  env=env)
    import pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=objective_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           initial_population=initial_population,
                           gene_space=solution_space,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           )
    ga_instance.run()
    best_solutions = ga_instance.best_solution()[0]
    return best_solutions


def preprocessing(scenarios):
    scenario = scenarios
    if mode == 'txt':
        input_path = ["Data/Test/dataset{}/ship.txt".format(scenario),
                      "Data/Test/dataset{}/patrol_aircraft.txt".format(scenario),
                      "Data/Test/dataset{}/SAM.txt".format(scenario),
                      "Data/Test/dataset{}/SSM.txt".format(scenario),
                      "Data/Test/dataset{}/inception.txt".format(scenario)]
    else:
        input_path = "Data/Test/dataset{}/input_data.xlsx".format(scenario)

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data


def evaluation(env):
    temp = random.uniform(fix_l, fix_u)
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0

    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    overtime = None

    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')

            optimal_control_sequence = GA_optimizer(env)

            temperature1 = optimal_control_sequence[0]
            interval_constant_blue1 = optimal_control_sequence[1]
            temperature2 = optimal_control_sequence[2]
            interval_constant_blue2 = optimal_control_sequence[3]
            air_alert_distance = optimal_control_sequence[4]
            warning_distance = optimal_control_sequence[5]

            agent_blue = Policy(env, rule='rule3', temperatures=[temperature1, temperature2])
            env.interval_constant_blue = [interval_constant_blue1, interval_constant_blue2]
            env.air_alert_distance_blue = air_alert_distance


            action_blue = agent_blue.get_action(avail_action_blue, target_distance_blue, air_alert_blue, open_fire_distance = warning_distance)
            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done, leakers = env.step(action_blue, action_yellow, rl=False)
            episode_reward += reward

        else:
            pass_transition = True
            actions_blue = list()
            for i in range(len(env.friendlies_fixed_list)):
                actions_blue.append([0, 0, 0, 0, 0, 0, 0, 0])
            env.step(action_blue=actions_blue, action_yellow=enemy_action_for_transition,pass_transition=pass_transition)

    return episode_reward, win_tag, leakers, overtime


if __name__ == "__main__":

    vessl_on = cfg.vessl
    if vessl_on == True:
        import vessl

        vessl.init()
        output_dir = "/output/"
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        print("시작")
        from torch.utils.tensorboard import SummaryWriter

        output_dir = "../output_susceptibility/"
        writer = SummaryWriter('./logs2')
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    import time

    """

    환경 시스템 관련 변수들

    """
    visualize =False # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
    size = [600, 600]  # 화면 size / 600, 600 pixel
    tick = 500  # 가시화 기능 사용 시 빠르기
    n_step = cfg.n_step
    simtime_per_frame = cfg.simtime_per_frame
    decision_timestep = cfg.decision_timestep
    detection_by_height = False  # 고도에 의한
    num_iteration = cfg.num_episode  # 시뮬레이션 반복횟수
    mode = 'excel'  # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'  # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)

    ciws_threshold = 1
    polar_chart_visualize = False
    #3scenarios = ['scenario1', 'scenario2', 'scenario3']
    lose_ratio = list()
    remains_ratio = list()
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용
    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    episode_polar_chart = polar_chart[0]
    records = list()

    datasets = [i for i in range(1, 29)]
    non_lose_ratio_list = []
    for dataset in datasets:
        data = preprocessing(dataset)
        t = 0


        eval_lose_ratio = list()
        eval_win_ratio = list()
        lose_ratio = list()
        win_ratio = list()
        reward_list = list()

        eval_lose_ratio1 = list()
        eval_win_ratio1 = list()
        print("noise", cfg.with_noise)
        non_lose_rate = list()

        n = cfg.n_test
        non_lose_rate = list()

        prediction_horizon = 10
        seed = cfg.seed  # 원래 SEED 1234
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        for j in range(n):



            env = modeler(data,
                          visualize=visualize,
                          size=size,
                          detection_by_height=detection_by_height,
                          tick=tick,
                          simtime_per_framerate=simtime_per_frame,
                          ciws_threshold=ciws_threshold,
                          action_history_step=cfg.action_history_step,
                          interval_constant_blue=[cfg.interval_constant_blue, cfg.interval_constant_blue]
                          )
            episode_reward, win_tag, leakers, overtime = evaluation(env)
            if win_tag == 'draw' or win_tag == 'win':
                non_lose_rate.append(1)
            else:
                non_lose_rate.append(0)
            print('전', win_tag, episode_reward, env.now, overtime, np.sum(non_lose_rate)/(j+1))
        non_lose_ratio_list.append(np.mean(non_lose_rate))
        df = pd.DataFrame(non_lose_ratio_list)
        df.to_csv("mpc_result.csv")