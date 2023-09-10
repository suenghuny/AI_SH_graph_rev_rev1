from Components.Modeler_Component import *
from Components.Adapter_Component import *
from Components.Policy import *
from cfg import get_cfg
import numpy as np
import gc


import pygad

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change")
    last_fitness = ga_instance.best_solution()[1]



def simulation(solution):
    temperature1 = solution[0]
    interval_constant_blue1 = solution[1]
    temperature2 = solution[2]
    interval_constant_blue2 = solution[3]
    air_alert_distance = solution[4]
    #data = preprocessing(scenarios)
    t = 0
    env = modeler(data,
                  visualize=visualize,
                  size=size,
                  detection_by_height=detection_by_height,

                  tick=tick,
                  simtime_per_framerate=simtime_per_frame,
                  ciws_threshold=ciws_threshold,
                  action_history_step=cfg.action_history_step,
                  interval_constant_blue=[interval_constant_blue1, interval_constant_blue2],
                  air_alert_distance = air_alert_distance
                  )
    anneal_episode = cfg.anneal_episode
    anneal_step = (cfg.per_beta - 1) / anneal_episode
    epsilon = 1
    min_epsilon = 0.01
    reward_list = list()
    agent = None
    non_lose = 0
    score = 0
    n = 100
    seed = 4
    np.random.seed(seed)
    random.seed(seed)
    for e in range(n):
        start = time.time()
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step,
                      interval_constant_blue = [interval_constant_blue1, interval_constant_blue2]
                      )
        epi_reward, epsilon, t, eval, win_tag= evaluation(agent, env, e, t, train_start=cfg.train_start, epsilon=epsilon,
                                                 min_epsilon=min_epsilon, anneal_step=anneal_step, initializer=False,
                                                 output_dir=None, vdn=True, n_step=n_step, action_availability_distribution=None,
                                                                                             temperature1=temperature1,
                                                                                             temperature2 = temperature2,
                                                                                             )


        if win_tag != 'lose':
            score += 1/n
        else:
            score += 0

        del env
        gc.collect()
    print(score, solution)
    return score

def fitness_func(solution):
    score = simulation(solution)
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
    scenario = scenarios
    if mode == 'txt':
        if vessl_on == True:
            input_path = ["/root/AI_SH_graph_rev/Data/{}/ship.txt".format(scenario),
                          "/root/AI_SH_graph_rev/Data/{}/patrol_aircraft.txt".format(scenario),
                          "/root/AI_SH_graph_rev/Data/{}/SAM.txt".format(scenario),
                          "/root/AI_SH_graph_rev/Data/{}/SSM.txt".format(scenario),
                          "/root/AI_SH_graph_rev/Data/{}/inception.txt".format(scenario)]
        else:
#            print(scenario)
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
mode ='txt'
vessl_on = False
polar_chart_visualize = False
polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용
polar_chart = [polar_chart_scenario1]
df_dict = {}
episode_polar_chart = polar_chart[0]
data = preprocessing('scenario1')

def on_start(ga_instance):
    print("on_start()")

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):
    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")

def evaluation(agent, env, e, t, train_start, epsilon, min_epsilon, anneal_step, initializer, output_dir, vdn, n_step, action_availability_distribution,
               temperature1,
               temperature2,
               ):
    temp = random.uniform(0, 50)
    agent_blue = Policy(env, rule='rule2', temperatures=[temperature1, temperature2])
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


    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')

            action_blue = agent_blue.get_action(avail_action_blue, target_distance_blue, air_alert_blue)
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
    return episode_reward, epsilon, t, eval, win_tag


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

    population_size = 10
    num_generations = 10
    mutation_range = 5

    solution_space = [[i for i in range(0, 50)], [i  for i in range(0, 50)],
    [i  for i in range(0, 50)], [i  for i in range(0, 50)], [i for i in range(0, 60)]]
    n_pool = population_size
    current_solution_pool=list()
    for n in range(n_pool):
        solutions = list()
        for k in range(len(solution_space)):
            s = np.random.choice(solution_space[k])
            solutions.append(s)
        current_solution_pool.append(solutions)

#    current_solution_pool =[[40.8, 32.1, 14.0, 19.9, 14.9], [34.1, 12.6, 11.4, 21.3, 45.0], [14.4, 30.6, 5.0, 23.2, 48.0], [0.8, 23.2, 1.7, 9.2, 15.3], [40.8, 32.1, 14.0, 21.3, 45.0], [34.1, 12.6, 11.4, 19.9, 14.9], [44.8, 34.8, 19.7, 22.3, 15.9]]

    cross_over_position = 3
    num_mutation = 10
    current_solution_pool = [[2, 47, 10, 22, 4], [2, 47, 10, 22, 1], [49, 46, 11, 21, 4], [5, 43, 11, 26, 1], [2, 47, 10, 22, 1], [2, 47, 10, 22, 4], [4, 47, 5, 23, 4], [47, 43, 5, 20, 7], [1, 48, 11, 19, 0], [5, 43, 11, 26, 1], [3, 43, 5, 25, 1], [49, 46, 11, 21, 4], [47, 49, 6, 20, 4], [1, 45, 14, 20, 3], [2, 48, 7, 23, 0], [48, 42, 14, 22, 59]]

    # current_solution_pool = [[40.8, 32.1, 14.0, 19.9, 14.9], [40.8, 33.2, 14.3, 21.3, 15.0], [34.1, 12.6, 11.4, 21.3, 45.0], [14.4, 30.6, 5.0, 23.2, 48.0]]
    # new_parents = sorted(current_solution_pool, key=fitness_func, reverse=True)[:1]
    # for t in range(0, num_mutation):
    #     mutation = deepcopy(new_parents[0])
    #     for l in range(len(mutation)):
    #         idx = solution_space[l].index(mutation[l])
    #         nex_idx = idx + np.random.randint(-mutation_range, mutation_range)
    #         try:
    #             mutation[l] = solution_space[l][nex_idx]
    #         except IndexError:
    #             pass
    #     current_solution_pool.append(mutation)

    for i in range(0, 10):

        print(episode_polar_chart)
        new_parents = sorted(current_solution_pool, key=fitness_func, reverse=True)[:4]
        print(f"optimal fitness in {i:0>2d} generation: {fitness_func(new_parents[0])}")
        # print("전", new_parents)
        crossovers = [
            new_parents[0][:cross_over_position] + new_parents[1][cross_over_position:],
            new_parents[1][:cross_over_position] + new_parents[0][cross_over_position:],
        ]
        current_solution_pool = new_parents + crossovers
        for t in range(0, num_mutation):
            mutation = deepcopy(new_parents[0])
            for l in range(len(mutation)):
                idx = solution_space[l].index(mutation[l])
                nex_idx = idx + np.random.randint(-mutation_range,mutation_range)
                try:
                    mutation[l] = solution_space[l][nex_idx]
                except IndexError:pass

            current_solution_pool.append(mutation)
        print(current_solution_pool)





