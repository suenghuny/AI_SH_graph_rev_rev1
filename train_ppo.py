from Components.Modeler_Component import *
from Components.Adapter_Component import *
from Components.Policy import *
from collections import deque
from cfg import get_cfg
from GPO import Agent
import numpy as np

from scipy.stats import randint


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


def train(agent, env, e, t):
    temp = random.uniform(0, 50)
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0
    step = 0
    eval = False

    if random.uniform(0, 1) > 0.5:
        interval_min = True

    else:
        interval_min = False

    interval_constant = random.uniform(0, 5)
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    step_checker = 0
    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            # print("다다다", env.now)

            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(interval_min=True,
                                                                                                 interval_constant=0.5,
                                                                                                 side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(interval_min,
                                                                                                       interval_constant,
                                                                                                       side='yellow')

            ship_feature = env.get_ship_feature()
            edge_index = env.get_edge_index()
            missile_node_feature = env.get_missile_node_feature()
            action_feature = env.get_action_feature()

            n_node_feature_missile = env.friendlies_fixed_list[0].air_tracking_limit + \
                                     env.friendlies_fixed_list[0].air_engagement_limit + \
                                     env.friendlies_fixed_list[0].num_m_sam + 1
            agent.eval_check(eval=True)
            node_representation = agent.get_node_representation(missile_node_feature, ship_feature, edge_index,n_node_feature_missile, mini_batch=False)  # 차원 : n_agents X n_representation_comm

            action_blue, prob, mask, a_index= agent.sample_action(node_representation, avail_action_blue, action_feature)
            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)

            reward, win_tag, done = env.step(action_blue, action_yellow)


            transition = (ship_feature, \
            edge_index, \
            missile_node_feature, \
            action_feature, \
            action_blue, \
            reward, \
            prob,\
            mask, \
            done, \
            avail_action_blue,
            a_index)


            agent.put_data(transition)
            # print(reward)
            episode_reward += reward

            status = None
            step_checker += 1





        else:
            pass_transition = True
            env.step(action_blue=[0, 0, 0, 0, 0, 0, 0, 0], action_yellow=enemy_action_for_transition,pass_transition=pass_transition)

    agent.eval_check(eval=False)
    agent.learn()
    return episode_reward, t


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

        output_dir = "../output_susceptibility/"
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
    ciws_threshold = 1
    polar_chart_visualize = False
    scenarios = ['scenario1', 'scenario2', 'scenario3']
    lose_ratio = list()
    remains_ratio = list()
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용

    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    # scenario = np.random.choice(scenarios)
    episode_polar_chart = polar_chart[0]
    records = list()
    import torch, random

    seed = 1230
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

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

    n_node_feature_missile = env.friendlies_fixed_list[0].air_tracking_limit + env.friendlies_fixed_list[0].air_engagement_limit +env.friendlies_fixed_list[0].num_m_sam +1
    agent = Agent(action_size = env.get_env_info()["action_feature_shape"],
                  feature_size_ship=env.get_env_info()["ship_feature_shape"],
                  feature_size_missile=env.get_env_info()["missile_feature_shape"],
                  feature_size_action=env.get_env_info()["action_feature_shape"],
                  n_node_feature_missile = n_node_feature_missile)



    reward_list = list()
    for e in range(num_iteration):

        start = time.time()
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step
                      )
        episode_reward, t = train(agent, env, e, t)
        if e >= cfg.train_start:
            if vessl_on == False:
                writer.add_scalar("episode", episode_reward, e - cfg.train_start)
        else:
            if vessl_on == False:
                writer.add_scalar("data aggregation", episode_reward, e)

        reward_list.append(episode_reward)
        if vessl_on == True:
            vessl.log(step=e, payload={'reward': episode_reward})

        if e % 10 == 0:
            import os
            import pandas as pd

            df = pd.DataFrame(reward_list)
            df.to_csv(output_dir + 'episode_reward.csv')

        # if e % 200 == 0:
        #     agent.save_model(e, t, epsilon, output_dir + "{}.pt".format(e))

        # print(len(agent.buffer.buffer[2]))
        print(
            "Total reward in episode {} = {}, time_step : {}, episode_duration : {}".format(
                e,
                np.round(episode_reward, 3),
                t, np.round(time.time() - start, 3)))

        # del data
        # del env