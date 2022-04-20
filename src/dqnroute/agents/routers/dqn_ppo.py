import random
import math
import logging
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import pandas as pd
import pprint
import os

from typing import List, Tuple, Dict, Union
from ..base import *
from .link_state import *
from ...constants import DQNROUTE_LOGGER
from ...messages import *
from ...memory import *
from ...utils import *
from ...networks import *

logger = logging.getLogger(DQNROUTE_LOGGER)


class InstantMessagesSimulationFix:

    routers = defaultdict(dict)

    @staticmethod
    def addToSimulation(router):
        InstantMessagesSimulationFix.routers[router.id] = router

    @staticmethod
    def sendMsg(sender: AgentId, to: AgentId, msg: Message):
        InstantMessagesSimulationFix.routers[to].handleMsgFrom(sender, msg)

    @staticmethod
    def sendMsgAndReturn(sender: AgentId, to: AgentId, msg: Message):
        return InstantMessagesSimulationFix.routers[to].handleMsgFrom(sender, msg)

# нужна, когда юзаем single network wtf
class SharedBrainStorage:
    INSTANCE = None
    PROCESSED_NODES = 0

    @staticmethod
    def load(brain_loader: Callable[[], QNetwork], no_nodes: int) -> QNetwork:
        if SharedBrainStorage.INSTANCE is None:
            SharedBrainStorage.INSTANCE = brain_loader()
        SharedBrainStorage.PROCESSED_NODES += 1
        # print(f"Brain initialization: {SharedBrainStorage.PROCESSED_NODES} / {no_nodes} agents")
        result = SharedBrainStorage.INSTANCE
        if SharedBrainStorage.PROCESSED_NODES == no_nodes:
            # all nodes have been processes
            # prepare this class for possible reuse
            SharedBrainStorage.INSTANCE = None
            SharedBrainStorage.PROCESSED_NODES = 0
        return result


class DQNPPORouter(LinkStateRouter, RewardAgent):
    """
    A router which implements the DQN-routing algorithm.
    """

    def __init__(self, batch_size: int, mem_capacity: int, nodes: List[AgentId],
                 optimizer='rmsprop', brain=None, random_init=False, max_act_time=None,
                 additional_inputs=[], softmax_temperature: float = 1.5,
                 probability_smoothing: float = 0.0, load_filename: str = None,
                 use_single_neural_network: bool = False,
                 use_reinforce: bool = True,
                 use_combined_model: bool = False,
                 count = 1, dqn_emb=False,
                 **kwargs):
        """
        Parameters added by Igor:
        :param softmax_temperature: larger temperature means larger entropy of routing decisions.
        :param probability_smoothing (from 0.0 to 1.0): if greater than 0, then routing probabilities will
            be separated from zero.
        :param load_filename: filename to load the neural network. If None, a new network will be created.
        :param use_single_neural_network: all routers will reference the same instance of the neural network.
            In particular, this very network will be influeced by training steps in all nodes.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.memory = Memory(mem_capacity)
        self.additional_inputs = additional_inputs
        self.nodes = nodes
        self.max_act_time = max_act_time

        # changed by Igor: custom temperatures for softmax:
        self.min_temp = softmax_temperature
        # added by Igor: probability smoothing (0 means no smoothing):
        self.probability_smoothing = probability_smoothing

        self.use_reinforce = use_reinforce
        self.use_combined_model = use_combined_model

        self.dqn_emb = dqn_emb
        if dqn_emb:
            self.count = 1
        else:
            self.count = count
        self.cur_batch_size = 0
        self.max_batch_size = 60


        # changed by Igor: brain loading process
        def load_brain():
            b = brain
            if b is None:
                b = self._makeBrain(additional_inputs=additional_inputs, **kwargs)
                if random_init:
                    b.init_xavier()
                else:
                    if load_filename is not None:
                        b.change_label(load_filename)
                    b.restore()
            return b

        if use_single_neural_network:
            self.brain = SharedBrainStorage.load(load_brain, len(nodes))
        else:
            self.brain = load_brain()
        self.use_single_neural_network = use_single_neural_network

        self.optimizer = get_optimizer(optimizer)(self.brain.parameters())
        self.loss_func = nn.MSELoss()

        InstantMessagesSimulationFix.addToSimulation(self)
        self.bag_info_storage = defaultdict(dict)
        # for debag usage
        self.bags_passed = defaultdict(dict)

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if self.max_act_time is not None and self.env.time() > self.max_act_time:
            return super().route(sender, pkg, allowed_nbrs)
        else:
            to, estimate, saved_state = self._act(pkg, allowed_nbrs)
            rewardMsg = self.registerResentPkgPPO(pkg, estimate, to, saved_state)
            already_passed = False
            if self.bags_passed.get(pkg.id) is not None:
                already_passed = True
                # print("AlreadyPassed")
            self.bags_passed[pkg.id] = pkg
            # return to, [OutMessage(self.id, sender, reward), OutMessage(self.id, to, "239")] if sender[0] != 'world' else [] #wtf
            if sender[0] == 'world':
                bag_info = Bag_info(pkg.id, saved_state, self.count)
                bag_info.setQvalue(estimate)
                self._addBagInfoToStorage(bag_info)
            else:
                if self.isThereBagInfo(pkg.id):
                    bag_info = self._getBagInfo(pkg.id)
                    reward = InstantMessagesSimulationFix.sendMsgAndReturn(self.id, sender, rewardMsg)
                    info = self._makeInfo(sender, saved_state, to, reward)
                    # bag_info.updateLast(info)
                    bag_info.append(info)
                    bag_info.setQvalue(estimate)
                    bag_info.setState(saved_state)
                    self._addBagInfoToStorage(bag_info)
                else:
                    InstantMessagesSimulationFix.sendMsg(self.id, sender, GetBagInfoMsg(self.id, pkg.id))
                    bag_info = self._getBagInfo(pkg.id)
                    # reward = InstantMessagesSimulationFix.sendMsgAndReturn(self.id, sender, rewardMsg)
                    # info = self._makeInfo(sender, saved_state, to, reward)
                    # bag_info.append(info)
                    bag_info.setQvalue(estimate)
                    bag_info.setState(saved_state)
                    self._addBagInfoToStorage(bag_info)
                bag_info = self._getBagInfo(pkg.id)
                if bag_info.isFullPath():
                    to_num = to[1]
                    if to_num >= 14 and to_num <= 17:
                        InstantMessagesSimulationFix.sendMsg(self.id, sender,
                                                             PathRewardMsg(sender, bag_info, self.count, True))
                    else:
                        InstantMessagesSimulationFix.sendMsg(self.id, sender, PathRewardMsg(sender, bag_info, self.count, False))

            return to, []
            # return to, [OutMessage(self.id, sender, rewardMsg)] if sender[0] != 'world' else [] #wtf

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[Message]:
        if isinstance(msg, RewardMsg):
            action, reward_new, prev_state = self.receiveRewardPPO(msg)
            return reward_new
        elif isinstance(msg, GetBagInfoMsg):
            bag_info = self._popBagInfo(msg.bag_id)
            InstantMessagesSimulationFix.sendMsg(self.id, msg.origin, UpdateTableMsg(self.id, bag_info))
        elif isinstance(msg, UpdateTableMsg):
            bag_info = msg.bag_info
            self._addBagInfoToStorage(bag_info)
        elif isinstance(msg, PathRewardMsg):
            msg.count = msg.count - 1
            bag_info = msg.bag_info

            if msg.count != 0:
                send_to = self._getPathBack(bag_info, msg.count)
                if msg.all_learn:
                    self._learn(bag_info, msg.count)
                InstantMessagesSimulationFix.sendMsg(self.id, send_to, msg)
            else:
                self._learn(bag_info, msg.count)
                # print("learn bag_id=" + str(bag_info.bag_id) + ", agent_id=" + str(self.id))
        else:
            return super().handleMsgFrom(sender, msg)

    def registerResentPkgPPO(self, pkg: Package, Q_estimate: float, action, data, **kwargs) -> RewardMsg:  # omg3
        rdata = self._getRewardData(pkg, data)
        self._pending_pkgs[pkg.id] = (action, rdata, data)

        # Igor Buzhinsky's hack to suppress a no-key exception in receiveReward
        self._last_tuple = action, rdata, data

        return self._mkRewardPPO(pkg, Q_estimate, rdata)

    def _mkRewardPPO(self, bag: Bag, Q_estimate: float, reward_data) -> ConveyorRewardMsg: # omg4
        time_processed, energy_gap = reward_data
        return ConveyorRewardMsg(self.id, bag, Q_estimate, time_processed, energy_gap)

    def receiveRewardPPO(self, msg: RewardMsg): # omg1
        try:
            action, old_reward_data, saved_data = self._pending_pkgs.pop(msg.pkg.id)
        except KeyError:
            self.log(f'not our package: {msg.pkg}, path:\n  {msg.pkg.node_path}\n', force=True)
            #raise
            # Igor Buzhinsky's hack to suppress a no-key exception in receiveReward
            action, old_reward_data, saved_data = self._last_tuple
        reward = self._computeRewardPPO(msg, old_reward_data)
        return action, reward, saved_data

    def _computeRewardPPO(self, msg: ConveyorRewardMsg, old_reward_data):  # omg2
        time_sent, _ = old_reward_data
        time_processed, energy_gap = msg.reward_data
        time_gap = time_processed - time_sent

        # self.log('time gap: {}, nrg gap: {}'.format(time_gap, energy_gap), True)
        # return time_gap
        return time_gap + self._e_weight * energy_gap

        # return time_gap + 0.*energy_gap

    def _sumRewards(self, bag_info, count):
        rewards = 0
        gamma = 0.95
        discount = 1
        l = self.count

        if self.dqn_emb:
            gamma = 1
        for i in range(l - count, l):
            info = bag_info.getPathRouter(i)
            rewards += discount*info[3]
            discount *= gamma
        return rewards, discount

    def _learn(self, bag_info, count):
        l = self.count
        rewards, discount = self._sumRewards(bag_info, l - count)
        Q_new = rewards + discount*bag_info.getQvalue()
        action = bag_info.getPathRouter(count)[2]
        prev_state = bag_info.getState()
        self.memory.add((prev_state, action[1], -Q_new))
        if self.dqn_emb:
            if self.use_reinforce:
                self._replay()
        else:
            self.cur_batch_size += 1
            if self.cur_batch_size >= self.max_batch_size:
                self.cur_batch_size = 0
                if self.use_reinforce:
                    self._replay()
        return []

    def _makeInfo(self, router, state, action_to, reward):
        return (router, state, action_to, reward)

    def _getBagInfo(self, bag_id):
        return self.bag_info_storage[bag_id]

    def _popBagInfoAndUpdate(self, bag_id, estimate, info):
        bag_info = self.bag_info_storage.pop(bag_id)
        bag_info.append(info)
        bag_info.setQvalue(estimate)
        return bag_info

    def _popBagInfo(self, bag_id):
        return self.bag_info_storage.pop(bag_id)

    def _getPathBack(self, bag_info, cnt):
        info = bag_info.getPathRouter(cnt - 1)
        return info[0]

    def _addBagInfoToStorage(self, bag_info: Bag_info):
        bag_id = bag_info.bag_id
        self.bag_info_storage[bag_id] = bag_info

    def isThereBagInfo(self, bag_id):
        if self.bag_info_storage.get(bag_id) is None:
            return False
        return True

    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs, one_out=False, **kwargs)

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        state = self._getNNState(pkg, allowed_nbrs)
        prediction = self._predict(state)[0]
        distr = softmax(prediction, self.min_temp)
        estimate = -np.dot(prediction, distr)

        to = -1
        while ('router', to) not in allowed_nbrs:
            to = sample_distr(distr)

        return ('router', to), estimate, state

    def _predict(self, x):
        self.brain.eval()
        return self.brain(*map(torch.from_numpy, x)).clone().detach().numpy() #wtf

    def _train(self, x, y): #wtf
        self.brain.train()
        self.optimizer.zero_grad()
        output = self.brain(*map(torch.from_numpy, x))
        loss = self.loss_func(output, torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def _getAddInput(self, tag, *args, **kwargs):
        if tag == 'amatrix':
            amatrix = nx.convert_matrix.to_numpy_array(
                self.network, nodelist=self.nodes, weight=self.edge_weight,
                dtype=np.float32)
            gstate = np.ravel(amatrix)
            return gstate
        else:
            raise Exception('Unknown additional input: ' + tag)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId]):
        n = len(self.nodes)
        addr = np.array(self.id[1])
        dst = np.array(pkg.dst[1])

        neighbours = np.array(
            list(map(lambda v: v in nbrs, self.nodes)),
            dtype=np.float32)
        input = [addr, dst, neighbours]

        for inp in self.additional_inputs:
            tag = inp['tag']
            add_inp = self._getAddInput(tag)
            if tag == 'amatrix':
                add_inp[add_inp > 0] = 1
            input.append(add_inp)

        return tuple(input)

    def _sampleMemStacked(self):
        """
        Samples a batch of episodes from memory and stacks
        states, actions and values from a batch together.
        """
        # i_batch = self.memory.sample(self.batch_size)
        i_batch = []
        if self.dqn_emb:
            i_batch = self.memory.sample(self.batch_size)
        else:
            i_batch = self.memory.getLastN(self.max_batch_size)
        batch = [b[1] for b in i_batch]

        states = stack_batch([l[0] for l in batch])
        actions = [l[1] for l in batch]
        values = [l[2] for l in batch]

        return states, actions, values

    def _replay(self):
        """
        Fetches a batch of samples from the memory and fits against them.
        """
        states, actions, values = self._sampleMemStacked()
        preds = self._predict(states)

        for i in range(self.batch_size):
            a = actions[i]
            preds[i][a] = values[i]

        self._train(states, preds)


class DQNPPORouterOO(DQNPPORouter):
    """
    Variant of DQN router which uses Q-network with scalar output.
    """

    # создает нейронную сеть, принимает d, n, y, G выход скаляр
    # какой скаляр?? wtf
    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs,
                        one_out=True, **kwargs)

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        state = self._getNNState(pkg, allowed_nbrs)
        prediction = self._predict(state).flatten()
        distr = softmax(prediction, self.min_temp)

        # Igor: probability smoothing
        distr = (1 - self.probability_smoothing) * distr + self.probability_smoothing / len(distr)

        to_idx = sample_distr(distr)
        estimate = -np.dot(prediction, distr)

        saved_state = [s[to_idx] for s in state]
        to = allowed_nbrs[to_idx]
        return to, estimate, saved_state

    def _nodeRepr(self, node):
        return np.array(node)

    def _getAddInput(self, tag, nbr):
        return super()._getAddInput(tag)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId]):
        n = len(self.nodes)
        addr = self._nodeRepr(self.id[1])
        dst = self._nodeRepr(pkg.dst[1])

        get_add_inputs = lambda nbr: [self._getAddInput(inp['tag'], nbr)
                                      for inp in self.additional_inputs]

        input = [[addr, dst, self._nodeRepr(v[1])] + get_add_inputs(v) for v in nbrs]
        return stack_batch(input)

    def _replay(self):
        states, _, values = self._sampleMemStacked()
        self._train(states, np.expand_dims(np.array(values, dtype=np.float32), axis=0))


class DQNPPORouterEmb(DQNPPORouterOO):
    """
    Variant of DQNPPORouter which uses graph embeddings instead of
    one-hot label encodings.
    """

    def __init__(self, embedding: Union[dict, Embedding], edges_num: int, **kwargs):
        # Those are used to only re-learn the embedding when the topology is changed
        self.prev_num_nodes = 0
        self.prev_num_edges = 0
        self.init_edges_num = edges_num
        self.network_initialized = False

        if type(embedding) == dict:
            self.embedding = get_embedding(**embedding)
        else:
            self.embedding = embedding

        super().__init__(**kwargs)

    def _makeBrain(self, additional_inputs=[], **kwargs):
        if not self.use_combined_model:
            return QNetwork(
                len(self.nodes), additional_inputs=additional_inputs,
                embedding_dim=self.embedding.dim, one_out=True, **kwargs
            )
        else:
            return CombinedNetwork(
                len(self.nodes), additional_inputs=additional_inputs,
                embedding_dim=self.embedding.dim, one_out=True, **kwargs
            )

    def _nodeRepr(self, node):
        return self.embedding.transform(node).astype(np.float32)

    def networkStateChanged(self): # wtf
        num_nodes = len(self.network.nodes)
        num_edges = len(self.network.edges)

        if not self.network_initialized and num_nodes == len(self.nodes) and num_edges == self.init_edges_num: #wtf
            self.network_initialized = True

        if self.network_initialized and (num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes): #wtf
            self.prev_num_nodes = num_nodes
            self.prev_num_edges = num_edges
            self.embedding.fit(self.network, weight=self.edge_weight)
            # self.log(pprint.pformat(self.embedding._X), force=self.id[1] == 0)


class DQNPPORouterNetwork(NetworkRewardAgent, DQNPPORouter):
    pass


class DQNPPORouterOONetwork(NetworkRewardAgent, DQNPPORouterOO):
    pass


class DQNPPORouterEmbNetwork(NetworkRewardAgent, DQNPPORouterEmb):
    pass


class ConveyorAddInputMixin:
    """
    Mixin which adds conveyor-specific additional NN inputs support
    """

    def _getAddInput(self, tag, nbr=None):
        if tag == 'work_status':
            return np.array(
                list(map(lambda n: self.network.nodes[n].get('works', False), self.nodes)),
                dtype=np.float32)
        if tag == 'working':
            nbr_works = 1 if self.network.nodes[nbr].get('works', False) else 0
            return np.array(nbr_works, dtype=np.float32)
        else:
            return super()._getAddInput(tag, nbr)


class DQNPPORouterConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNPPORouter):
    pass


class DQNPPORouterOOConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNPPORouterOO):
    pass


class DQNPPORouterEmbConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNPPORouterEmb):
    pass

