from functools import total_ordering
from copy import deepcopy
from typing import Tuple

from .utils import AgentId, InterfaceId

##
# Elementary datatypes
#

class WorldEvent:
    """
    Utility class, which allows access to arbitrary attrs defined
    at object creation. Base class for `Message` and `Action`.
    """

    def __init__(self, **kwargs):
        self.contents = kwargs

    def __str__(self):
        return '{}: {}'.format(self.__class__.__name__, str(self.contents))

    def __repr__(self):
        return '<{}>'.format(str(self))

    def __getattr__(self, name):
        try:
            return super().__getattribute__('contents')[name]
        except KeyError:
            raise AttributeError(name)

    def getContents(self):
        return self.contents


class Message(WorldEvent):
    """
    Event which is a message from some other agent
    """
    pass

class Action(WorldEvent):
    """
    Event which is some agent's action in the physical world
    """
    pass

class UnsupportedEventType(Exception):
    """
    Exception which is thrown by event handlers on encounter of
    unknown event type
    """
    pass

class UnsupportedMessageType(Exception):
    """
    Exception which is thrown by message handlers on encounter of
    unknown message type
    """
    pass

class UnsupportedActionType(Exception):
    """
    Exception which is thrown by the environment on encounter of
    unknown agent action type
    """
    pass

##
# Centralized control
#

class MasterEvent(WorldEvent):
    """
    An event which is actually generated by a master controller,
    which can choose freely which agent should yield an event
    """
    def __init__(self, agent: AgentId, inner: WorldEvent, **kwargs):
        super().__init__(agent=agent, inner=inner, **kwargs)

##
# Delays/timeouts
#

class DelayedEvent(WorldEvent):
    """
    Special wrapper event which should be handled not immediately,
    but after some time. Models the timeout logic in agents themselves.
    """
    def __init__(self, id: int, delay: float, inner: WorldEvent):
        super().__init__(id=id, delay=delay, inner=inner)

class DelayInterrupt(WorldEvent):
    """
    Special event which is used to un-schedule the handling of
    `DelayedEvent`
    """
    def __init__(self, delay_id: int):
        super().__init__(delay_id=delay_id)

def delayed_first(evs):
    def _priority(ev):
        if isinstance(ev, MasterEvent):
            return _priority(ev.inner)
        elif isinstance(ev, DelayedEvent):
            return (0, ev.id)
        elif isinstance(ev, DelayInterrupt):
            return (1, ev.delay_id)
        else:
            return (2, 0)

    return sorted(evs, key=_priority)

##
# Changes in connection graph
#

# Global

class LinkUpdateEvent(WorldEvent):
    """
    A link between nodes has appeared or disappeared
    """
    def __init__(self, u: AgentId, v: AgentId, **kwargs):
        super().__init__(u=u, v=v, **kwargs)

class AddLinkEvent(LinkUpdateEvent):
    """
    Event which router receives when a link is connected (or restored).
    """
    def __init__(self, u: AgentId, v: AgentId, params={}):
        super().__init__(u, v, params=params)

class RemoveLinkEvent(LinkUpdateEvent):
    """
    Event which router receives when link breakage is detected
    """
    def __init__(self, u: AgentId, v: AgentId):
        super().__init__(u, v)

def swapUV(event: LinkUpdateEvent) -> LinkUpdateEvent:
    kwargs = event.getContents()
    u = kwargs.pop('u')
    v = kwargs.pop('v')
    return event.__class__(v, u, **kwargs)

# Handler-level

class InterfaceUpdateEvent(WorldEvent):
    """
    An interface is up or down
    """
    def __init__(self, interface: InterfaceId, **kwargs):
        super().__init__(interface=interface, **kwargs)

class InterfaceSetupEvent(InterfaceUpdateEvent):
    """
    A new interface is set up
    """
    def __init__(self, interface: InterfaceId, neighbour: AgentId, params={}):
        super().__init__(interface, neighbour=neighbour, params=params)

class InterfaceShutdownEvent(InterfaceUpdateEvent):
    """
    An interface is shut down
    """
    def __init__(self, interface: InterfaceId):
        super().__init__(interface)

##
# Core messages, handled by `ConnectionModel`
#

class WireTransferMsg(Message):
    """
    Message which has a payload and a number of interface which it relates to
    (which it came from or which it is going to).
    The `ConnectionModel` deals only with those messages. Interface id -1 is a
    loopback interface.
    """
    def __init__(self, interface: InterfaceId, payload: Message):
        super().__init__(interface=interface, payload=payload)

class WireOutMsg(WireTransferMsg):
    pass

class WireInMsg(WireTransferMsg):
    pass


##
# Basic message classes on a `MessageHandler` level.
#

class InitMessage(Message):
    """
    Message which router receives as environment starts
    """
    def __init__(self, config):
        super().__init__(config=config)

class TransferMessage(Message):
    """
    Wrapper message which is used to send data between nodes
    """
    def __init__(self, from_node: AgentId, to_node: AgentId, inner_msg: Message):
        super().__init__(from_node=from_node, to_node=to_node, inner_msg=inner_msg)

class InMessage(TransferMessage):
    """
    Wrapper message which has came from the outside.
    """
    pass

class OutMessage(TransferMessage):
    """
    Wrapper message which is sent to a neighbor through the interface
    with given ID.
    """
    pass

class DelayTriggerMsg(Message):
    """
    Utility message which is meant to be used only
    with `DelayedEvent`, so that agent can plan some actions
    for some time in the future
    """
    def __init__(self, delay_id: int):
        super().__init__(delay_id=delay_id)

class ServiceMessage(Message):
    """
    Message which does not contain a package and, hence,
    contains no destination.
    """
    pass

class EventMessage(Message):
    """
    Message which contains `Event` which should be handled
    """
    def __init__(self, event: WorldEvent):
        super().__init__(event=event)

class ActionMessage(Message):
    """
    Message which contains `Action` which an agent should perform.
    """
    def __init__(self, action: Action):
        super().__init__(action=action)

class SlaveEvent(WorldEvent):
    """
    An event detected by a slave controller with given ID
    """
    def __init__(self, slave_id: AgentId, inner: WorldEvent):
        super().__init__(slave_id=slave_id, inner=inner)

##
# Computer network events/actions
#

# Class to store partial bag paths
class Bag_info:
    def __init__(self, bag_id, state, max_length=3):
        self.bag_id = bag_id
        self.path = []
        self.max_length = max_length
        self.q_value = None
        self.state = state

    def getPathRouter(self, ind):
        return self.path[ind]

    def getPathLast(self):
        return self.getPathRouter(len(self.path)-1)

    def append(self, router_state_action_reward):
        self.path.append(router_state_action_reward)
        if len(self.path) > self.max_length:
            self.path.pop(0)

    def update(self, idx, router_state_action_reward):
        self.path[idx] = router_state_action_reward

    def updateLast(self, router_state_action_reward):
        self.update(len(self.path)-1,router_state_action_reward)

    def getQvalue(self):
        return self.q_value

    def setQvalue(self, new_q_value):
        self.q_value = new_q_value

    def getRouter(self, ind):
        return self.path[ind][0]

    def setState(self, new_state):
        self.state = new_state

    def getState(self):
        return self.state

    def isFullPath(self):
        return len(self.path) == self.max_length


# Packages
@total_ordering
class Package:
    def __init__(self, pkg_id, size, dst, start_time, contents):
        self.id = pkg_id
        self.size = size
        self.dst = dst
        self.start_time = start_time
        self.contents = contents
        self.node_path = []
        # self.route = None
        # self.rnn_state = (np.zeros((1, state_size)),
        #                   np.zeros((1, state_size)))

    # def route_add(self, data, cols):
    #     if self.route is None:
    #         self.route = pd.DataFrame(columns=cols)
    #     self.route.loc[len(self.route)] = data

    def __str__(self):
        return '{}#{}{}'.format(self.__class__.__name__, self.id,
                                str((self.dst, self.size, self.start_time, self.contents)))

    def __repr__(self):
        return '<{}>'.format(str(self))

    def __hash__(self):
        return hash((self.id, self.contents))

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

class PkgEnqueuedEvent(WorldEvent):
    """
    Some package got enqueued into the router
    """
    def __init__(self, sender: AgentId, recipient: AgentId, pkg: Package):
        super().__init__(sender=sender, recipient=recipient, pkg=pkg)

class PkgProcessingEvent(WorldEvent):
    """
    Some package is now ready to be routed further
    """
    def __init__(self, sender: AgentId, recipient: AgentId, pkg: Package, allowed_nbrs=None):
        super().__init__(sender=sender, recipient=recipient, pkg=pkg, allowed_nbrs=allowed_nbrs)

class PkgReceiveAction(Action):
    """
    A destination router has received a package.
    """
    def __init__(self, pkg: Package):
        super().__init__(pkg=pkg)

class PkgRouteAction(Action):
    """
    Router has re-routed a package to a neighbour
    """
    def __init__(self, to: AgentId, pkg: Package):
        super().__init__(to=to, pkg=pkg)

class PkgRoutePredictionAction(Action):
    """
    Router has re-routed a package to a neighbour
    """
    def __init__(self, to: AgentId, pkg: Package):
        super().__init__(to=to, pkg=pkg)

##
# Conveyors events/actions
#

class Bag(Package):
    def __init__(self, bag_id, dst, start_time, contents):
        super().__init__(bag_id, 0, dst, start_time, contents)
        self.last_conveyor = -1

class BagAppearanceEvent(WorldEvent):
    def __init__(self, src_id: int, bag: Bag):
        super().__init__(src_id=src_id, bag=bag)

class BagDetectionEvent(WorldEvent):
    def __init__(self, bag: Bag):
        super().__init__(bag=bag)

class ConveyorBreakEvent(WorldEvent):
    def __init__(self, conv_idx: int):
        super().__init__(conv_idx=conv_idx)

class ConveyorRestoreEvent(WorldEvent):
    def __init__(self, conv_idx: int):
        super().__init__(conv_idx=conv_idx)

class BagReceiveAction(Action):
    def __init__(self, bag: Bag):
        super().__init__(bag=bag)

class DiverterKickAction(Action):
    def __init__(self):
        super().__init__()

class ConveyorSpeedChangeAction(Action):
    def __init__(self, new_speed: float):
        super().__init__(new_speed=new_speed)

class IncomingBagEvent(WorldEvent):
    def __init__(self, sender: AgentId, bag: Bag, node: AgentId):
        super().__init__(sender=sender, bag=bag, node=node)

class OutgoingBagEvent(WorldEvent):
    def __init__(self, bag: Bag, node: AgentId):
        super().__init__(bag=bag, node=node)

class PassedBagEvent(WorldEvent):
    def __init__(self, bag: Bag, node: AgentId):
        super().__init__(bag=bag, node=node)


#
# Service messages
#
class GetBagInfoMsg(ServiceMessage):
    def __init__(self, origin: AgentId, bag_id):
        super().__init__(origin=origin, bag_id=bag_id)

class UpdateTableMsg(ServiceMessage):
    def __init__(self, origin: AgentId, bag_info):
        super().__init__(origin=origin, bag_info=bag_info)

class PathRewardMsg(ServiceMessage):
    def __init__(self, origin: AgentId, bag_info, count, all_learn=True):
        super().__init__(origin=origin, bag_info=bag_info, count=count, all_learn=all_learn)


class RewardMsg(ServiceMessage):
    def __init__(self, origin: AgentId, pkg: Package, Q_estimate: float, reward_data):
        super().__init__(origin=origin, pkg=pkg, Q_estimate=Q_estimate,
                         reward_data=reward_data)

class NetworkRewardMsg(RewardMsg):
    def __init__(self, origin: AgentId, pkg: Package, Q_estimate: float, time_received: float):
        super().__init__(origin, pkg, Q_estimate, time_received)

class ConveyorRewardMsg(RewardMsg):
    def __init__(self, origin: AgentId, bag: Bag, Q_estimate: float,
                 time_processed: float, energy_gap: float):
        super().__init__(origin, bag, Q_estimate, (time_processed, energy_gap))

class TrainingRewardMsg(RewardMsg):
    def __init__(self, orig_msg: RewardMsg, true_reward=None):
        super().__init__(orig_msg.origin, orig_msg.pkg,
                         orig_msg.Q_estimate, (orig_msg.reward_data, true_reward))

class StateAnnouncementMsg(ServiceMessage):
    def __init__(self, node: AgentId, seq: int, state):
        super().__init__(node=node, seq=seq, state=state)

class WrappedRouterMsg(ServiceMessage):
    """
    Wrapped message which allows to reuse router code in conveyors
    """
    def __init__(self, from_router: AgentId, to_router: AgentId, inner: Message):
        super().__init__(from_router=from_router, to_router=to_router, inner=inner)

#
# Conveyor control messages
#

class ConveyorBagMsg(ServiceMessage):
    def __init__(self, bag: Bag, node: AgentId):
        super().__init__(bag=bag, node=node)

class IncomingBagMsg(ConveyorBagMsg):
    pass

class OutgoingBagMsg(ConveyorBagMsg):
    pass

class PassedBagMsg(ConveyorBagMsg):
    pass

class ConveyorStartMsg(ServiceMessage):
    pass

class ConveyorStopMsg(ServiceMessage):
    pass

class StopTimeUpdMsg(ServiceMessage):
    def __init__(self, time: float):
        super().__init__(time=time)

class DiverterNotification(ServiceMessage):
    def __init__(self, bag: Bag, pos: float):
        super().__init__(bag=bag, pos=pos)

class DiverterPrediction(ServiceMessage):
    def __init__(self, bag: Bag, kick: bool):
        super().__init__(bag=bag, kick=kick)
