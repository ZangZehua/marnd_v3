from .rnn_agent import RNNAgent
from .rnn_msg_agent import RnnMsgAgent
from .immac_agent import ImmacAgent
from .marnd_agent import MarndAgent

REGISTRY = {
    "rnn": RNNAgent,
    'rnn_msg': RnnMsgAgent,
    'immac_agent': ImmacAgent,
    'immac-vffac': RnnMsgAgent,
    "marnd_agent": MarndAgent
}

