from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .vffac_learner import QLearner as VffacLearner
from .immac_learner import QLearner as ImmacLearner
from .marnd_learner import QLearner as MarndLearner

REGISTRY = {
    "q_learner": QLearner,
    "coma_learner": COMALearner,
    "qtran_learner": QTranLearner,
    "vffac_learner": VffacLearner,
    "immac_learner": ImmacLearner,
    "marnd_learner": MarndLearner
}

