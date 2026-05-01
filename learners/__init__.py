from .q_learner import QLearner
from .qtran_learner import QLearner as QTranLearner
from .max_q_learner import MAXQLearner
from .opt_q_learner import OPTQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["opt_q_learner"] = OPTQLearner
