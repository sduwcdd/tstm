from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.sgsac import SGSAC
from algorithms.madi import MaDi
from algorithms.madi_compare import MaDiCompare
from algorithms.madi_compare_online import MaDiCompareOnline
from algorithms.simgrl import SimGRL
from algorithms.tstm import TSTM
from algorithms.ppo import PPO
from algorithms.tstm_ppo import TSTM_PPO
from algorithms.tstm_ablation import (
    TSTM_NoSeg,
    TSTM_NoVICReg,
    TSTM_NoPolicyConsistency
)
algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "sgsac": SGSAC,
    "madi": MaDi,
    "madi_compare": MaDiCompare,
    "madi_compare_online": MaDiCompareOnline,
    "simgrl": SimGRL,
    "ppo": PPO,
    "tstm_ppo": TSTM_PPO,
    "tstm": TSTM,
    #ablation
    "tstm_noseg": TSTM_NoSeg,
    "tstm_novicreg": TSTM_NoVICReg,
    "tstm_noINV": TSTM_NoVICReg,
    "tstm_nopc": TSTM_NoPolicyConsistency,
}


def make_agent(obs_shape, action_shape, args):
    return algorithm[args.algorithm](obs_shape, action_shape, args)
