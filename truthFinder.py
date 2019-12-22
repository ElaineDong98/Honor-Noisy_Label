import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd

def compute_truth(trust, c_df, dampening_factor):
    trust_score = score(trust)
    truth_score = c_df.groupby(
        ['object_id',
         'value']).apply(lambda x: np.sum(trust_score[x.source_id]))
    truth_score = adjust(truth_score)
    truth = to_prob(truth_score, dampening_factor)
    return truth


def compute_trust(truth, c_df):
    trust = c_df.groupby('source_id').apply(lambda x: _compute_trust(
        x[['object_id', 'value']].values, truth))
    return trust


def adjust(truth_score):
    truth_score_df = pd.DataFrame(data=truth_score, columns=['score'])
    truth_score_df.reset_index(inplace=True)

    # adjust
    truth_score_df = truth_score_df.groupby(
        'object_id').apply(lambda x: _adjust(x))

    return truth_score_df.set_index(['object_id', 'value'])['score']


def _adjust(facts, rho=0.5):
    """adjust score of facts about the same object
    
    Parameters
    ----------
    facts: pd.DataFrame
        a panda data frame [[object_id, value, score]]
    
    rho: float
        adjusted parameter
        
    Returns
    -------
    adjusted_scores: np.ndarray
        a 1D numpy array of adjusted score
    """
    values = facts.values[:, 1].copy()
    adjusted_scores = facts.values[:, 2].copy()
    for i in range(len(adjusted_scores)):
        adjusted_score = adjusted_scores[i]
        for j in range(len(adjusted_scores)):
            if j != i:
                adjusted_score += rho * imp(values[j], values[i])

#                 print(f'implication - {imp(values[j], values[i])}')
        adjusted_scores[i] = adjusted_score
    facts['score'] = adjusted_scores
    return facts


def score(x):
    if np.any((1 - x) == 0.):
        raise ValueError('Probs==1 -> -ln(1-x) is invalid')

    return -np.log(1 - x)


def to_prob(x, dampening_factor):
    probs = 1 / (1 + np.exp(-x * dampening_factor))
    # Take care of overconfidence. @TODO implement adjusted confidence score
    probs[np.where(probs == 1.)[0]] = 0.999
    return probs


def sim(f1, f2):
    return np.abs(f1 - f2)


def imp(f1, f2, base_sim=0):
    return np.tanh(sim(f1, f2) - base_sim)


def sim_trust(t1, t2):
    return 1 - cosine(t1, t2)


def _compute_trust(facts, truth):
    n_fact = facts.shape[0]
    total_trust = 0.0
    for f in facts:
        total_trust += truth[tuple(f)]
    return total_trust / n_fact

def truthfinder(claims,
                imp_func=imp,
                initial_trust=0.9,
                similarity_threshold=(1 - 1e-05),
                dampening_factor=0.3,
                verbose=False):
    """
	performs truth discovery using truthfinder
    
    TruthFinder works as follows
    1. Initialize all source reliabilities to initial_t
    2. While sim(t_prev, t_now) < threshold$:
        - compute fact confidences
        - compute website trustworthiness
    where t are the vector of all source reliabities and similarity function is the cosine similarity. 
    Parameters
    ----------
    claims: pd.DataFrame
        a data frame that has columns [source_id, object_id, value]
        
    initial_trust: float
        initial source trustworthiness. This value is in [0, 1]
        
    similarity_threshold: float
        the threshold to determine the convergence of source trustworthiness.
        
    base_sim: float
        This is to define imp
        
    Returns
    -------
    trust_df: pd.DataFrame
    
    truth_df: p.DataFrame
    """
    # compute metadata
    c_df = claims[['source_id', 'object_id', 'value']].copy()
    n_sources = c_df.source_id.nunique()

    # compute trust and truth
    trust = initial_trust * np.ones(n_sources)
    truth = None
    while True:
        truth = compute_truth(trust, c_df, dampening_factor)
        trust_next = compute_trust(truth, c_df)

        if verbose:
            print(f'trust similarity - {sim_trust(trust, trust_next)}')

        if (sim_trust(trust, trust_next) > similarity_threshold):
            trust = trust_next
            break
        else:
            trust = trust_next

    truth_df = pd.DataFrame(data=truth)
    truth_df.reset_index(inplace=True)
    truth_df = truth_df.rename(columns={'score': 'confidence'})
    trust_df = pd.DataFrame(data=trust, columns=['trust_worthiness'])
    trust_df.reset_index(inplace=True)
    return trust_df, truth_df
