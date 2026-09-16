"""
Drift attribution under deployment feedback: simulator core.

World model (per environment e):
    X ~ N(mu_e, I_d)                       covariates
    r_hat = f(X)                            deployed risk model (fixed logistic model, trained in e0)
    A ~ pi_e(a | x, r_hat)                  action: pi_0 = standard care, pi_1 = alert-mediated policy
    Y ~ Bernoulli(sigmoid(beta_e . x + b_e + beta_A * A))   outcome, beta_A < 0 (treatment helps)

Scenarios (differences between e0 and e1):
    S1  covariate shift          mu_1 = mu_0 + delta
    S2  outcome-mechanism shift  (beta_1, b_1) != (beta_0, b_0)   [coding / biology / care pathway]
    S3  performative success     pi_0 -> pi_1 only
    S4  all three

Estimators of the decomposition of  Delta R = R_e1(f) - R_e0(f)  (log loss):
    E1  marginal Shapley game over {P(X), P(Y|X)}                      (monitoring pipeline, A not logged)
    E2  Zhang et al. game over {P(X), P(A|X,r), P(Y|X,A)}              (reports positivity failure)
    E3  proposed: exogenous Shapley over {P(X), P(Y|X,A)} with the policy held at pi_0,
        plus a policy term Delta_pi = R_e1(f; pi_1) - R_e1(f; pi_0) evaluated inside e1
        (a) oracle policy weights, (b) randomised unalerted arm inside e1, (c) threshold discontinuity
"""
import numpy as np
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logloss(p, y, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


@dataclass
class Env:
    mu: np.ndarray                # covariate mean
    beta: np.ndarray              # outcome coefficients
    b: float                      # outcome intercept
    beta_A: float                 # treatment effect on the logit (negative = protective)
    policy: str                   # "pi0" or "pi1"
    tau: float = 0.3              # alert threshold on r_hat
    p_adh: float = 0.7            # clinician adherence to an alert
    p_std_high: float = 0.9       # standard care treats obvious cases with this prob
    c_high: float = 2.0           # "obvious case" cutoff on the severity index s(x) = x[:,0]
    rollout_control: float = 0.0  # fraction of e1 patients randomised to an unalerted arm (pi_0)


@dataclass
class Sample:
    X: np.ndarray
    r: np.ndarray                 # r_hat = f(X)
    A: np.ndarray
    Y: np.ndarray
    Y0: np.ndarray                # potential outcome under A = 0 (oracle, for utility metrics)
    Y1: np.ndarray                # potential outcome under A = 1 (oracle)
    alerted: np.ndarray           # r_hat >= tau
    arm: np.ndarray               # 1 = alert-mediated policy applied, 0 = randomised control (pi_0)
    p_pi0: np.ndarray             # pi_0(A = a_obs | x)
    p_pi1: np.ndarray             # pi_1(A = a_obs | x, r)


def pi0_prob(X, env):
    """Standard care: treat obvious cases only. Returns P(A=1 | x)."""
    s = X[:, 0]
    return np.where(s > env.c_high, env.p_std_high, 0.0)


def pi1_prob(X, r, env):
    """Alert-mediated policy: alerted patients are treated on adherence OR by standard care."""
    p0 = pi0_prob(X, env)
    alerted = r >= env.tau
    p_alert = 1.0 - (1.0 - env.p_adh) * (1.0 - p0)
    return np.where(alerted, p_alert, p0)


def sample_env(env, f, n, rng):
    d = env.mu.shape[0]
    X = rng.normal(env.mu, 1.0, size=(n, d))
    r = f(X)
    logit0 = X @ env.beta + env.b
    # potential outcomes under a monotone coupling (same latent uniform): same latent uniform, shifted logit (monotone coupling)
    u = rng.uniform(size=n)
    Y0 = (u < sigmoid(logit0)).astype(int)
    Y1 = (u < sigmoid(logit0 + env.beta_A)).astype(int)
    p0 = pi0_prob(X, env)
    p1 = pi1_prob(X, r, env)
    arm = np.ones(n, dtype=int)
    if env.policy == "pi1" and env.rollout_control > 0:
        arm = (rng.uniform(size=n) >= env.rollout_control).astype(int)
    p_apply = np.where((env.policy == "pi1") & (arm == 1), p1, p0)
    A = (rng.uniform(size=n) < p_apply).astype(int)
    Y = np.where(A == 1, Y1, Y0)
    p_pi0_obs = np.where(A == 1, p0, 1 - p0)
    p_pi1_obs = np.where(A == 1, p1, 1 - p1)
    return Sample(X, r, A, Y, Y0, Y1, r >= env.tau, arm, p_pi0_obs, p_pi1_obs)


# ----------------------------------------------------------------------------- model
def train_risk_model(X, Y):
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(X, Y)
    return lambda Xn: clf.predict_proba(Xn)[:, 1], clf


# ----------------------------------------------------------------------------- density ratio
def density_ratio_X(X0, X1):
    """Classifier-based estimate of P1(X)/P0(X) evaluated on both samples."""
    Xc = np.vstack([X0, X1]); z = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xc, z)
    def ratio(X):
        p = np.clip(clf.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
        return (p / (1 - p)) * (len(X0) / len(X1))
    return ratio


def wmean(v, w):
    return np.sum(v * w) / np.sum(w)


# ----------------------------------------------------------------------------- estimators
def shapley_two(vX, vY, vAll):
    """Shapley values for a two-player game with players X and Y|X."""
    phi_X = 0.5 * (vX + (vAll - vY))
    phi_Y = 0.5 * (vY + (vAll - vX))
    return phi_X, phi_Y


def E1_marginal(s0, s1, f, ratio):
    """Marginal game over {P(X), P(Y|X)}; A is not used (not logged)."""
    m = s1.arm == 1                   # the monitored (deployed) population; a control arm is not part of monitoring
    L0, L1 = logloss(s0.r, s0.Y), logloss(s1.r[m], s1.Y[m])
    R0, R1 = L0.mean(), L1.mean()
    wX0 = ratio(s0.X)                 # P1(X)/P0(X) on e0 samples
    wX1 = 1.0 / ratio(s1.X[m])        # P0(X)/P1(X) on e1 samples
    vX = wmean(L0, wX0) - R0          # P1(X) P0(Y|X)
    vY = wmean(L1, wX1) - R0          # P0(X) P1(Y|X)
    vAll = R1 - R0
    phi_X, phi_Y = shapley_two(vX, vY, vAll)
    return {"dR": vAll, "phi_X": phi_X, "phi_YgX": phi_Y}


def E2_zhang_with_action(s0, s1, f, ratio, env1, clip=50.0):
    """Zhang et al. with the action mechanism as a player. Reports positivity failure.
    Coalitions containing P(A|X,r) but not P(Y|X,A) need e0 outcomes reweighted by pi1/pi0."""
    # weight needed on e0 samples to emulate pi1 actions: pi1(a_obs|x,r)/pi0(a_obs|x)
    p0 = pi0_prob(s0.X, env1); p1 = pi1_prob(s0.X, s0.r, env1)
    p0_obs = np.where(s0.A == 1, p0, 1 - p0); p1_obs = np.where(s0.A == 1, p1, 1 - p1)
    w_pol = np.where(p0_obs > 0, p1_obs / np.where(p0_obs > 0, p0_obs, 1.0), np.nan)
    # the emulation also needs e0 outcomes for (x, a=1) where pi0(1|x)=0 but pi1(1|x,r)>0: no such samples exist
    missing_cells = (p0 == 0) & (p1 > 0)
    undefined = np.isnan(w_pol) | missing_cells
    # share of e1 policy mass with no support under pi0
    no_support_mass = float(np.mean(s1.alerted & (pi0_prob(s1.X, env1) == 0)) * env1.p_adh)
    # trimmed variant: drop undefined, clip the rest
    L0 = logloss(s0.r, s0.Y)
    w_tr = np.where(undefined, 0.0, np.clip(np.nan_to_num(w_pol, nan=0.0), 0, clip))
    R_pol_only_trimmed = wmean(L0, w_tr) if w_tr.sum() > 0 else np.nan
    return {"frac_undefined_weights": float(np.mean(undefined)),
            "no_support_mass_e1": float(no_support_mass),
            "R_pi1_on_e0_trimmed": float(R_pol_only_trimmed),
            "R0": float(L0.mean())}


def E3_proposed(s0, s1, f, ratio, env1, mode="oracle"):
    """Exogenous Shapley over {P(X), P(Y|X,A)} with policy fixed at pi_0, plus the policy term.
    mode='oracle': reweight e1 samples by pi0/pi1 (requires both policies known).
    mode='rollout': use the randomised unalerted arm inside e1 to estimate R_e1(f; pi_0)."""
    L0, L1 = logloss(s0.r, s0.Y), logloss(s1.r, s1.Y)
    R0 = L0.mean()
    treated_arm = s1.arm == 1
    R1 = L1[treated_arm].mean()                       # observed post-deployment risk (policy pi1)
    wX0 = ratio(s0.X); wX1 = 1.0 / ratio(s1.X)
    if mode == "oracle":
        # pi0(a|x)/pi1(a|x,r): finite since supp pi0 <= supp pi1; equals 1 if e1 still runs pi0
        w_pol = s1.p_pi0 / s1.p_pi1 if env1.policy == "pi1" else np.ones(len(s1.Y))
        m = treated_arm
        R_e1_pi0 = wmean(L1[m], w_pol[m])                          # P1(X) pi0 P1(Y|X,A)
        R_mixed = wmean(L1[m], (wX1 * w_pol)[m])                   # P0(X) pi0 P1(Y|X,A)
    else:
        m = s1.arm == 0
        if m.sum() < 50: raise ValueError("no control arm")
        R_e1_pi0 = L1[m].mean()
        R_mixed = wmean(L1[m], wX1[m])
    vX = wmean(L0, wX0) - R0                                       # P1(X) pi0 P0(Y|X,A)
    vY = R_mixed - R0
    vAll_exo = R_e1_pi0 - R0
    phi_X, phi_Y = shapley_two(vX, vY, vAll_exo)
    delta_pi = R1 - R_e1_pi0
    return {"dR": R1 - R0, "phi_X": phi_X, "phi_YgXA": phi_Y, "delta_pi": delta_pi}


def rd_local_effect(s1, tau, h=0.05):
    """Threshold discontinuity on the outcome: local linear fit of Y on (r - tau) each side."""
    m = (s1.arm == 1) & (np.abs(s1.r - tau) < h)
    r = s1.r[m] - tau; y = s1.Y[m]; side = (r >= 0).astype(float)
    Xd = np.c_[np.ones(len(r)), r, side, r * side]
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return float(coef[2])   # jump at tau


# ----------------------------------------------------------------------------- utility metrics
def policy_value_events(s):
    """Events under the realised policy (oracle counterfactuals available)."""
    return float(np.mean(s.Y))


def counterfactual_net_benefit(s, tau, use_untreated_truth=True):
    """Net benefit of alerting at tau. Standard DCA uses observed Y; the counterfactual version
    uses Y0 (the event that would occur without the model-triggered action), which is the
    quantity the alert is meant to catch."""
    y = s.Y0 if use_untreated_truth else s.Y
    pred = s.r >= tau
    tp = np.mean(pred & (y == 1)); fp = np.mean(pred & (y == 0))
    return float(tp - fp * tau / (1 - tau))


def auroc(s, use_untreated_truth=False):
    y = s.Y0 if use_untreated_truth else s.Y
    return float(roc_auc_score(y, s.r))
