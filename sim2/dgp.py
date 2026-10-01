"""
Drift attribution under deployment feedback: simulator.

Union-graph SCM   X -> r_hat -> A -> Y   with mechanisms
    m_X    : P(X)
    m_pi   : pi(A | X, r_hat)        (action / alerting policy)
    m_Y    : P(Y | X, A)             (outcome mechanism)

Two environments e0 (pre-deployment) and e1 (post-deployment).
Deployment switches m_pi from pi0 (clinician judgement only) to pi1
(clinician judgement OR model alert).

Three attribution estimators, one per stated assumption of
Zhang et al. (ICML 2023):

  naive2   two players {P(X), P(Y|X)}, marginalising over A.
           This is what a monitor WITHOUT action logging can compute.
           Ties to Assumption 3.1 (all variables observed).

  union3   three players {P(X), pi, P(Y|X,A)} on the union graph,
           importance-sampling estimator (their Eq. 5).
           Ties to Assumption 3.3 (support inclusion): the weight
           pi1(a|x,r)/pi0(a|x) diverges on the alert-only region.

  union3_clip   same, with importance weights clipped at c -- the fix
           practitioners actually apply. Finite, smooth, biased.

Ground truth is obtained by forward-sampling the known DGP with
mechanism subset S swapped -- available to us, not to an analyst
holding only observational e0/e1 data.
"""
import itertools
import numpy as np

SQ2 = np.sqrt(2.0)


# ----------------------------------------------------------------- config
class Config:
    """Parameters of the data-generating process."""

    def __init__(
        self,
        eps=1e-3,          # pi0 mass on treatment in the alert-only region
        theta=1.5,         # treatment effect on the outcome logit (>0 = benefit)
        delta=0.0,         # exogenous covariate-mean shift e0 -> e1
        dy=0.0,            # exogenous outcome-mechanism shift (logit intercept)
        phi=np.pi / 3.0,   # angle between clinician and model directions
        tau_s=0.6,         # clinician suspicion threshold
        tau_r=0.6,         # model alerting threshold (on r_hat)
        q_hi=0.80,         # P(treat) when clinician is suspicious
        p_alert=0.70,      # P(treat) when the model alerts
        b0=-0.4,           # outcome logit intercept
        triage=0.0,        # endogenous P(X) shift caused by deployment
        deploy=True,       # does e1 actually differ from e0 in the policy?
        seed=0,
        d=2,               # extra coordinates are available for robustness runs
        nonlinear_outcome=0.0,  # truth-only interaction; deployed score stays logistic-linear
    ):
        self.deploy = deploy
        self.eps, self.theta, self.delta, self.dy = eps, theta, delta, dy
        self.phi, self.tau_s, self.tau_r = phi, tau_s, tau_r
        self.q_hi, self.p_alert, self.b0 = q_hi, p_alert, b0
        self.triage, self.seed = triage, seed
        if d < 2:
            raise ValueError("d must be at least 2")
        self.d, self.nonlinear_outcome = d, nonlinear_outcome
        # model direction (well specified w.r.t. the outcome), clinician direction
        self.w_r = np.zeros(d)
        self.w_r[:2] = 1.0 / SQ2
        if d > 2:
            self.w_r[2:] = 0.30 / np.sqrt(d - 2)
        self.w_s = np.zeros(d)
        self.w_s[:2] = [np.cos(phi), np.sin(phi)]
        self.w_y = self.w_r.copy()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------- mechanism pieces
def mu_of(cfg, env, mechs):
    """Mean of P(X) in the coalition environment."""
    m = np.zeros(cfg.d)
    if "X" in mechs:
        m[:2] += cfg.delta                     # same shift magnitude across dimensions
    if "pi" in mechs:
        m[:2] += cfg.triage                    # deployment-induced triage shift
    return m


def logpdf_x(cfg, x, mu):
    d = x - mu
    return -0.5 * (d ** 2).sum(1) - 0.5 * x.shape[1] * np.log(2 * np.pi)


def r_hat(cfg, x):
    """The audited model's predicted risk."""
    return sigmoid(x @ cfg.w_r + cfg.b0)


def suspicion(cfg, x):
    return x @ cfg.w_s


def p_treat(cfg, x, deployed):
    """pi(A=1 | X, r_hat). deployed=False -> pi0, True -> pi1."""
    susp = suspicion(cfg, x) > cfg.tau_s
    p = np.where(susp, cfg.q_hi, cfg.eps)
    if deployed:
        alert = r_hat(cfg, x) > cfg.tau_r
        p = np.where(susp, cfg.q_hi, np.where(alert, cfg.p_alert, cfg.eps))
    return p


def p_y1(cfg, x, a, shifted):
    """P(Y=1 | X, A)."""
    lin = x @ cfg.w_y + cfg.b0 - cfg.theta * a
    if cfg.nonlinear_outcome:
        third = x[:, 2] if cfg.d > 2 else x[:, 1]
        lin = lin + cfg.nonlinear_outcome * (x[:, 0] * third + 0.3 * (third ** 2 - 1))
    if shifted:
        lin = lin + cfg.dy
    return sigmoid(lin)


# ------------------------------------------------------------- sampling
def sample_env(cfg, mechs, n, rng):
    """Forward-sample the coalition environment where mechanisms in
    `mechs` take their e1 form and the rest their e0 form."""
    mu = mu_of(cfg, None, mechs)
    x = rng.normal(size=(n, cfg.d)) + mu
    a = (rng.random(n) < p_treat(cfg, x, deployed=("pi" in mechs and cfg.deploy))).astype(float)
    y = (rng.random(n) < p_y1(cfg, x, a, shifted=("Y" in mechs))).astype(float)
    return x, a, y


# ------------------------------------------------------------- metrics
def brier(cfg, x, y, w=None):
    r = r_hat(cfg, x)
    v = (r - y) ** 2
    return float(np.average(v, weights=w))


def auroc(cfg, x, y, w=None):
    r = r_hat(cfg, x)
    if w is None:
        w = np.ones_like(r)
    o = np.argsort(r)
    r, y, w = r[o], y[o], w[o]
    pos, neg = w * y, w * (1 - y)
    # weighted Mann-Whitney with tie handling via cumulative negatives
    cneg = np.cumsum(neg) - neg / 2.0
    num = float((pos * cneg).sum())
    den = float(pos.sum() * neg.sum())
    return num / den if den > 0 else np.nan


def net_benefit(cfg, x, y, t, w=None):
    """Vickers net benefit at threshold t, computed on observed (post-
    treatment) outcomes -- i.e. the quantity a naive DCA reports."""
    r = r_hat(cfg, x)
    if w is None:
        w = np.ones_like(r)
    w = w / w.sum()
    flag = r > t
    tp = float((w * (flag & (y == 1))).sum())
    fp = float((w * (flag & (y == 0))).sum())
    return tp - fp * (t / (1.0 - t))


def event_rate(cfg, x, y, w=None):
    return float(np.average(y, weights=w))


METRICS = {"brier": brier, "auroc": auroc, "event_rate": event_rate}


# --------------------------------------------------- ground-truth values
def truth_values(cfg, players, n=400_000, metric="brier", seed=0):
    """v(S) for every coalition, by forward simulation of the true DGP."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(len(players) + 1):
        for S in itertools.combinations(players, k):
            x, a, y = sample_env(cfg, set(S), n, rng)
            out[frozenset(S)] = METRICS[metric](cfg, x, y)
    return out


# ------------------------------------------------- IS estimator (union3)
def _weights_union3(cfg, x, a, y, S, clip=None):
    """Product of e1/e0 mechanism density ratios for coalition S."""
    w = np.ones(len(x))
    diag = {}
    if "X" in S or "pi" in S:
        mu1 = mu_of(cfg, None, S & {"X", "pi"})
        wx = np.exp(logpdf_x(cfg, x, mu1) - logpdf_x(cfg, x, np.zeros(cfg.d)))
        w = w * wx
    if "pi" in S:
        p1 = p_treat(cfg, x, deployed=cfg.deploy)
        p0 = p_treat(cfg, x, deployed=False)
        num = np.where(a == 1, p1, 1 - p1)
        den = np.where(a == 1, p0, 1 - p0)
        wpi = num / den
        diag["max_w_pi"] = float(np.max(wpi))
        diag["n_eff_divergent"] = int(np.sum(wpi > 100))
        diag["ess"] = float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0
        w = w * wpi
    if "Y" in S:
        q1 = p_y1(cfg, x, a, shifted=True)
        q0 = p_y1(cfg, x, a, shifted=False)
        w = w * (np.where(y == 1, q1, 1 - q1) / np.where(y == 1, q0, 1 - q0))
    if clip is not None:
        w = np.minimum(w, clip)
    return w, diag


def is_values(cfg, players, n=20_000, metric="brier", clip=None, seed=1):
    """v(S) estimated by importance sampling from an e0 sample only."""
    rng = np.random.default_rng(seed)
    x, a, y = sample_env(cfg, set(), n, rng)
    out, diags = {}, {}
    for k in range(len(players) + 1):
        for S in itertools.combinations(players, k):
            Sf = frozenset(S)
            w, d = _weights_union3(cfg, x, a, y, set(S), clip=clip)
            out[Sf] = METRICS[metric](cfg, x, y, w=w)
            diags[Sf] = d
    return out, diags


# ----------------------------------------------------- naive 2-player IS
def _p_y1_marg(cfg, x, deployed, shifted):
    p = p_treat(cfg, x, deployed)
    return p * p_y1(cfg, x, np.ones(len(x)), shifted) + \
        (1 - p) * p_y1(cfg, x, np.zeros(len(x)), shifted)


def naive2_values(cfg, n=20_000, metric="brier", seed=1):
    """Two players {P(X), P(Y|X)}: A is unobserved, so the analyst sees
    only the marginal outcome mechanism."""
    rng = np.random.default_rng(seed)
    x, a, y = sample_env(cfg, set(), n, rng)
    out = {}
    for S in [(), ("X",), ("Ygx",), ("X", "Ygx")]:
        w = np.ones(len(x))
        if "X" in S:
            w = w * np.exp(logpdf_x(cfg, x, mu_of(cfg, None, {"X"}))
                           - logpdf_x(cfg, x, np.zeros(cfg.d)))
        if "Ygx" in S:
            # e1 marginal: deployment on (pi1) and any exogenous outcome shift
            q1 = _p_y1_marg(cfg, x, deployed=cfg.deploy, shifted=True)
            q0 = _p_y1_marg(cfg, x, deployed=False, shifted=False)
            w = w * (np.where(y == 1, q1, 1 - q1) / np.where(y == 1, q0, 1 - q0))
        out[frozenset(S)] = METRICS[metric](cfg, x, y, w=w)
    return out


# --------------------------------------------------------------- Shapley
def shapley(values, players):
    """Exact Shapley values from a full coalition-value table."""
    P = list(players)
    n = len(P)
    phi = {p: 0.0 for p in P}
    from math import factorial
    fact = [float(factorial(k)) for k in range(n + 1)]
    for p in P:
        rest = [q for q in P if q != p]
        for k in range(n):
            for S in itertools.combinations(rest, k):
                Sf, Spf = frozenset(S), frozenset(S) | {p}
                wgt = fact[k] * fact[n - k - 1] / fact[n]
                phi[p] += wgt * (values[Spf] - values[Sf])
    return phi
