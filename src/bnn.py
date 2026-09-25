"""
The four Bayesian models of the paper, reimplemented in TensorFlow.

  LinearGaussian   linear, Gaussian prior on all weights
  GaussianBNN      1 hidden layer, Gaussian prior on all weights
  LinearHorseshoe  linear, horseshoe prior on all weights
  HorseshoeBNN     1 hidden layer, *tied* horseshoe prior on the first layer,
                   Gaussian prior on the output layer   <- the paper's model

"Tied" is the paper's contribution: every weight leaving input feature j shares
one shrinkage parameter tau_j, so shrinking tau_j switches feature j off as a
whole.  Non-centred parameterisation:

    W1[j, i] = tau_j * lambda_ * beta[j, i],   beta ~ N(0, 1)
    tau_j    ~ HalfCauchy(b_0)        per input feature  (local)
    lambda_  ~ HalfCauchy(b_g)        shared             (global)

Reimplementation choices, all deliberate and all listed in the README:

  * Framework.  The authors' code is PyTorch; this is TensorFlow, because that
    is what was installed on the machine this ran on.  The maths is the paper's.
  * Variational family.  Mean-field Gaussian on beta and the biases, log-normal
    on tau and lambda.  The paper cites Louizos et al. (2017), who introduce
    auxiliary inverse-gamma variables to make the half-Cauchy KL analytic.  We
    keep the half-Cauchy prior exactly as specified and estimate its KL by
    Monte Carlo with the same samples already drawn for the likelihood, which
    needs no auxiliary variables.  Same prior, different estimator.
  * Training length.  Table 8 says 5000 epochs; that is a cap, and the paper
    says models are "trained until convergence".  We early-stop on a held-out
    slice of the training fold.  Actual epoch counts are logged per fold.

Everything else -- 50 hidden units, lr 1e-3, batch 64, 10 weight samples for
training and 100 for testing, prior sd 1.0, b_0 = b_g = 1.0 -- is Table 8.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from paper_spec import HYPERPARAMS

LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------- utilities --
def _softplus_inv(x):
    return np.log(np.expm1(x))


def _gaussian_kl(mu, sigma, prior_sd):
    """Analytic KL( N(mu, sigma^2) || N(0, prior_sd^2) ), summed."""
    var = tf.square(sigma)
    pv = prior_sd ** 2
    return 0.5 * tf.reduce_sum(
        (var + tf.square(mu)) / pv - 1.0 - tf.math.log(var / pv))


def _log_half_cauchy(x, scale):
    """log density of HalfCauchy(scale) at x > 0."""
    return (np.log(2.0) - np.log(np.pi) - tf.math.log(scale)
            - tf.math.log1p(tf.square(x / scale)))


def _log_lognormal(x, mu, sigma):
    """log density of a log-normal at x > 0, given log x ~ N(mu, sigma^2)."""
    lx = tf.math.log(x)
    return (-lx - tf.math.log(sigma) - 0.5 * LOG2PI
            - 0.5 * tf.square((lx - mu) / sigma))


class _ScaleVariable:
    """A positive latent with a log-normal posterior and half-Cauchy prior."""

    def __init__(self, shape, prior_scale, name, init_log_mu=0.0):
        self.prior_scale = float(prior_scale)
        self.mu = tf.Variable(tf.fill(shape, float(init_log_mu)),
                              name=f"{name}_logmu")
        self.rho = tf.Variable(tf.fill(shape, float(_softplus_inv(0.1))),
                               name=f"{name}_logsigma")

    @property
    def sigma(self):
        return tf.nn.softplus(self.rho)

    def sample(self, n_samples):
        """Returns (n_samples, *shape) positive draws."""
        s = self.sigma
        eps = tf.random.normal((n_samples,) + tuple(self.mu.shape))
        return tf.exp(self.mu + s * eps)

    def kl_mc(self, draws):
        """MC estimate of KL(q||p), averaged over the sample axis."""
        logq = _log_lognormal(draws, self.mu, self.sigma)
        logp = _log_half_cauchy(draws, tf.constant(self.prior_scale))
        return tf.reduce_mean(tf.reduce_sum(logq - logp,
                                            axis=list(range(1, len(draws.shape)))))

    @property
    def mean(self):
        """E[x] for log-normal."""
        return tf.exp(self.mu + 0.5 * tf.square(self.sigma))

    def variables(self):
        return [self.mu, self.rho]


class _GaussianVariable:
    """A mean-field Gaussian weight block with an N(0, prior_sd^2) prior."""

    def __init__(self, shape, prior_sd, name, init_sd=0.05):
        self.prior_sd = float(prior_sd)
        init = tf.random.normal(shape, stddev=init_sd)
        self.mu = tf.Variable(init, name=f"{name}_mu")
        self.rho = tf.Variable(tf.fill(shape, float(_softplus_inv(0.05))),
                               name=f"{name}_rho")

    @property
    def sigma(self):
        return tf.nn.softplus(self.rho)

    def sample(self, n_samples):
        eps = tf.random.normal((n_samples,) + tuple(self.mu.shape))
        return self.mu + self.sigma * eps

    def kl(self):
        return _gaussian_kl(self.mu, self.sigma, self.prior_sd)

    def variables(self):
        return [self.mu, self.rho]


# ------------------------------------------------------------------ models --
class BayesianModel:
    """
    Shared training loop.  Subclasses provide `_forward` and `_kl`.

    kind: "linear" or "bnn";  prior: "gaussian" or "horseshoe".
    """

    def __init__(self, n_features, kind="bnn", prior="gaussian",
                 hp=None, seed=0):
        self.hp = dict(HYPERPARAMS if hp is None else hp)
        self.n_features = n_features
        self.kind = kind
        self.prior = prior
        self.seed = seed
        tf.random.set_seed(seed)

        sd = self.hp["gaussian_prior_sd"]
        h = self.hp["n_hidden_units"]
        self._params = []

        if kind == "linear":
            out_in = n_features
        else:
            self.W1 = _GaussianVariable((n_features, h), sd, "W1")
            self.b1 = _GaussianVariable((h,), sd, "b1")
            self._params += self.W1.variables() + self.b1.variables()
            out_in = h

        self.W2 = _GaussianVariable((out_in, 1), sd, "W2")
        self.b2 = _GaussianVariable((1,), sd, "b2")
        self._params += self.W2.variables() + self.b2.variables()

        if prior == "horseshoe":
            # tau is per *input feature* -- this is the "tied" part.
            self.tau = _ScaleVariable((n_features,), self.hp["horseshoe_b_0"],
                                      "tau", init_log_mu=-1.0)
            self.lam = _ScaleVariable((), self.hp["horseshoe_b_g"],
                                      "lambda", init_log_mu=0.0)
            self._params += self.tau.variables() + self.lam.variables()
        else:
            self.tau = self.lam = None

        self.history_ = []

    # -- forward ------------------------------------------------------------
    def _scaled_first_layer(self, n_samples):
        """Draw the feature-facing weight block, applying the tied horseshoe."""
        base = self.W1 if self.kind == "bnn" else self.W2   # (F, h) or (F, 1)
        w = base.sample(n_samples)                          # (S, F, *)
        aux = {}
        if self.prior == "horseshoe":
            tau = self.tau.sample(n_samples)                # (S, F)
            lam = self.lam.sample(n_samples)                # (S,)
            scale = tau * lam[:, None]                      # (S, F)
            w = w * scale[:, :, None]
            aux = {"tau": tau, "lam": lam}
        return w, aux

    def _forward(self, X, n_samples):
        """Returns logits (S, N) and the auxiliary draws used."""
        w_first, aux = self._scaled_first_layer(n_samples)
        if self.kind == "linear":
            b2 = self.b2.sample(n_samples)                       # (S, 1)
            logits = tf.einsum("nf,sfo->sno", X, w_first)[..., 0]
            logits = logits + b2
        else:
            b1 = self.b1.sample(n_samples)                       # (S, h)
            hpre = tf.einsum("nf,sfh->snh", X, w_first) + b1[:, None, :]
            hact = tf.nn.relu(hpre)
            W2 = self.W2.sample(n_samples)                       # (S, h, 1)
            b2 = self.b2.sample(n_samples)                       # (S, 1)
            logits = tf.einsum("snh,sho->sno", hact, W2)[..., 0]
            logits = logits + b2
        return logits, aux

    # -- KL -----------------------------------------------------------------
    def _kl(self, aux):
        kl = self.b2.kl()
        if self.kind == "bnn":
            kl += self.W1.kl() + self.b1.kl() + self.W2.kl()
        else:
            kl += self.W2.kl()
        if self.prior == "horseshoe":
            kl += self.tau.kl_mc(aux["tau"]) + self.lam.kl_mc(aux["lam"][:, None])
        return kl

    # -- training -----------------------------------------------------------
    def fit(self, X, y, max_epochs=None, patience=8, val_frac=0.1,
            batch_size=None, verbose=False, on_epoch=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        rng = np.random.default_rng(self.seed)

        idx = rng.permutation(len(X))
        n_val = max(1, int(val_frac * len(X)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[val_idx], y[val_idx]

        bs = batch_size or self.hp["batch_size"]
        S = self.hp["n_weight_samples_train"]
        n_train = len(Xtr)
        max_epochs = max_epochs or self.hp["n_epochs"]

        opt = tf.keras.optimizers.Adam(learning_rate=self.hp["learning_rate"])
        params = self._params

        @tf.function(reduce_retracing=True)
        def step(xb, yb, scale):
            with tf.GradientTape() as tape:
                logits, aux = self._forward(xb, S)
                ll = -tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.broadcast_to(yb, tf.shape(logits)), logits=logits)
                # mean over samples, sum over batch, rescaled to the full set
                exp_ll = tf.reduce_sum(tf.reduce_mean(ll, axis=0)) * scale
                loss = self._kl(aux) - exp_ll
            grads = tape.gradient(loss, params)
            opt.apply_gradients(zip(grads, params))
            return loss

        best, best_epoch, wait = np.inf, 0, 0
        best_state = None
        scale = tf.constant(float(n_train) / float(bs), tf.float32)

        for epoch in range(max_epochs):
            order = rng.permutation(n_train)
            for s in range(0, n_train - bs + 1, bs):
                b = order[s:s + bs]
                step(tf.constant(Xtr[b]), tf.constant(ytr[b]), scale)

            vnll = self._val_nll(Xva, yva)
            self.history_.append(vnll)
            if on_epoch is not None:
                on_epoch(self, epoch, vnll)
            if vnll < best - 1e-4:
                best, best_epoch, wait = vnll, epoch, 0
                best_state = [v.numpy().copy() for v in params]
            else:
                wait += 1
                if wait >= patience:
                    break
            if verbose:
                print(f"    epoch {epoch:4d}  val NLL {vnll:.4f}")

        if best_state is not None:
            for v, val in zip(params, best_state):
                v.assign(val)
        self.epochs_run_ = epoch + 1
        self.best_epoch_ = best_epoch
        return self

    def _val_nll(self, X, y, n_samples=32):
        p = self.predict_proba(X, n_samples=n_samples)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # -- prediction ---------------------------------------------------------
    def predict_proba(self, X, n_samples=None, chunk=4096):
        """Posterior predictive P(deceased), averaged over weight samples."""
        S = n_samples or self.hp["n_weight_samples_test"]
        X = np.asarray(X, dtype=np.float32)
        out = []
        for s in range(0, len(X), chunk):
            xb = tf.constant(X[s:s + chunk])
            logits, _ = self._forward(xb, S)
            out.append(tf.reduce_mean(tf.sigmoid(logits), axis=0).numpy())
        return np.concatenate(out)

    # -- interpretability ---------------------------------------------------
    def feature_relevance(self):
        """
        Figure 2's quantity: the norm of the posterior-mean weights leaving
        each input feature.  Near-zero means the model considers the feature
        irrelevant.
        """
        base = self.W1 if self.kind == "bnn" else self.W2
        mu = base.mu.numpy()                       # (F, h) or (F, 1)
        if self.prior == "horseshoe":
            tau = self.tau.mean.numpy()            # (F,)
            lam = float(self.lam.mean.numpy())
            mu = mu * (tau * lam)[:, None]
        return np.linalg.norm(mu, axis=1)


def build(name, n_features, seed=0, hp=None):
    """Factory keyed by the paper's model names."""
    spec = {
        "LinearGaussian":  ("linear", "gaussian"),
        "GaussianBNN":     ("bnn",    "gaussian"),
        "LinearHorseshoe": ("linear", "horseshoe"),
        "HorseshoeBNN":    ("bnn",    "horseshoe"),
    }[name]
    return BayesianModel(n_features, kind=spec[0], prior=spec[1],
                         hp=hp, seed=seed)
