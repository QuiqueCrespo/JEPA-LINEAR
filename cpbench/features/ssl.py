"""Self-supervised feature learners: LeJEPA variants (incl. the switching-linear-predictor jepa_mix), AE, (beta-)VAE, their linear-latent-predictor versions, CPC,
TS2Vec (reference-faithful) and TS2Vec-lite.  Each is a TorchFeatureLearner whose build() returns the network from
cpbench/models/nets.py; training, embedding and saving are shared."""
from cpbench.models import nets
from .base import register_learner, learner_preset
from .torch import TorchFeatureLearner


@register_learner("jepa")
class JEPA(TorchFeatureLearner):
    """Linear/MLP-predictor LeJEPA with SIGReg.  design: shared | separate (shared backbone, own final linear layer)
    | separate_full (two encoders)."""
    defaults = dict(design="shared", predictor="linear", lam=0.5)

    def build(self, D, W):
        return nets.JEPA(D, W, **self.net_hp())

    def views(self, windows):
        """Separate designs also expose the target encoder's features."""
        return {} if self.design == "shared" else {"target": self._encode(windows, self.net.encode_target)}


@register_learner("jepa_mix")
class MixtureJEPA(TorchFeatureLearner):
    """LeJEPA with a switching linear predictor sum_k w_k(window) A_k z_t; the features are the mixing weights w
    (views: "state" = z_t, "logits" = gate logits).  experts = number of matrices (a model hyperparameter).
    The state dimension d matters: SIGReg makes every state dimension unit-variance, so a large d made mostly of
    unpredictable dimensions leaves the gate almost no gradient (rotation data, D=64, 2000 steps: d=124 -> uniform
    weights, d=16 -> 0.60, d=4 -> 0.75 GMM accuracy on the weights).  Sweep d."""
    defaults = dict(d=4, experts=8, state_window=1, temperature=1.0, lam=0.5, balance=0.1, sharpness=0.05)

    def build(self, D, W):
        return nets.MixtureJEPA(D, W, **self.net_hp())

    def views(self, windows):
        return {"state": self._encode(windows, self.net.encode_state),
                "logits": self._encode(windows, self.net.gate)}


@register_learner("ae")
class AE(TorchFeatureLearner):
    def build(self, D, W):
        return nets.AE(D, W, **self.net_hp())


@register_learner("vae")
class VAE(TorchFeatureLearner):
    defaults = dict(beta=1.0)

    def build(self, D, W):
        return nets.VAE(D, W, **self.net_hp())


@register_learner("ae_pred")
class PredAE(TorchFeatureLearner):
    """Autoencoder with a linear predictor of the target window's latent from the context window's (weight mu)."""
    defaults = dict(mu=1.0)

    def build(self, D, W):
        return nets.PredAE(D, W, **self.net_hp())


@register_learner("vae_pred")
class PredVAE(TorchFeatureLearner):
    """beta-VAE with a linear predictor of the target posterior mean from a context posterior sample (weight mu)."""
    defaults = dict(beta=1.0, mu=1.0)

    def build(self, D, W):
        return nets.PredVAE(D, W, **self.net_hp())


@register_learner("cpc")
class CPC(TorchFeatureLearner):
    defaults = dict(temperature=0.1)

    def build(self, D, W):
        return nets.CPC(D, W, **self.net_hp())


@register_learner("ts2vec")
class TS2Vec(TorchFeatureLearner):
    """TS2Vec following the reference implementation (overlapping crops, latent timestamp masking, dilated CNN)."""
    defaults = dict(hidden=64, depth=10, mask_p=0.5, alpha=0.5, temporal_unit=0)

    def build(self, D, W):
        return nets.TS2Vec(D, W, **self.net_hp())


@register_learner("ts2vec_lite")
class TS2VecLite(TorchFeatureLearner):
    """The earlier simplified TS2Vec (two masked + jittered views of the same window, no crops): an ablation.  Results
    labelled `ts2vec` in runs before 2026-10-15 are this model."""
    defaults = dict(mask_p=0.5, jitter=0.1)

    def build(self, D, W):
        return nets.TS2VecLite(D, W, **self.net_hp())


learner_preset("jepa_shared", "jepa", design="shared", predictor="linear")
learner_preset("jepa_separate", "jepa", design="separate", predictor="linear")
learner_preset("jepa_separate_full", "jepa", design="separate_full", predictor="linear")
learner_preset("jepa_mlp_shared", "jepa", design="shared", predictor="mlp")
learner_preset("jepa_mlp_separate", "jepa", design="separate", predictor="mlp")
learner_preset("bvae4", "vae", beta=4.0)
