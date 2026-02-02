# kf.py
import jax
import chex
import jax.numpy as jnp
from functools import partial

from base import Rebayes  # <-- si usas mini-paquete: from rebayes.base import Rebayes


@chex.dataclass
class KFBel:
    mean: chex.Array            # (state_dim,)
    cov: chex.Array             # (state_dim, state_dim)
    obs_noise_var: float = 1.0  # para compatibilidad con Rebayes.obs_cov() si se usa
    t: int = 0


class KalmanFilter(Rebayes):
    def __init__(
        self,
        transition_matrix,
        system_noise,
        observation_noise,
        observation_matrix=None,
    ):
        # Nota: NO llamamos super().__init__ porque tu Rebayes.__init__ pide funciones
        # de emisión/dist que este KF clásico no necesita para el ejemplo.
        self.transition_matrix = transition_matrix
        self.observation_matrix = observation_matrix
        self.system_noise = system_noise
        self.observation_noise = observation_noise

    # ---------- helpers (permiten matrices/ruidos constantes o dependientes de t) ----------
    def get_trans_mat_of(self, t: int):
        return self.transition_matrix(t) if callable(self.transition_matrix) else self.transition_matrix

    def get_obs_mat_of(self, t: int):
        if self.observation_matrix is None:
            return None
        return self.observation_matrix(t) if callable(self.observation_matrix) else self.observation_matrix

    def get_system_noise_of(self, t: int):
        return self.system_noise(t) if callable(self.system_noise) else self.system_noise

    def get_observation_noise_of(self, t: int):
        R = self.observation_noise(t) if callable(self.observation_noise) else self.observation_noise
        return jnp.atleast_2d(R)

    # ---------- required by Rebayes (abstract methods) ----------
    def init_bel(self, initial_mean, initial_covariance, Xinit=None, Yinit=None) -> KFBel:
        """Inicializa la creencia. initial_covariance puede ser escalar, vector (diag) o matriz."""
        mean0 = jnp.asarray(initial_mean)

        cov_in = jnp.asarray(initial_covariance)
        if cov_in.ndim == 0:
            cov0 = jnp.eye(mean0.shape[0]) * cov_in
        elif cov_in.ndim == 1:
            cov0 = jnp.diag(cov_in)
        else:
            cov0 = cov_in

        # Por compatibilidad (solo se usa si adaptive_emission_covariance=True en otros métodos)
        R0 = self.get_observation_noise_of(0)
        obs_noise_var = float(jnp.squeeze(jnp.diag(R0))[0]) if R0.size else 1.0

        return KFBel(mean=mean0, cov=cov0, obs_noise_var=obs_noise_var, t=0)

    @partial(jax.jit, static_argnums=(0,))
    def predict_state(self, bel: KFBel) -> KFBel:
        A = self.get_trans_mat_of(bel.t)
        Q = self.get_system_noise_of(bel.t)

        cov_pred = A @ bel.cov @ A.T + Q
        mean_pred = A @ bel.mean

        return bel.replace(mean=mean_pred, cov=cov_pred)

    @partial(jax.jit, static_argnums=(0,))
    def predict_obs(self, bel: KFBel, X) -> chex.Array:
        """
        Predicción de observación.
        - Si observation_matrix está definido: y = C @ state
        - Si observation_matrix es None: interpreta X como vector de features y = <mean, X>
        """
        C = self.get_obs_mat_of(bel.t)
        if C is not None:
            C = jnp.atleast_2d(C)
            return (C @ bel.mean).squeeze()

        x = jnp.asarray(X)
        return jnp.einsum("i,...i->...", bel.mean, x)

    @partial(jax.jit, static_argnums=(0,))
    def update_state(self, bel: KFBel, X, y) -> KFBel:
        """
        Paso de corrección (Kalman update).
        - Si observation_matrix está definido, usa ese C
        - Si no, usa C = X (como fila: 1 x state_dim)
        """
        C = self.get_obs_mat_of(bel.t)
        if C is None:
            C = jnp.atleast_2d(jnp.asarray(X))
        else:
            C = jnp.atleast_2d(C)

        y = jnp.atleast_1d(y)
        R = self.get_observation_noise_of(bel.t)

        S = C @ bel.cov @ C.T + R                       # (obs_dim, obs_dim)
        K = jnp.linalg.solve(S, C @ bel.cov).T          # (state_dim, obs_dim)

        pred_obs = jnp.atleast_1d(C @ bel.mean)         # (obs_dim,)
        innovation = y - pred_obs                       # (obs_dim,)

        mean = bel.mean + K @ innovation                # (state_dim,)

        # Joseph form (más estable)
        I = jnp.eye(bel.mean.shape[0])
        tmp = I - K @ C
        cov = tmp @ bel.cov @ tmp.T + K @ R @ K.T

        return bel.replace(mean=mean, cov=cov, t=bel.t + 1)

    @partial(jax.jit, static_argnums=(0, 3))
    def sample_state(self, bel: KFBel, key, n_samples: int = 100, temperature: float = 1.0):
        """Muestrea estados desde N(mean, temperature * cov)."""
        cov = bel.cov * temperature
        return jax.random.multivariate_normal(
            key,
            bel.mean,
            cov,
            shape=(n_samples,),
        )

    # (Opcional pero útil) covarianza predictiva de la observación
    @partial(jax.jit, static_argnums=(0,))
    def predict_obs_cov(self, bel: KFBel, X, aleatoric_factor: float = 1.0, apply_fn=None):
        C = self.get_obs_mat_of(bel.t)
        if C is None:
            C = jnp.atleast_2d(jnp.asarray(X))
        else:
            C = jnp.atleast_2d(C)

        R = self.get_observation_noise_of(bel.t) * aleatoric_factor
        return C @ bel.cov @ C.T + R
