"""Composants liés à la physique du bras articulé."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple


def clamp(value: float, vmin: float, vmax: float) -> float:
    """Clamp a value between the provided bounds."""
    return max(vmin, min(vmax, value))


@dataclass
class ArmParameters:
    """Paramètres utilisateur pour le bras et la trajectoire."""

    length: float = 0.7
    mass_arm: float = 1.2
    mass_load: float = 0.4
    gravity: float = 9.81
    start_angle_deg: float = -30.0
    target_angle_deg: float = 75.0
    max_torque: float = 18.0
    max_velocity: float = 6.0
    max_acceleration: float = 20.0
    gravity_compensation: bool = True
    trajectory_mode: str = "fixed_duration"  # ou "time_optimal", "square_accel", "manual"
    target_duration: float = 3.0
    manual_input_mode: str = "acceleration"  # ou "torque"
    manual_profile_dt: float = 0.1
    manual_profile_values: str = "5, -5"
    auto_limits: bool = True

    def inertia(self) -> float:
        i_arm = (1.0 / 3.0) * self.mass_arm * self.length ** 2
        i_load = self.mass_load * self.length ** 2
        return i_arm + i_load

    def gravity_term(self) -> float:
        return (
            self.mass_arm * self.gravity * self.length / 2.0
            + self.mass_load * self.gravity * self.length
        )

    def update_auto_limits(self):
        if not self.auto_limits:
            return
        duration = max(0.1, float(self.target_duration))
        theta0 = math.radians(self.start_angle_deg)
        thetaf = math.radians(self.target_angle_deg)
        delta = abs(thetaf - theta0)
        if delta < 1e-4:
            delta = 1e-4
        self.max_velocity = max(0.1, 2.0 * delta / duration)
        self.max_acceleration = max(0.1, 4.0 * delta / (duration ** 2))
        inertia = self.inertia()
        torque_required = inertia * self.max_acceleration + abs(self.gravity_term())
        self.max_torque = max(1.0, torque_required * 1.1)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        return data

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "ArmParameters":
        params = ArmParameters()
        for key, value in data.items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params


@dataclass
class ArmState:
    theta: float = 0.0
    omega: float = 0.0
    alpha: float = 0.0


class ArmModel:
    """Modèle physique élémentaire du bras rigide."""

    def __init__(self, params: ArmParameters):
        self.params = params

    def gravity_torque(self, theta: float) -> float:
        """Compute the gravitational torque applied on the arm.

        The reference configuration (``theta = 0``) corresponds to a horizontal
        arm that is subject to the full gravitational torque. When the arm is
        vertical the gravity torque should cancel out. Using the cosine ensures
        a non-zero (negative) torque at ``theta = 0`` that goes to zero as the
        arm approaches the vertical position (``theta = pi/2``).
        """

        return -self.params.gravity_term() * math.cos(theta)

    def required_torque(self, alpha: float, theta: float) -> Tuple[float, float]:
        """Retourne le couple moteur nécessaire pour imposer une accélération donnée.

        Le modèle est volontairement simplifié : on considère uniquement l'inertie
        totale du bras et le couple gravitationnel instantané. Aucun frottement ni
        autre dissipation n'est pris en compte. Le résultat correspond donc au
        couple à appliquer pour obtenir exactement ``alpha`` en présence de la
        gravité actuelle.
        """

        tau_g = self.gravity_torque(theta)
        torque = self.params.inertia() * alpha - tau_g
        return torque, tau_g

    def compute_alpha(self, couple_moteur: float, theta: float, omega: float) -> Tuple[float, float]:
        tau_g = self.gravity_torque(theta)
        alpha = (couple_moteur + tau_g) / self.params.inertia()
        return alpha, tau_g
