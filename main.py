"""Simulation interactive d'un bras rigide à un axe avec édition intégrée.

Le programme structure les responsabilités en quatre blocs principaux :
- Modèle physique : inertie, calcul des couples et intégration.
- Planification : génération des trajectoires respectant les contraintes.
- Interface : widgets pygame pour boutons, champs texte, cases à cocher, listes.
- Affichage : animation du bras, panneaux d'état et de courbes en temps réel.

Le script peut être lancé directement :
    python main.py

Pour les environnements sans affichage (tests automatiques) :
    python main.py --headless-test
    (la simulation tourne quelques secondes et s'arrête)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pygame

# ---------------------------------------------------------------------------
# Constantes globales d'interface
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS_TARGET = 60
PHYSICS_HZ = 600
SIM_PANEL_WIDTH = int(WINDOW_WIDTH * 0.48)
PLOT_PANEL_WIDTH = WINDOW_WIDTH - SIM_PANEL_WIDTH
TOOLBAR_HEIGHT = 60
PANEL_MARGIN = 10
FONT_NAME = "freesansbold.ttf"
CONFIG_PATH = "config.json"
EXPORT_PATH = "export.csv"

BACKGROUND_COLOR = (15, 18, 26)
PANEL_BG = (28, 31, 40)
TOOLBAR_BG = (42, 45, 60)
TEXT_COLOR = (230, 234, 245)
HIGHLIGHT_COLOR = (70, 140, 255)
ERROR_COLOR = (255, 90, 90)
SUCCESS_COLOR = (120, 220, 120)

ARM_COLOR = (200, 210, 255)
LOAD_COLOR = (255, 200, 120)
PIVOT_COLOR = (120, 140, 180)
TRACE_COLORS = {
    "angle": (255, 200, 0),
    "omega": (0, 220, 255),
    "alpha": (180, 255, 140),
    "tau_motor": (255, 120, 120),
    "tau_gravity": (255, 200, 180),
    "tau_damping": (150, 200, 120),
    "power": (220, 180, 255),
    "x": (220, 220, 220),
    "y": (160, 220, 200),
    "z": (140, 160, 220),
    "vx": (200, 150, 255),
    "vy": (150, 200, 255),
    "ax": (255, 180, 220),
    "ay": (180, 220, 255),
}

# ---------------------------------------------------------------------------
# Bloc : Modèle physique
# ---------------------------------------------------------------------------


def clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


@dataclass
class ArmParameters:
    """Paramètres utilisateur pour le bras et la trajectoire."""

    length: float = 0.7
    mass_arm: float = 1.2
    mass_load: float = 0.4
    gravity: float = 9.81
    damping: float = 0.12
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
        return -self.params.gravity_term() * math.sin(theta)

    def damping_torque(self, omega: float) -> float:
        return -self.params.damping * omega

    def compute_alpha(self, tau_motor: float, theta: float, omega: float) -> Tuple[float, float, float]:
        tau_g = self.gravity_torque(theta)
        tau_d = self.damping_torque(omega)
        alpha = (tau_motor + tau_g + tau_d) / self.params.inertia()
        return alpha, tau_g, tau_d


# ---------------------------------------------------------------------------
# Bloc : planification de trajectoire
# ---------------------------------------------------------------------------


class TrajectoryError(Exception):
    pass


@dataclass
class TrajectorySample:
    theta: float
    omega: float
    alpha: float


@dataclass
class TrajectoryProfile:
    """Représente une trajectoire uniaxe à profil trapézoïdal/triangulaire."""

    theta0: float
    thetaf: float
    t_acc: float
    t_flat: float
    duration: float
    a_nominal: float
    scale: float = 1.0

    def sample(self, t: float) -> TrajectorySample:
        if t <= 0.0:
            return TrajectorySample(self.theta0, 0.0, 0.0)
        if t >= self.duration:
            return TrajectorySample(self.thetaf, 0.0, 0.0)

        u = t / self.scale
        a = self.a_nominal
        dir_sign = 1.0 if self.thetaf >= self.theta0 else -1.0
        t1 = self.t_acc
        t2 = self.t_acc + self.t_flat
        v_peak = a * t1

        if u < t1:
            theta_min = 0.5 * a * u ** 2
            omega_min = a * u
            alpha_min = a
        elif u < t2:
            tau = u - t1
            theta_min = 0.5 * a * t1 ** 2 + v_peak * tau
            omega_min = v_peak
            alpha_min = 0.0
        else:
            tau = u - t2
            u_dec = self.duration / self.scale - u
            # symétrie : même durée d'accélération
            theta_min_total = 0.5 * a * t1 ** 2 + v_peak * self.t_flat + 0.5 * a * t1 ** 2
            theta_min = theta_min_total - (0.5 * a * u_dec ** 2)
            omega_min = v_peak - a * tau
            alpha_min = -a

        theta = self.theta0 + dir_sign * theta_min
        omega = dir_sign * (omega_min / self.scale)
        alpha = dir_sign * (alpha_min / (self.scale ** 2))
        return TrajectorySample(theta, omega, alpha)


@dataclass
class SquareAccelProfile:
    """Profil avec accélération carrée : +K puis -K."""

    theta0: float
    thetaf: float
    duration: float
    accel: float

    def sample(self, t: float) -> TrajectorySample:
        sign = 1.0 if self.thetaf >= self.theta0 else -1.0
        accel = self.accel * sign
        if t <= 0.0:
            return TrajectorySample(self.theta0, 0.0, accel)
        if t >= self.duration:
            return TrajectorySample(self.thetaf, 0.0, -accel)

        half = self.duration / 2.0
        if t <= half:
            omega = accel * t
            theta = self.theta0 + 0.5 * accel * t ** 2
            alpha = accel
        else:
            tau = t - half
            omega_mid = accel * half
            theta_mid = self.theta0 + 0.5 * accel * half ** 2
            omega = omega_mid - accel * tau
            theta = theta_mid + omega_mid * tau - 0.5 * accel * tau ** 2
            alpha = -accel
        return TrajectorySample(theta, omega, alpha)


@dataclass
class ManualProfile:
    """Profil défini par l'utilisateur à partir d'accélérations ou de couples."""

    theta0: float
    duration: float
    dt: float
    mode: str
    times: List[float]
    theta_values: List[float]
    omega_values: List[float]
    alpha_values: List[float]
    torque_values: List[float]

    def sample(self, t: float) -> TrajectorySample:
        if t <= 0.0:
            return TrajectorySample(self.theta_values[0], self.omega_values[0], self.alpha_values[0])
        if t >= self.duration:
            return TrajectorySample(self.theta_values[-1], self.omega_values[-1], self.alpha_values[-1])

        index = min(int(t // self.dt), len(self.alpha_values) - 1)
        local_t = t - self.times[index]
        theta0 = self.theta_values[index]
        omega0 = self.omega_values[index]
        alpha = self.alpha_values[index]
        theta = theta0 + omega0 * local_t + 0.5 * alpha * local_t ** 2
        omega = omega0 + alpha * local_t
        return TrajectorySample(theta, omega, alpha)

    def torque_at(self, t: float) -> float:
        if self.mode != "torque":
            return 0.0
        if t <= 0.0:
            return self.torque_values[0]
        if t >= self.duration:
            return self.torque_values[-1]
        index = min(int(t // self.dt), len(self.torque_values) - 1)
        return self.torque_values[index]


class TrajectoryPlanner:
    """Planifie une trajectoire respectant les contraintes couple/vitesse."""

    def __init__(self, params: ArmParameters):
        self.params = params

    def compute_limits(self) -> Tuple[float, float]:
        inertia = self.params.inertia()
        if inertia <= 0:
            raise TrajectoryError("Inertie non positive : vérifier les masses et la longueur")
        torque_limit = max(1e-6, abs(self.params.max_torque))
        accel_from_torque = torque_limit / inertia
        if self.params.max_acceleration > 0:
            accel_limit = min(accel_from_torque, self.params.max_acceleration)
        else:
            accel_limit = accel_from_torque
        if accel_limit <= 0:
            raise TrajectoryError("Limite d'accélération nulle ou négative")
        if self.params.max_velocity <= 0:
            raise TrajectoryError("Vitesse maximale doit être > 0")
        return accel_limit, self.params.max_velocity

    def plan_minimal_profile(self) -> Tuple[TrajectoryProfile, float]:
        a_limit, v_limit = self.compute_limits()
        theta0 = math.radians(self.params.start_angle_deg)
        thetaf = math.radians(self.params.target_angle_deg)
        delta = thetaf - theta0
        distance = abs(delta)
        if distance < 1e-6:
            profile = TrajectoryProfile(theta0, thetaf, 0.0, 0.0, 0.0, a_limit, 1.0)
            return profile, 0.0

        t_acc_limit = v_limit / a_limit
        d_acc_limit = 0.5 * a_limit * t_acc_limit ** 2

        if distance <= 2 * d_acc_limit + 1e-9:
            # profil triangulaire
            t_acc = math.sqrt(distance / a_limit)
            duration = 2.0 * t_acc
            profile = TrajectoryProfile(theta0, thetaf, t_acc, 0.0, duration, a_limit, 1.0)
            return profile, duration
        else:
            # profil trapézoïdal
            t_acc = t_acc_limit
            d_acc = d_acc_limit
            d_flat = distance - 2 * d_acc
            t_flat = d_flat / v_limit
            duration = 2 * t_acc + t_flat
            profile = TrajectoryProfile(theta0, thetaf, t_acc, t_flat, duration, a_limit, 1.0)
            return profile, duration

    def plan(self) -> TrajectoryProfile:
        self.params.update_auto_limits()
        profile, minimal_duration = self.plan_minimal_profile()
        mode = self.params.trajectory_mode
        if minimal_duration == 0:
            profile.duration = 0.0
            profile.scale = 1.0
            return profile

        if mode == "time_optimal":
            profile.duration = minimal_duration
            profile.scale = 1.0
            return profile

        if mode == "fixed_duration":
            target = max(0.01, self.params.target_duration)
            if target < minimal_duration - 1e-6:
                self.params.target_duration = minimal_duration
                raise TrajectoryError(
                    "Durée demandée {:.2f}s trop courte. Minimum possible : {:.2f}s (valeur proposée).".format(
                        target, minimal_duration
                    )
                )
            scale = target / minimal_duration
            profile.scale = scale
            profile.duration = target
            return profile

        if mode == "square_accel":
            target = max(0.01, self.params.target_duration)
            theta0 = math.radians(self.params.start_angle_deg)
            thetaf = math.radians(self.params.target_angle_deg)
            delta = thetaf - theta0
            if abs(delta) < 1e-9:
                return SquareAccelProfile(theta0, thetaf, target, 0.0)
            accel_needed = 4.0 * abs(delta) / (target ** 2)
            accel_limit, _ = self.compute_limits()
            if accel_needed > accel_limit + 1e-9:
                raise TrajectoryError(
                    "Accélération carrée requise {:.2f} rad/s² dépasse la limite {:.2f} rad/s²".format(
                        accel_needed, accel_limit
                    )
                )
            return SquareAccelProfile(theta0, thetaf, target, accel_needed)

        if mode == "manual":
            return self.build_manual_profile()

        raise TrajectoryError(f"Mode de trajectoire inconnu : {mode}")

    def build_manual_profile(self) -> ManualProfile:
        dt = max(1e-3, float(self.params.manual_profile_dt))
        raw_values = [v.strip() for v in self.params.manual_profile_values.split(",") if v.strip()]
        if not raw_values:
            raise TrajectoryError("Aucune valeur saisie pour le profil manuel")
        try:
            values = [float(v) for v in raw_values]
        except ValueError:
            raise TrajectoryError("Valeurs du profil manuel invalides")

        theta0 = math.radians(self.params.start_angle_deg)
        model = ArmModel(self.params)
        inertia = self.params.inertia()
        theta = theta0
        omega = 0.0
        t = 0.0
        times = [0.0]
        theta_values = [theta0]
        omega_values = [0.0]
        alpha_values: List[float] = []
        torque_values: List[float] = []

        for value in values:
            if self.params.manual_input_mode == "torque":
                tau_motor = clamp(value, -abs(self.params.max_torque), abs(self.params.max_torque))
                tau_g = model.gravity_torque(theta)
                tau_d = model.damping_torque(omega)
                alpha = (tau_motor + tau_g + tau_d) / inertia
            else:
                alpha = value
                tau_motor = inertia * alpha - model.gravity_torque(theta) - model.damping_torque(omega)
                tau_motor = clamp(tau_motor, -abs(self.params.max_torque), abs(self.params.max_torque))
            alpha_values.append(alpha)
            torque_values.append(tau_motor)
            theta += omega * dt + 0.5 * alpha * dt * dt
            omega += alpha * dt
            t += dt
            times.append(t)
            theta_values.append(theta)
            omega_values.append(omega)

        duration = times[-1]
        if duration <= 0.0:
            raise TrajectoryError("Durée totale du profil manuel nulle")

        return ManualProfile(
            theta0=theta0,
            duration=duration,
            dt=dt,
            mode=self.params.manual_input_mode,
            times=times,
            theta_values=theta_values,
            omega_values=omega_values,
            alpha_values=alpha_values,
            torque_values=torque_values,
        )


# ---------------------------------------------------------------------------
# Bloc : Simulation et enregistrement des données
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    t: float
    theta: float
    omega: float
    alpha: float
    vx: float
    vy: float
    ax: float
    ay: float
    theta_ref: float
    omega_ref: float
    alpha_ref: float
    tau_motor: float
    tau_gravity: float
    tau_damping: float
    power: float
    x: float
    y: float
    z: float


class ArmSimulation:
    """Gère l'intégration physique et l'exécution d'une trajectoire."""

    def __init__(self, params: ArmParameters):
        self.params = params
        self.params.update_auto_limits()
        self.model = ArmModel(params)
        self.state = ArmState(theta=math.radians(params.start_angle_deg))
        self.running = False
        self.paused = False
        self.time = 0.0
        self.trajectory: TrajectoryProfile = TrajectoryProfile(
            self.state.theta, self.state.theta, 0.0, 0.0, 0.0, 1.0, 1.0
        )
        self.last_tau_motor = 0.0
        self.last_tau_gravity = 0.0
        self.last_tau_damping = 0.0
        self.last_power = 0.0
        self.log: List[LogEntry] = []
        self.progress = 0.0
        self.freeze_logs = False
        self.kp = 60.0
        self.kd = 8.0

    def update_parameters(self, params: ArmParameters):
        self.params = params
        self.params.update_auto_limits()
        self.model = ArmModel(params)

    def set_trajectory(self, trajectory: TrajectoryProfile):
        self.trajectory = trajectory
        self.reset_state()

    def reset_state(self):
        self.state = ArmState(theta=math.radians(self.params.start_angle_deg))
        self.time = 0.0
        self.running = False
        self.paused = False
        self.progress = 0.0
        self.last_tau_motor = 0.0
        self.last_tau_damping = 0.0
        self.last_tau_gravity = self.model.gravity_torque(self.state.theta)
        self.last_power = 0.0
        self.log.clear()

    def start(self):
        self.running = True
        self.paused = False
        self.time = 0.0
        self.progress = 0.0
        self.log.clear()

    def toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused

    def physics_step(self, dt: float):
        if not self.running or self.paused:
            return

        self.time += dt
        duration = max(self.trajectory.duration, 1e-6)
        self.progress = clamp(self.time / duration, 0.0, 1.0)
        sample = self.trajectory.sample(self.time)

        theta_ref = sample.theta
        omega_ref = sample.omega
        alpha_ref = sample.alpha

        # Avance de modèle + correcteur proportionnel dérivé simple
        inertia = self.params.inertia()
        manual_torque = None
        if isinstance(self.trajectory, ManualProfile) and self.trajectory.mode == "torque":
            manual_torque = self.trajectory.torque_at(self.time)

        if manual_torque is None:
            tau_model = inertia * alpha_ref
            if self.params.gravity_compensation:
                tau_model -= self.model.gravity_torque(theta_ref)
            tau_model -= self.model.damping_torque(omega_ref)

            error_theta = theta_ref - self.state.theta
            error_omega = omega_ref - self.state.omega
            tau_pd = self.kp * error_theta + self.kd * error_omega
            tau_command = tau_model + tau_pd
            tau_command = clamp(tau_command, -abs(self.params.max_torque), abs(self.params.max_torque))
        else:
            tau_command = manual_torque

        alpha, tau_g, tau_d = self.model.compute_alpha(tau_command, self.state.theta, self.state.omega)
        self.state.omega += alpha * dt
        self.state.omega = clamp(self.state.omega, -abs(self.params.max_velocity) * 1.5, abs(self.params.max_velocity) * 1.5)
        self.state.theta += self.state.omega * dt
        self.state.alpha = alpha

        self.last_tau_motor = tau_command
        self.last_tau_gravity = tau_g
        self.last_tau_damping = tau_d
        self.last_power = tau_command * self.state.omega

        end_point = self.tip_position()
        x_tip, y_tip = end_point
        length = self.params.length
        omega = self.state.omega
        alpha = self.state.alpha
        vx = -length * omega * math.sin(self.state.theta)
        vy = length * omega * math.cos(self.state.theta)
        ax_lin = -length * (
            math.sin(self.state.theta) * alpha + math.cos(self.state.theta) * omega ** 2
        )
        ay_lin = length * (
            math.cos(self.state.theta) * alpha - math.sin(self.state.theta) * omega ** 2
        )
        if not self.freeze_logs:
            self.log.append(
                LogEntry(
                    t=self.time,
                    theta=self.state.theta,
                    omega=self.state.omega,
                    alpha=self.state.alpha,
                    vx=vx,
                    vy=vy,
                    ax=ax_lin,
                    ay=ay_lin,
                    theta_ref=theta_ref,
                    omega_ref=omega_ref,
                    alpha_ref=alpha_ref,
                    tau_motor=tau_command,
                    tau_gravity=tau_g,
                    tau_damping=tau_d,
                    power=self.last_power,
                    x=x_tip,
                    y=y_tip,
                    z=0.0,
                )
            )

        if self.time >= self.trajectory.duration:
            self.running = False
            self.progress = 1.0

    def tip_position(self) -> Tuple[float, float]:
        x = self.params.length * math.cos(self.state.theta)
        y = self.params.length * math.sin(self.state.theta)
        return x, y


# ---------------------------------------------------------------------------
# Bloc : Interface utilisateur (widgets simplifiés)
# ---------------------------------------------------------------------------


class Widget:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.visible = True

    def handle_event(self, event: pygame.event.Event):
        pass

    def draw(self, surface: pygame.Surface):
        pass


class Button(Widget):
    def __init__(self, rect: pygame.Rect, label: str, callback: Callable[[], None], font: pygame.font.Font):
        super().__init__(rect)
        self.label = label
        self.callback = callback
        self.font = font
        self.hover = False

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()

    def draw(self, surface: pygame.Surface):
        color = HIGHLIGHT_COLOR if self.hover else (90, 95, 120)
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        text_surf = self.font.render(self.label, True, (0, 0, 0))
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))


class Checkbox(Widget):
    def __init__(self, rect: pygame.Rect, label: str, value: bool, font: pygame.font.Font):
        super().__init__(rect)
        self.label = label
        self.value = value
        self.font = font

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value

    def draw(self, surface: pygame.Surface):
        box_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.height, self.rect.height)
        pygame.draw.rect(surface, (200, 200, 210), box_rect, width=2, border_radius=4)
        if self.value:
            inner = box_rect.inflate(-6, -6)
            pygame.draw.rect(surface, (220, 220, 240), inner, border_radius=3)
        label_surf = self.font.render(self.label, True, TEXT_COLOR)
        surface.blit(label_surf, (self.rect.x + self.rect.height + 8, self.rect.y))


class Dropdown(Widget):
    def __init__(self, rect: pygame.Rect, options: Sequence[str], value: str, font: pygame.font.Font):
        super().__init__(rect)
        self.options = list(options)
        self.value = value
        self.open = False
        self.font = font

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.open:
                for i, option in enumerate(self.options):
                    opt_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width, self.rect.height)
                    if opt_rect.collidepoint(event.pos):
                        self.value = option
                        self.open = False
                        return
                self.open = False
            elif self.rect.collidepoint(event.pos):
                self.open = True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button != 1:
            self.open = False

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, (80, 85, 100), self.rect, border_radius=4)
        text = self.font.render(self.value, True, TEXT_COLOR)
        surface.blit(text, text.get_rect(center=self.rect.center))
        if self.open:
            for i, option in enumerate(self.options):
                opt_rect = pygame.Rect(self.rect.x, self.rect.y + (i + 1) * self.rect.height, self.rect.width, self.rect.height)
                pygame.draw.rect(surface, (60, 60, 70), opt_rect, border_radius=4)
                txt = self.font.render(option, True, TEXT_COLOR)
                surface.blit(txt, txt.get_rect(center=opt_rect.center))


class TextInput(Widget):
    def __init__(self, rect: pygame.Rect, text: str, font: pygame.font.Font):
        super().__init__(rect)
        self.text = text
        self.font = font
        self.active = False

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if len(event.unicode) == 1 and (event.unicode.isprintable()):
                    self.text += event.unicode

    def draw(self, surface: pygame.Surface):
        color = HIGHLIGHT_COLOR if self.active else (90, 95, 120)
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        txt = self.font.render(self.text, True, (0, 0, 0))
        surface.blit(txt, txt.get_rect(center=self.rect.center))


# ---------------------------------------------------------------------------
# Bloc : éditeur de paramètres et panneaux
# ---------------------------------------------------------------------------


class ParameterEditor:
    def __init__(self, font: pygame.font.Font, params: ArmParameters, area: pygame.Rect):
        self.font = font
        self.params = params
        self.area = area
        self.widgets: Dict[str, Widget] = {}
        self.error_message = ""
        self.success_message = ""
        self._build_widgets()

    def _build_widgets(self):
        padding = 6
        line_height = 32
        labels = [
            ("Angle départ (deg)", "start_angle_deg"),
            ("Angle arrivée (deg)", "target_angle_deg"),
            ("Longueur (m)", "length"),
            ("Masse bras (kg)", "mass_arm"),
            ("Masse charge (kg)", "mass_load"),
            ("Gravité (m/s²)", "gravity"),
            ("Durée cible (s)", "target_duration"),
            ("Profil manuel dt (s)", "manual_profile_dt"),
            ("Profil manuel valeurs", "manual_profile_values"),
        ]
        self.labels = labels
        self.checkbox = Checkbox(
            pygame.Rect(self.area.x + 10, self.area.y + 10, 24, 24),
            "Compensation gravité",
            self.params.gravity_compensation,
            self.font,
        )
        self.dropdown = Dropdown(
            pygame.Rect(self.area.x + 10, self.area.y + 10 + 2 * 24, 230, 28),
            ["time_optimal", "fixed_duration", "square_accel", "manual"],
            self.params.trajectory_mode,
            self.font,
        )
        self.manual_mode_dropdown = Dropdown(
            pygame.Rect(self.area.x + 250, self.area.y + 10 + 2 * 24, 200, 28),
            ["acceleration", "torque"],
            self.params.manual_input_mode,
            self.font,
        )
        y = self.area.y + 10 + 3 * 28
        for label, key in labels:
            rect = pygame.Rect(self.area.x + 10, y, self.area.width - 20, line_height)
            text_input = TextInput(rect, f"{getattr(self.params, key)}", self.font)
            self.widgets[key] = text_input
            y += line_height + padding

    def update_from_params(self):
        self.checkbox.value = self.params.gravity_compensation
        self.dropdown.value = self.params.trajectory_mode
        self.manual_mode_dropdown.value = self.params.manual_input_mode
        for key, widget in self.widgets.items():
            if isinstance(widget, TextInput):
                widget.text = f"{getattr(self.params, key)}"

    def handle_event(self, event: pygame.event.Event):
        self.checkbox.handle_event(event)
        self.dropdown.handle_event(event)
        self.manual_mode_dropdown.handle_event(event)
        for widget in self.widgets.values():
            widget.handle_event(event)

    def apply_changes(self) -> bool:
        try:
            for key, widget in self.widgets.items():
                if isinstance(widget, TextInput):
                    value = widget.text.strip().replace(",", ".")
                    if value == "":
                        raise ValueError(f"Champ {key} vide")
                    current = getattr(self.params, key)
                    if isinstance(current, float):
                        setattr(self.params, key, float(value))
                    elif isinstance(current, (int,)):
                        setattr(self.params, key, int(float(value)))
                    else:
                        setattr(self.params, key, value)
            self.params.gravity_compensation = self.checkbox.value
            self.params.trajectory_mode = self.dropdown.value
            self.params.manual_input_mode = self.manual_mode_dropdown.value
            self.error_message = ""
            self.success_message = "Paramètres mis à jour"
            return True
        except ValueError as exc:
            self.error_message = str(exc)
            self.success_message = ""
            return False

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, PANEL_BG, self.area)
        self.checkbox.draw(surface)
        label_mode = self.font.render("Mode trajectoire", True, TEXT_COLOR)
        surface.blit(label_mode, (self.dropdown.rect.x, self.dropdown.rect.y - 20))
        self.dropdown.draw(surface)
        label_manual_mode = self.font.render("Entrée manuelle", True, TEXT_COLOR)
        surface.blit(label_manual_mode, (self.manual_mode_dropdown.rect.x, self.manual_mode_dropdown.rect.y - 20))
        self.manual_mode_dropdown.draw(surface)
        y = self.dropdown.rect.bottom + 10
        for label, key in self.labels:
            label_surface = self.font.render(label, True, TEXT_COLOR)
            surface.blit(label_surface, (self.area.x + 12, y))
            widget = self.widgets[key]
            widget.rect.y = y + 18
            widget.draw(surface)
            y = widget.rect.bottom + 6
        auto_text = self.font.render(
            "Limites de couple/vitesse/accélération calculées automatiquement.",
            True,
            TEXT_COLOR,
        )
        surface.blit(auto_text, (self.area.x + 12, y + 6))
        y += auto_text.get_height() + 10
        if self.error_message:
            err = self.font.render(self.error_message, True, ERROR_COLOR)
            surface.blit(err, (self.area.x + 10, self.area.bottom - 60))
        elif self.success_message:
            msg = self.font.render(self.success_message, True, SUCCESS_COLOR)
            surface.blit(msg, (self.area.x + 10, self.area.bottom - 60))


class PlotPanel:
    def __init__(self, area: pygame.Rect, font: pygame.font.Font):
        self.area = area
        self.font = font
        self.signals = [
            "angle",
            "omega",
            "alpha",
            "tau_motor",
            "tau_gravity",
            "tau_damping",
            "power",
            "x",
            "y",
            "vx",
            "vy",
            "ax",
            "ay",
            "z",
        ]
        self.selected: Dict[str, bool] = {
            name: name
            in ("angle", "omega", "alpha", "tau_motor", "x", "y", "vx", "vy", "ax", "ay")
            for name in self.signals
        }
        self.freeze = False
        self.max_points = 2000
        self.cached_data: Dict[str, List[float]] = {name: [] for name in self.signals}
        self.time_data: List[float] = []
        self.scroll_offset = 0.0
        self.checkbox_rects: Dict[str, pygame.Rect] = {}
        self.legend_height = 22
        self.num_ticks = 5

    def toggle_freeze(self):
        self.freeze = not self.freeze

    def reset(self):
        self.cached_data = {name: [] for name in self.signals}
        self.time_data = []

    def update(self, logs: List[LogEntry]):
        if self.freeze:
            return
        self.reset()
        for entry in logs[-self.max_points :]:
            self.time_data.append(entry.t)
            self.cached_data["angle"].append(entry.theta)
            self.cached_data["omega"].append(entry.omega)
            self.cached_data["alpha"].append(entry.alpha)
            self.cached_data["vx"].append(entry.vx)
            self.cached_data["vy"].append(entry.vy)
            self.cached_data["ax"].append(entry.ax)
            self.cached_data["ay"].append(entry.ay)
            self.cached_data["tau_motor"].append(entry.tau_motor)
            self.cached_data["tau_gravity"].append(entry.tau_gravity)
            self.cached_data["tau_damping"].append(entry.tau_damping)
            self.cached_data["power"].append(entry.power)
            self.cached_data["x"].append(entry.x)
            self.cached_data["y"].append(entry.y)
            self.cached_data["z"].append(entry.z)

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            for name, rect in self.checkbox_rects.items():
                if rect.collidepoint(pos):
                    self.selected[name] = not self.selected[name]

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, PANEL_BG, self.area)
        legend_area = pygame.Rect(self.area.x + 10, self.area.y + 10, self.area.width - 20, 20)
        # Draw checkboxes
        x = legend_area.x
        self.checkbox_rects.clear()
        for name in self.signals:
            checkbox = pygame.Rect(x, legend_area.y, 16, 16)
            pygame.draw.rect(surface, TRACE_COLORS[name], checkbox, width=2)
            if self.selected[name]:
                pygame.draw.rect(surface, TRACE_COLORS[name], checkbox.inflate(-4, -4))
            label = self.font.render(name, True, TEXT_COLOR)
            surface.blit(label, (checkbox.right + 4, checkbox.y - 2))
            x = checkbox.right + label.get_width() + 20
            self.checkbox_rects[name] = checkbox

        plot_rect = pygame.Rect(
            self.area.x + 10,
            legend_area.bottom + 10,
            self.area.width - 20,
            self.area.height - legend_area.height - 40,
        )
        pygame.draw.rect(surface, (20, 20, 28), plot_rect)
        pygame.draw.rect(surface, (80, 80, 90), plot_rect, width=1)

        if len(self.time_data) < 2:
            return

        t0 = self.time_data[0]
        t1 = self.time_data[-1]
        if t1 - t0 < 1e-6:
            return

        active_signals = [name for name in self.signals if self.selected.get(name, False) and self.cached_data[name]]
        if not active_signals:
            return

        global_min = min(min(self.cached_data[name]) for name in active_signals)
        global_max = max(max(self.cached_data[name]) for name in active_signals)
        if abs(global_max - global_min) < 1e-9:
            global_max += 1.0
            global_min -= 1.0
        padding = 0.05 * (global_max - global_min)
        y_min = global_min - padding
        y_max = global_max + padding
        scale_text = self.font.render(f"Échelle : [{y_min:.2f}, {y_max:.2f}]", True, TEXT_COLOR)
        surface.blit(scale_text, (plot_rect.x + 4, plot_rect.bottom + 6))

        for i in range(self.num_ticks + 1):
            frac = i / self.num_ticks
            y = plot_rect.bottom - frac * plot_rect.height
            value = y_min + frac * (y_max - y_min)
            pygame.draw.line(surface, (60, 60, 70), (plot_rect.x, y), (plot_rect.right, y), width=1)
            label = self.font.render(f"{value:.2f}", True, (150, 150, 160))
            surface.blit(label, (plot_rect.x - label.get_width() - 6, y - label.get_height() / 2))

        pygame.draw.line(surface, (120, 120, 140), (plot_rect.x, plot_rect.bottom), (plot_rect.x, plot_rect.y), width=1)

        legend_index = 0
        for name, values in self.cached_data.items():
            if not self.selected.get(name, False):
                continue
            color = TRACE_COLORS.get(name, (200, 200, 200))
            if not values:
                continue
            min_v = min(values)
            max_v = max(values)
            points = []
            for t, v in zip(self.time_data, values):
                px = plot_rect.x + (t - t0) / (t1 - t0) * plot_rect.width
                py = plot_rect.bottom - (v - y_min) / (y_max - y_min) * plot_rect.height
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(surface, color, False, points, 2)
            label = self.font.render(f"{name}: [{min_v:.2f}, {max_v:.2f}]", True, color)
            surface.blit(label, (plot_rect.x + 4, plot_rect.y + 4 + self.legend_height * legend_index))
            legend_index += 1


# ---------------------------------------------------------------------------
# Affichage de la simulation et HUD
# ---------------------------------------------------------------------------


class SimulationRenderer:
    def __init__(self, area: pygame.Rect, font: pygame.font.Font):
        self.area = area
        self.font = font
        self.pixels_per_meter = 250

    def draw(self, surface: pygame.Surface, sim: ArmSimulation):
        pygame.draw.rect(surface, PANEL_BG, self.area)
        center = (self.area.centerx, self.area.centery + 100)
        pygame.draw.circle(surface, PIVOT_COLOR, center, 12)
        length_px = sim.params.length * self.pixels_per_meter
        end_x = center[0] + length_px * math.cos(sim.state.theta)
        end_y = center[1] - length_px * math.sin(sim.state.theta)
        pygame.draw.line(surface, ARM_COLOR, center, (end_x, end_y), 8)
        pygame.draw.circle(surface, LOAD_COLOR, (int(end_x), int(end_y)), 18)

        # Reference arm (transparent) when running
        if sim.running:
            sample = sim.trajectory.sample(sim.time)
            ref_end_x = center[0] + length_px * math.cos(sample.theta)
            ref_end_y = center[1] - length_px * math.sin(sample.theta)
            pygame.draw.line(surface, (120, 120, 160), center, (ref_end_x, ref_end_y), 2)
            pygame.draw.circle(surface, (120, 120, 180), (int(ref_end_x), int(ref_end_y)), 10, width=2)

        info_x = self.area.x + 20
        info_y = self.area.y + 20
        lines = [
            f"Angle : {math.degrees(sim.state.theta):6.2f}°",
            f"Vitesse : {sim.state.omega:6.2f} rad/s",
            f"Accélération : {sim.state.alpha:6.2f} rad/s²",
            f"Couple moteur : {sim.last_tau_motor:6.2f} N·m",
            f"Couple gravité : {sim.last_tau_gravity:6.2f} N·m",
            f"Couple amort. : {sim.last_tau_damping:6.2f} N·m",
            f"Inertie équiv. : {sim.params.inertia():6.3f} kg·m²",
            f"Puissance : {sim.last_power:6.2f} W",
            f"Mode : {sim.params.trajectory_mode}",
            f"Compensation g : {'ON' if sim.params.gravity_compensation else 'OFF'}",
        ]
        for line in lines:
            text = self.font.render(line, True, TEXT_COLOR)
            surface.blit(text, (info_x, info_y))
            info_y += 22

        # Progress bar
        bar_rect = pygame.Rect(self.area.x + 40, self.area.bottom - 40, self.area.width - 80, 18)
        pygame.draw.rect(surface, (70, 70, 90), bar_rect, border_radius=6)
        fill = bar_rect.inflate(-4, -4)
        fill.width = int(fill.width * sim.progress)
        pygame.draw.rect(surface, HIGHLIGHT_COLOR, fill, border_radius=6)
        progress_text = self.font.render(f"Progression : {sim.progress * 100:5.1f}%", True, TEXT_COLOR)
        surface.blit(progress_text, (bar_rect.x, bar_rect.y - 24))


# ---------------------------------------------------------------------------
# Application principale
# ---------------------------------------------------------------------------


class Application:
    def __init__(self, headless_test: bool = False):
        self.headless_test = headless_test
        pygame.init()
        flags = pygame.HIDDEN if headless_test else 0
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
        pygame.display.set_caption("Bras rigide 1 axe")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(FONT_NAME, 18)
        self.small_font = pygame.font.Font(FONT_NAME, 14)
        self.params = self.load_params()
        self.simulation = ArmSimulation(self.params)
        self.planner = TrajectoryPlanner(self.params)
        self.mode = "run"  # run ou edit
        self.message = ""
        self.message_time = 0.0
        self.toolbar_buttons: List[Button] = []
        self.plot_panel = PlotPanel(
            pygame.Rect(SIM_PANEL_WIDTH, TOOLBAR_HEIGHT, PLOT_PANEL_WIDTH, WINDOW_HEIGHT - TOOLBAR_HEIGHT),
            self.small_font,
        )
        self.sim_renderer = SimulationRenderer(
            pygame.Rect(0, TOOLBAR_HEIGHT, SIM_PANEL_WIDTH, WINDOW_HEIGHT - TOOLBAR_HEIGHT), self.font
        )
        editor_area = pygame.Rect(
            SIM_PANEL_WIDTH + 10,
            TOOLBAR_HEIGHT + 10,
            PLOT_PANEL_WIDTH - 20,
            WINDOW_HEIGHT - TOOLBAR_HEIGHT - 20,
        )
        self.editor = ParameterEditor(self.small_font, self.params, editor_area)
        self.create_toolbar()
        self.recompute_trajectory(initial=True, start_after=False)

    # ------------------------------------------------------------------
    # Config gestion
    # ------------------------------------------------------------------

    def load_params(self) -> ArmParameters:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                return ArmParameters.from_dict(data)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Impossible de charger {CONFIG_PATH}: {exc}")
        return ArmParameters()

    def save_params(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
                json.dump(self.params.to_dict(), fp, indent=2)
            self.set_message("Configuration sauvegardée", success=True)
        except OSError as exc:
            self.set_message(f"Erreur sauvegarde config : {exc}", success=False)

    # ------------------------------------------------------------------

    def create_toolbar(self):
        labels = [
            ("Modifier", self.toggle_mode),
            ("Lancer", self.on_launch),
            ("Réinitialiser", self.on_reset),
            ("Exporter", self.on_export),
            ("Pause", self.on_pause),
            ("Courbes freeze", self.on_freeze_plots),
            ("Courbes reset", self.on_reset_plots),
            ("Sauver cfg", self.save_params),
            ("Charger cfg", self.on_reload_config),
        ]
        x = 10
        for label, callback in labels:
            rect = pygame.Rect(x, 10, 130, 40)
            self.toolbar_buttons.append(Button(rect, label, callback, self.small_font))
            x += rect.width + 8

    def toggle_mode(self):
        if self.mode == "run":
            self.mode = "edit"
            self.editor.update_from_params()
            self.simulation.paused = True
            self.set_message("Mode édition actif", success=True)
        else:
            if self.editor.apply_changes():
                self.mode = "run"
                self.simulation.update_parameters(self.params)
                self.planner = TrajectoryPlanner(self.params)
                try:
                    self.recompute_trajectory(start_after=False)
                    self.set_message("Retour mode simulation", success=True)
                except TrajectoryError as exc:
                    self.mode = "edit"
                    self.set_message(str(exc), success=False)
            else:
                self.set_message(self.editor.error_message, success=False)

    def on_launch(self):
        try:
            self.recompute_trajectory(start_after=True)
            self.set_message("Trajectoire lancée", success=True)
        except TrajectoryError as exc:
            self.set_message(str(exc), success=False)

    def on_reset(self):
        self.simulation.reset_state()
        self.set_message("Simulation réinitialisée", success=True)

    def on_export(self):
        try:
            with open(EXPORT_PATH, "w", encoding="utf-8") as fp:
                fp.write(
                    "time,theta,omega,alpha,theta_ref,omega_ref,alpha_ref,tau_motor,tau_gravity,tau_damping,power,x,y,vx,vy,ax,ay\n"
                )
                for entry in self.simulation.log:
                    fp.write(
                        f"{entry.t:.5f},{entry.theta:.6f},{entry.omega:.6f},{entry.alpha:.6f},{entry.theta_ref:.6f},{entry.omega_ref:.6f},{entry.alpha_ref:.6f},{entry.tau_motor:.6f},{entry.tau_gravity:.6f},{entry.tau_damping:.6f},{entry.power:.6f},{entry.x:.6f},{entry.y:.6f},{entry.vx:.6f},{entry.vy:.6f},{entry.ax:.6f},{entry.ay:.6f}\n"
                    )
            self.set_message(f"Exporté vers {EXPORT_PATH}", success=True)
        except OSError as exc:
            self.set_message(f"Erreur export : {exc}", success=False)

    def on_pause(self):
        self.simulation.toggle_pause()
        self.set_message("Pause" if self.simulation.paused else "Lecture", success=True)

    def on_freeze_plots(self):
        self.plot_panel.toggle_freeze()
        self.simulation.freeze_logs = self.plot_panel.freeze
        self.set_message("Courbes gelées" if self.plot_panel.freeze else "Courbes actives", success=True)

    def on_reset_plots(self):
        self.plot_panel.reset()
        self.set_message("Courbes réinitialisées", success=True)

    def on_reload_config(self):
        self.params = self.load_params()
        self.simulation.update_parameters(self.params)
        self.planner = TrajectoryPlanner(self.params)
        self.editor.params = self.params
        self.editor.update_from_params()
        try:
            self.recompute_trajectory(start_after=False)
            self.set_message("Configuration chargée", success=True)
        except TrajectoryError as exc:
            self.set_message(str(exc), success=False)

    def set_message(self, text: str, success: bool):
        self.message = text
        self.message_time = time.time()
        self.message_color = SUCCESS_COLOR if success else ERROR_COLOR

    # ------------------------------------------------------------------

    def recompute_trajectory(self, initial: bool = False, start_after: bool = True):
        self.params.update_auto_limits()
        profile = self.planner.plan()
        self.simulation.set_trajectory(profile)
        if not initial and start_after:
            self.simulation.start()

    # ------------------------------------------------------------------

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.on_launch()
                elif event.key == pygame.K_r:
                    self.on_reset()
                elif event.key == pygame.K_p:
                    self.on_pause()
            for button in self.toolbar_buttons:
                button.handle_event(event)
            if self.mode == "edit":
                self.editor.handle_event(event)
            else:
                self.plot_panel.handle_event(event)
        return True

    def draw_toolbar(self):
        pygame.draw.rect(self.screen, TOOLBAR_BG, pygame.Rect(0, 0, WINDOW_WIDTH, TOOLBAR_HEIGHT))
        for button in self.toolbar_buttons:
            button.draw(self.screen)
        if self.message:
            if time.time() - self.message_time > 4.0:
                self.message = ""
            else:
                text = self.small_font.render(self.message, True, self.message_color)
                self.screen.blit(text, (WINDOW_WIDTH - text.get_width() - 20, 20))

    def update(self, dt: float):
        sub_steps = max(1, int(dt * PHYSICS_HZ))
        step_dt = dt / sub_steps
        for _ in range(sub_steps):
            self.simulation.physics_step(step_dt)
        self.plot_panel.update(self.simulation.log)

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_toolbar()
        self.sim_renderer.draw(self.screen, self.simulation)
        if self.mode == "edit":
            self.editor.draw(self.screen)
        else:
            self.plot_panel.draw(self.screen)
        pygame.display.flip()

    def run(self):
        running = True
        elapsed = 0.0
        while running:
            dt = self.clock.tick(FPS_TARGET) / 1000.0
            elapsed += dt
            running = self.handle_events()
            self.update(dt)
            self.render()
            if self.headless_test and elapsed > 2.0:
                break
        pygame.quit()


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Simulation bras rigide")
    parser.add_argument("--headless-test", action="store_true", help="Exécute brièvement sans interface")
    args = parser.parse_args(argv)
    if args.headless_test:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    app = Application(headless_test=args.headless_test)
    app.run()


if __name__ == "__main__":
    main()
