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
import csv
import json
import math
import os
import sys
import time
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import pygame

from physics import ArmModel, ArmParameters, ArmState, clamp

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
TABLE_EXPORT_DIR = "tables"

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
    "couple_moteur": (255, 120, 120),
    "couple_gravite": (255, 200, 180),
    "couple_total": (255, 160, 90),
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
                couple_moteur = clamp(value, -abs(self.params.max_torque), abs(self.params.max_torque))
                couple_gravite = model.gravity_torque(theta)
                alpha = (couple_moteur + couple_gravite) / inertia
            else:
                alpha = value
                couple_moteur = inertia * alpha - model.gravity_torque(theta)
                couple_moteur = clamp(couple_moteur, -abs(self.params.max_torque), abs(self.params.max_torque))
            alpha_values.append(alpha)
            torque_values.append(couple_moteur)
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
    couple_moteur: float
    couple_gravite: float
    couple_total: float
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
        self.last_couple_moteur = 0.0
        self.last_couple_gravite = 0.0
        self.last_couple_total = 0.0
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
        self.last_couple_moteur = 0.0
        self.last_couple_total = 0.0
        self.last_couple_gravite = self.model.gravity_torque(self.state.theta)
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

        couple_gravite_state = self.model.gravity_torque(self.state.theta)

        if manual_torque is None:
            couple_commande, tau_g = self.model.required_torque(
                alpha_ref, self.state.theta
            )
            if not self.params.gravity_compensation:
                couple_commande += tau_g
            couple_commande = clamp(
                couple_commande,
                -abs(self.params.max_torque),
                abs(self.params.max_torque),
            )
            alpha = alpha_ref
            couple_gravite = couple_gravite_state
        else:
            couple_commande = clamp(
                manual_torque,
                -abs(self.params.max_torque),
                abs(self.params.max_torque),
            )
            couple_gravite = couple_gravite_state
            alpha = (couple_commande + couple_gravite) / inertia

        self.state.omega += alpha * dt
        self.state.omega = clamp(self.state.omega, -abs(self.params.max_velocity) * 1.5, abs(self.params.max_velocity) * 1.5)
        self.state.theta += self.state.omega * dt
        self.state.alpha = alpha

        self.last_couple_moteur = couple_commande
        self.last_couple_gravite = couple_gravite
        self.last_couple_total = couple_commande + couple_gravite
        self.last_power = couple_commande * self.state.omega

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
                    couple_moteur=couple_commande,
                    couple_gravite=couple_gravite,
                    couple_total=couple_commande + couple_gravite,
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
    def __init__(
        self,
        rect: pygame.Rect,
        options: Sequence[str],
        value: str,
        font: pygame.font.Font,
        on_change: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(rect)
        self.options = list(options)
        self.value = value
        self.open = False
        self.font = font
        self.on_change = on_change

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.open:
                for i, option in enumerate(self.options):
                    opt_rect = pygame.Rect(
                        self.rect.x,
                        self.rect.y + (i + 1) * self.rect.height,
                        self.rect.width,
                        self.rect.height,
                    )
                    if opt_rect.collidepoint(event.pos):
                        self.value = option
                        self.open = False
                        if self.on_change:
                            self.on_change(option)
                        return True
                self.open = False
                return True
            if self.rect.collidepoint(event.pos):
                self.open = True
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.open:
                self.open = False
                return True
        elif event.type == pygame.KEYDOWN:
            if self.open and event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                self.open = False
                return True
        return False

    def _draw_base(self, surface: pygame.Surface):
        pygame.draw.rect(surface, (80, 85, 100), self.rect, border_radius=4)
        text = self.font.render(self.value, True, TEXT_COLOR)
        text_rect = text.get_rect()
        text_rect.midleft = (self.rect.x + 12, self.rect.centery)
        surface.blit(text, text_rect)
        arrow_points = [
            (self.rect.right - 16, self.rect.centery - 4),
            (self.rect.right - 8, self.rect.centery - 4),
            (self.rect.right - 12, self.rect.centery + 4),
        ]
        pygame.draw.polygon(surface, TEXT_COLOR, arrow_points)

    def draw(self, surface: pygame.Surface):
        self._draw_base(surface)

    def draw_overlay(self, surface: pygame.Surface):
        if not self.open:
            return
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (0, 0))
        self._draw_base(surface)
        for i, option in enumerate(self.options):
            opt_rect = pygame.Rect(
                self.rect.x,
                self.rect.y + (i + 1) * self.rect.height,
                self.rect.width,
                self.rect.height,
            )
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
        self.option_catalog: List[Tuple[str, str]] = []
        self.option_map: Dict[str, str] = {}
        self.highlight_key: Optional[str] = None
        self.highlight_time = 0.0
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
        self.option_catalog = [
            ("Compensation gravité", "gravity_compensation"),
            ("Mode trajectoire", "trajectory_mode"),
            ("Entrée manuelle", "manual_input_mode"),
        ]
        y = self.area.y + 10 + 3 * 28
        for label, key in labels:
            rect = pygame.Rect(self.area.x + 10, y, self.area.width - 20, line_height)
            text_input = TextInput(rect, f"{getattr(self.params, key)}", self.font)
            self.widgets[key] = text_input
            self.option_catalog.append((label, key))
            y += line_height + padding
        self.option_map = {label: key for label, key in self.option_catalog}

    def update_from_params(self):
        self.checkbox.value = self.params.gravity_compensation
        self.dropdown.value = self.params.trajectory_mode
        self.manual_mode_dropdown.value = self.params.manual_input_mode
        for key, widget in self.widgets.items():
            if isinstance(widget, TextInput):
                widget.text = f"{getattr(self.params, key)}"

    def has_open_dropdown(self) -> bool:
        return self.dropdown.open or self.manual_mode_dropdown.open

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.has_open_dropdown():
            if self.dropdown.open:
                self.dropdown.handle_event(event)
            if self.manual_mode_dropdown.open:
                self.manual_mode_dropdown.handle_event(event)
            return True
        if self.dropdown.handle_event(event):
            return True
        if self.manual_mode_dropdown.handle_event(event):
            return True
        self.checkbox.handle_event(event)
        for widget in self.widgets.values():
            widget.handle_event(event)
        return False

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
            self.highlight_key = None
            return True
        except ValueError as exc:
            self.error_message = str(exc)
            self.success_message = ""
            return False

    def option_labels(self) -> List[str]:
        return [label for label, _ in self.option_catalog]

    def focus_option_by_label(self, label: str):
        key = self.option_map.get(label)
        if not key:
            return
        self.highlight_key = key
        self.highlight_time = time.time()
        if key in self.widgets:
            for widget in self.widgets.values():
                if isinstance(widget, TextInput):
                    widget.active = False
            widget = self.widgets[key]
            if isinstance(widget, TextInput):
                widget.active = True
        elif key == "trajectory_mode":
            self.dropdown.open = True
        elif key == "manual_input_mode":
            self.manual_mode_dropdown.open = True

    def clear_highlight(self):
        self.highlight_key = None
        self.dropdown.open = False
        self.manual_mode_dropdown.open = False

    def draw_overlays(self, surface: pygame.Surface):
        self.dropdown.draw_overlay(surface)
        self.manual_mode_dropdown.draw_overlay(surface)

    def draw(self, surface: pygame.Surface):
        if self.highlight_key and time.time() - self.highlight_time > 4.0:
            self.clear_highlight()
        pygame.draw.rect(surface, PANEL_BG, self.area)
        if self.highlight_key == "gravity_compensation":
            pygame.draw.rect(surface, HIGHLIGHT_COLOR, self.checkbox.rect.inflate(8, 8), width=2, border_radius=6)
        self.checkbox.draw(surface)
        label_mode = self.font.render("Mode trajectoire", True, TEXT_COLOR)
        surface.blit(label_mode, (self.dropdown.rect.x, self.dropdown.rect.y - 20))
        if self.highlight_key == "trajectory_mode":
            pygame.draw.rect(surface, HIGHLIGHT_COLOR, self.dropdown.rect.inflate(8, 8), width=2, border_radius=6)
        self.dropdown.draw(surface)
        label_manual_mode = self.font.render("Entrée manuelle", True, TEXT_COLOR)
        surface.blit(label_manual_mode, (self.manual_mode_dropdown.rect.x, self.manual_mode_dropdown.rect.y - 20))
        if self.highlight_key == "manual_input_mode":
            pygame.draw.rect(
                surface,
                HIGHLIGHT_COLOR,
                self.manual_mode_dropdown.rect.inflate(8, 8),
                width=2,
                border_radius=6,
            )
        self.manual_mode_dropdown.draw(surface)
        y = self.dropdown.rect.bottom + 10
        for label, key in self.labels:
            label_surface = self.font.render(label, True, TEXT_COLOR)
            surface.blit(label_surface, (self.area.x + 12, y))
            widget = self.widgets[key]
            widget.rect.y = y + 18
            if self.highlight_key == key and isinstance(widget, TextInput):
                pygame.draw.rect(surface, HIGHLIGHT_COLOR, widget.rect.inflate(8, 8), width=2, border_radius=6)
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


@dataclass
class TableRequest:
    parameter_key: str
    parameter_label: str
    values: List[float]
    filename: str


class TableGeneratorPanel:
    def __init__(
        self,
        font: pygame.font.Font,
        params: ArmParameters,
        area: pygame.Rect,
        on_generate: Callable[[TableRequest], Tuple[bool, str, List[str], List[Dict[str, object]]]],
    ):
        self.font = font
        self.params = params
        self.area = area
        self.on_generate = on_generate
        self.status_message = ""
        self.status_success = True
        self.preview_lines: List[str] = []
        self.param_options: List[Tuple[str, str]] = [
            ("Angle départ (deg)", "start_angle_deg"),
            ("Angle arrivée (deg)", "target_angle_deg"),
            ("Longueur (m)", "length"),
            ("Masse bras (kg)", "mass_arm"),
            ("Masse charge (kg)", "mass_load"),
            ("Gravité (m/s²)", "gravity"),
            ("Durée cible (s)", "target_duration"),
            ("Max vitesse (rad/s)", "max_velocity"),
            ("Max accélération (rad/s²)", "max_acceleration"),
            ("Max couple (N·m)", "max_torque"),
        ]
        self.param_map = {label: key for label, key in self.param_options}
        dropdown_rect = pygame.Rect(self.area.x + 12, self.area.y + 48, 300, 32)
        self.param_dropdown = Dropdown(
            dropdown_rect,
            [label for label, _ in self.param_options],
            self.param_options[0][0],
            self.font,
            on_change=self._on_param_change,
        )
        input_width = 200
        first_rect = pygame.Rect(self.area.x + 12, dropdown_rect.bottom + 40, input_width, 32)
        second_rect = pygame.Rect(first_rect.right + 16, first_rect.y, input_width, 32)
        count_rect = pygame.Rect(self.area.x + 12, first_rect.bottom + 36, input_width, 32)
        file_rect = pygame.Rect(second_rect.x, count_rect.y, input_width, 32)
        self.first_input = TextInput(first_rect, "0.0", self.font)
        self.second_input = TextInput(second_rect, "0.0", self.font)
        self.count_input = TextInput(count_rect, "5", self.font)
        self.filename_input = TextInput(file_rect, "tableau", self.font)
        self.inputs = [self.first_input, self.second_input, self.count_input, self.filename_input]
        button_rect = pygame.Rect(self.area.x + 12, count_rect.bottom + 40, 180, 40)
        self.generate_button = Button(button_rect, "Générer", self._request_generation, self.font)
        self._update_inputs_for_param(self.param_options[0][0])

    def sync_params(self, params: ArmParameters):
        self.params = params
        self._update_inputs_for_param(self.param_dropdown.value)

    def clear_status(self):
        self.status_message = ""
        self.preview_lines = []

    def _on_param_change(self, label: str):
        self._update_inputs_for_param(label)

    def _update_inputs_for_param(self, label: str):
        key = self.param_map[label]
        value = getattr(self.params, key, 0.0)
        formatted = f"{value:.6g}"
        self.first_input.text = formatted
        if self.second_input.text.strip() == "" or self.second_input.text == self.first_input.text:
            try:
                increment = value * 0.1 if value != 0 else 1.0
                self.second_input.text = f"{(value + increment):.6g}"
            except TypeError:
                self.second_input.text = formatted

    def _request_generation(self):
        request = self._build_request()
        if not request:
            return
        if not self.on_generate:
            return
        success, message, columns, rows = self.on_generate(request)
        self.status_message = message
        self.status_success = success
        if success:
            preview = []
            for row in rows[: min(6, len(rows))]:
                preview.append(
                    f"{request.parameter_label}={row.get('parameter_value', 0.0):.2f} | "
                    f"t={row.get('time', 0.0):.2f}s | "
                    f"theta={row.get('theta', 0.0):.2f}rad | "
                    f"omega={row.get('omega', 0.0):.2f} | "
                    f"alpha={row.get('alpha', 0.0):.2f} | "
                    f"couple={row.get('couple_moteur', 0.0):.2f}"
                )
            self.preview_lines = preview
        else:
            self.preview_lines = []

    def _build_request(self) -> Optional[TableRequest]:
        label = self.param_dropdown.value
        key = self.param_map[label]
        try:
            first = float(self.first_input.text.strip())
            last = float(self.second_input.text.strip())
            count = int(float(self.count_input.text.strip()))
        except ValueError:
            self.status_message = "Valeurs numériques invalides"
            self.status_success = False
            return None
        if count <= 0:
            self.status_message = "Nombre de lignes doit être positif"
            self.status_success = False
            return None
        filename = self.filename_input.text.strip() or "tableau"
        if count == 1:
            values = [first]
        else:
            step = (last - first) / (count - 1)
            values = [first + i * step for i in range(count)]
        return TableRequest(
            parameter_key=key,
            parameter_label=label,
            values=values,
            filename=filename,
        )

    def has_open_dropdown(self) -> bool:
        return self.param_dropdown.open

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.param_dropdown.open:
            self.param_dropdown.handle_event(event)
            return True
        if self.param_dropdown.handle_event(event):
            return True
        for widget in self.inputs:
            widget.handle_event(event)
        self.generate_button.handle_event(event)
        return False

    def draw_overlays(self, surface: pygame.Surface):
        self.param_dropdown.draw_overlay(surface)

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, PANEL_BG, self.area)
        title = self.font.render("Générateur de tableau", True, TEXT_COLOR)
        surface.blit(title, (self.area.x + 12, self.area.y + 12))
        instruction = self.font.render(
            "Choisissez un paramètre et définissez deux valeurs de début et fin.", True, TEXT_COLOR
        )
        surface.blit(instruction, (self.area.x + 12, self.area.y + 24 + title.get_height()))
        label_param = self.font.render("Paramètre à balayer", True, TEXT_COLOR)
        surface.blit(label_param, (self.param_dropdown.rect.x, self.param_dropdown.rect.y - 24))
        self.param_dropdown.draw(surface)
        label_values = self.font.render("Valeurs début et fin", True, TEXT_COLOR)
        surface.blit(label_values, (self.first_input.rect.x, self.first_input.rect.y - 24))
        for widget in self.inputs:
            widget.draw(surface)
        count_label = self.font.render("Nombre de lignes", True, TEXT_COLOR)
        surface.blit(count_label, (self.count_input.rect.x, self.count_input.rect.y - 24))
        file_label = self.font.render("Nom du fichier", True, TEXT_COLOR)
        surface.blit(file_label, (self.filename_input.rect.x, self.filename_input.rect.y - 24))
        self.generate_button.draw(surface)
        status_color = SUCCESS_COLOR if self.status_success else ERROR_COLOR
        status_height = 0
        if self.status_message:
            status = self.font.render(self.status_message, True, status_color)
            surface.blit(status, (self.area.x + 12, self.generate_button.rect.bottom + 12))
            status_height = status.get_height() + 6
        if self.preview_lines:
            preview_title = self.font.render("Aperçu (max 6 lignes)", True, TEXT_COLOR)
            y = self.generate_button.rect.bottom + 12 + status_height
            surface.blit(preview_title, (self.area.x + 12, y))
            y += preview_title.get_height() + 6
            for line in self.preview_lines:
                text = self.font.render(line, True, TEXT_COLOR)
                surface.blit(text, (self.area.x + 12, y))
                y += text.get_height() + 2

class PlotPanel:
    def __init__(self, area: pygame.Rect, font: pygame.font.Font):
        self.area = area
        self.font = font
        self.signals = [
            "angle",
            "omega",
            "alpha",
            "couple_moteur",
            "couple_gravite",
            "couple_total",
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
            in (
                "angle",
                "omega",
                "alpha",
                "couple_moteur",
                "couple_gravite",
                "couple_total",
                "power",
                "x",
                "y",
                "vx",
                "vy",
                "ax",
                "ay",
            )
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
        self.cursor_enabled = False
        self.cursor_pos: Optional[Tuple[int, int]] = None
        self.plot_rect = pygame.Rect(0, 0, 0, 0)

    def toggle_freeze(self):
        self.freeze = not self.freeze

    def toggle_cursor(self) -> bool:
        self.cursor_enabled = not self.cursor_enabled
        if not self.cursor_enabled:
            self.cursor_pos = None
        return self.cursor_enabled

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
            self.cached_data["couple_moteur"].append(entry.couple_moteur)
            self.cached_data["couple_gravite"].append(entry.couple_gravite)
            self.cached_data["couple_total"].append(entry.couple_total)
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
        elif event.type == pygame.MOUSEMOTION and self.cursor_enabled:
            if self.plot_rect.collidepoint(event.pos):
                self.cursor_pos = event.pos
            else:
                self.cursor_pos = None

    def draw(self, surface: pygame.Surface):
        pygame.draw.rect(surface, PANEL_BG, self.area)
        legend_rows = 2
        checkbox_height = 16
        row_spacing = 8
        legend_height = legend_rows * (checkbox_height + row_spacing) - row_spacing
        legend_area = pygame.Rect(
            self.area.x + 10,
            self.area.y + 10,
            self.area.width - 20,
            legend_height,
        )
        # Draw checkboxes
        items_per_row = math.ceil(len(self.signals) / legend_rows)
        row_x = [legend_area.x for _ in range(legend_rows)]
        self.checkbox_rects.clear()
        for index, name in enumerate(self.signals):
            row = min(index // items_per_row, legend_rows - 1)
            col_y = legend_area.y + row * (checkbox_height + row_spacing)
            checkbox = pygame.Rect(row_x[row], col_y, checkbox_height, checkbox_height)
            pygame.draw.rect(surface, TRACE_COLORS[name], checkbox, width=2)
            if self.selected[name]:
                pygame.draw.rect(surface, TRACE_COLORS[name], checkbox.inflate(-4, -4))
            label = self.font.render(name, True, TEXT_COLOR)
            surface.blit(label, (checkbox.right + 4, checkbox.y - 2))
            row_x[row] = checkbox.right + label.get_width() + 20
            self.checkbox_rects[name] = checkbox

        plot_rect = pygame.Rect(
            self.area.x + 10,
            legend_area.bottom + 10,
            self.area.width - 20,
            self.area.height - legend_area.height - 40,
        )
        self.plot_rect = plot_rect
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

        if self.cursor_enabled and self.cursor_pos and plot_rect.collidepoint(self.cursor_pos):
            cursor_x, cursor_y = self.cursor_pos
            cursor_x = clamp(cursor_x, plot_rect.left, plot_rect.right)
            cursor_y = clamp(cursor_y, plot_rect.top, plot_rect.bottom)
            pygame.draw.line(surface, HIGHLIGHT_COLOR, (cursor_x, plot_rect.top), (cursor_x, plot_rect.bottom), 1)
            pygame.draw.line(surface, HIGHLIGHT_COLOR, (plot_rect.left, cursor_y), (plot_rect.right, cursor_y), 1)

            cursor_time = t0 + (cursor_x - plot_rect.x) / plot_rect.width * (t1 - t0)
            closest_index = min(range(len(self.time_data)), key=lambda i: abs(self.time_data[i] - cursor_time))
            info_lines = [f"t = {self.time_data[closest_index]:.3f} s"]
            for name in active_signals:
                value = self.cached_data[name][closest_index]
                info_lines.append(f"{name} = {value:.3f}")

            info_width = max(self.font.render(text, True, TEXT_COLOR).get_width() for text in info_lines) + 12
            info_height = len(info_lines) * 18 + 8
            info_surface = pygame.Surface((info_width, info_height), pygame.SRCALPHA)
            info_surface.fill((20, 20, 28, 220))
            for i, text in enumerate(info_lines):
                label = self.font.render(text, True, TEXT_COLOR)
                info_surface.blit(label, (6, 4 + i * 18))
            info_x = clamp(cursor_x + 12, plot_rect.left, plot_rect.right - info_surface.get_width())
            info_y = clamp(cursor_y - info_surface.get_height() - 12, plot_rect.top, plot_rect.bottom - info_surface.get_height())
            surface.blit(info_surface, (info_x, info_y))


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
            f"Couple moteur : {sim.last_couple_moteur:6.2f} N·m",
            f"Couple gravité : {sim.last_couple_gravite:6.2f} N·m",
            f"Couple total   : {sim.last_couple_total:6.2f} N·m",
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
        if headless_test:
            flags = pygame.HIDDEN
            size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        else:
            flags = pygame.FULLSCREEN
            size = (0, 0)
        self.screen = pygame.display.set_mode(size, flags)
        self.window_width, self.window_height = self.screen.get_size()
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
        self.toolbar_dropdowns: List[Dropdown] = []
        self.modifier_dropdown: Optional[Dropdown] = None
        self.sim_panel_width = int(self.window_width * 0.48)
        self.plot_panel_width = self.window_width - self.sim_panel_width
        self.plot_panel = PlotPanel(
            pygame.Rect(
                self.sim_panel_width,
                TOOLBAR_HEIGHT,
                self.plot_panel_width,
                self.window_height - TOOLBAR_HEIGHT,
            ),
            self.small_font,
        )
        self.sim_renderer = SimulationRenderer(
            pygame.Rect(
                0,
                TOOLBAR_HEIGHT,
                self.sim_panel_width,
                self.window_height - TOOLBAR_HEIGHT,
            ),
            self.font,
        )
        editor_area = pygame.Rect(
            self.sim_panel_width + 10,
            TOOLBAR_HEIGHT + 10,
            self.plot_panel_width - 20,
            self.window_height - TOOLBAR_HEIGHT - 20,
        )
        self.editor = ParameterEditor(self.small_font, self.params, editor_area)
        self.table_panel = TableGeneratorPanel(
            self.small_font,
            self.params,
            editor_area,
            on_generate=self.on_generate_table,
        )
        self.previous_mode = "run"
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
        self.toolbar_buttons.clear()
        self.toolbar_dropdowns = []
        dropdown_width = 260
        dropdown_height = 32
        dropdown_rect = pygame.Rect(10, (TOOLBAR_HEIGHT - dropdown_height) // 2, dropdown_width, dropdown_height)
        options = ["Modifier..."] + self.editor.option_labels()
        self.modifier_dropdown = Dropdown(
            dropdown_rect,
            options,
            "Modifier...",
            self.small_font,
            on_change=self.on_modifier_option_selected,
        )
        self.toolbar_dropdowns.append(self.modifier_dropdown)
        x = dropdown_rect.right + 12
        labels = [
            ("Modifier", self.toggle_mode),
            ("Tableau", self.toggle_table_mode),
            ("Lancer", self.on_launch),
            ("Réinitialiser", self.on_reset),
            ("Exporter", self.on_export),
            ("Pause", self.on_pause),
            ("Courbes freeze", self.on_freeze_plots),
            ("Courbes reset", self.on_reset_plots),
            ("Curseur", self.on_toggle_cursor),
            ("Sauver cfg", self.save_params),
            ("Charger cfg", self.on_reload_config),
        ]
        for label, callback in labels:
            rect = pygame.Rect(x, 10, 130, 40)
            self.toolbar_buttons.append(Button(rect, label, callback, self.small_font))
            x += rect.width + 8

    def on_modifier_option_selected(self, option: str):
        if option == "Modifier...":
            return
        if self.mode != "edit":
            self.toggle_mode()
        if self.mode == "edit":
            self.editor.focus_option_by_label(option)
            self.set_message(f"Sélection : {option}", success=True)
        if self.modifier_dropdown:
            self.modifier_dropdown.value = "Modifier..."
            self.modifier_dropdown.open = False

    def toggle_mode(self):
        if self.mode == "table":
            self.mode = "edit"
            self.table_panel.sync_params(self.params)
            self.table_panel.clear_status()
            self.editor.update_from_params()
            self.editor.clear_highlight()
            self.simulation.paused = True
            self.set_message("Mode édition actif", success=True)
            return
        if self.mode == "run":
            self.mode = "edit"
            self.editor.update_from_params()
            self.editor.clear_highlight()
            self.simulation.paused = True
            self.set_message("Mode édition actif", success=True)
        else:
            if self.editor.apply_changes():
                self.editor.clear_highlight()
                self.mode = "run"
                self.simulation.update_parameters(self.params)
                self.planner = TrajectoryPlanner(self.params)
                try:
                    self.recompute_trajectory(start_after=False)
                    self.table_panel.sync_params(self.params)
                    self.set_message("Retour mode simulation", success=True)
                except TrajectoryError as exc:
                    self.mode = "edit"
                    self.set_message(str(exc), success=False)
            else:
                self.set_message(self.editor.error_message, success=False)

    def toggle_table_mode(self):
        if self.mode == "table":
            self.mode = self.previous_mode if self.previous_mode in ("run", "edit") else "run"
            if self.mode == "edit":
                self.editor.update_from_params()
                self.editor.clear_highlight()
                self.simulation.paused = True
            else:
                self.simulation.paused = False
            self.table_panel.clear_status()
            self.set_message("Mode tableau fermé", success=True)
            return

        if self.mode == "edit":
            if not self.editor.apply_changes():
                self.set_message(self.editor.error_message, success=False)
                return
            self.editor.clear_highlight()
            self.simulation.update_parameters(self.params)
            self.planner = TrajectoryPlanner(self.params)
            try:
                self.recompute_trajectory(start_after=False)
            except TrajectoryError as exc:
                self.mode = "edit"
                self.set_message(str(exc), success=False)
                return
            self.mode = "run"
            self.set_message("Paramètres appliqués", success=True)

        self.previous_mode = self.mode
        self.mode = "table"
        self.simulation.paused = True
        self.table_panel.sync_params(self.params)
        self.set_message("Mode tableau actif", success=True)

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
                    "time,theta,omega,alpha,couple_moteur,couple_gravite\n"
                )
                for entry in self.simulation.log:
                    fp.write(
                        f"{entry.t:.2f},{entry.theta:.2f},{entry.omega:.2f},{entry.alpha:.2f},{entry.couple_moteur:.2f},{entry.couple_gravite:.2f}\n"
                    )
            self.set_message(f"Exporté vers {EXPORT_PATH}", success=True)
        except OSError as exc:
            self.set_message(f"Erreur export : {exc}", success=False)

    def on_generate_table(self, request: TableRequest) -> Tuple[bool, str, List[str], List[Dict[str, object]]]:
        preview_rows: List[Dict[str, object]] = []
        table_blocks: List[Tuple[float, List[Dict[str, object]]]] = []
        columns = [
            "parameter_label",
            "parameter_value",
            "time",
            "theta",
            "omega",
            "alpha",
            "couple_moteur",
            "couple_gravite",
        ]
        for value in request.values:
            sweep_params = replace(self.params)
            setattr(sweep_params, request.parameter_key, value)
            if request.parameter_key in {"max_velocity", "max_acceleration", "max_torque"}:
                sweep_params.auto_limits = False
            sweep_params.update_auto_limits()
            planner = TrajectoryPlanner(sweep_params)
            try:
                profile = planner.plan()
            except TrajectoryError as exc:
                message = f"Trajectoire impossible pour {request.parameter_label}={value:.4g} : {exc}"
                self.set_message(message, success=False)
                return False, message, [], []
            simulation = ArmSimulation(sweep_params)
            simulation.set_trajectory(profile)
            simulation.start()
            dt = 1.0 / PHYSICS_HZ
            max_steps = max(1, int(profile.duration / dt) + PHYSICS_HZ * 2)
            for _ in range(max_steps):
                simulation.physics_step(dt)
                if not simulation.running:
                    break
            if not simulation.log:
                message = f"Aucune donnée générée pour {request.parameter_label}={value:.4g}"
                self.set_message(message, success=False)
                return False, message, [], []
            rows_for_value = self._rows_from_log_for_table(request, value, simulation.log)
            table_blocks.append((value, rows_for_value))
            preview_rows.extend(rows_for_value)

        try:
            file_path = self._write_table_file(
                request.filename, request.parameter_label, columns, table_blocks
            )
        except OSError as exc:
            message = f"Erreur écriture tableau : {exc}"
            self.set_message(message, success=False)
            return False, message, [], []

        message = f"Tableau exporté vers {file_path}"
        self.set_message(message, success=True)
        return True, message, columns, preview_rows

    def _rows_from_log_for_table(
        self,
        request: TableRequest,
        value: float,
        log: List[LogEntry],
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for entry in log:
            rows.append(
                {
                    "parameter_label": request.parameter_label,
                    "parameter_value": round(value, 2),
                    "time": round(entry.t, 2),
                    "theta": round(entry.theta, 2),
                    "omega": round(entry.omega, 2),
                    "alpha": round(entry.alpha, 2),
                    "couple_moteur": round(entry.couple_moteur, 2),
                    "couple_gravite": round(entry.couple_gravite, 2),
                }
            )
        return rows

    def _write_table_file(
        self,
        filename: str,
        parameter_label: str,
        columns: List[str],
        table_blocks: List[Tuple[float, List[Dict[str, object]]]],
    ) -> str:
        os.makedirs(TABLE_EXPORT_DIR, exist_ok=True)
        sanitized = filename.strip() or "tableau"
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        sanitized = "".join(ch for ch in sanitized if ch in allowed or ch == ".")
        if not sanitized:
            sanitized = "tableau"
        if sanitized.lower().endswith(".csv"):
            extension = ".csv"
        elif sanitized.lower().endswith(".xlsx"):
            extension = ".xlsx"
        else:
            sanitized += ".xlsx"
            extension = ".xlsx"
        path = os.path.join(TABLE_EXPORT_DIR, sanitized)
        if extension == ".csv":
            with open(path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                column_count = len(columns)
                for index, (value, rows) in enumerate(table_blocks):
                    if index > 0:
                        writer.writerow([])
                    summary_row: List[str] = [""] * column_count
                    if column_count >= 1:
                        summary_row[0] = parameter_label
                    if column_count >= 2:
                        summary_row[1] = f"{value:.2f}"
                    writer.writerow(summary_row)
                    writer.writerow(columns)
                    for row in rows:
                        formatted_row: List[str] = []
                        for key in columns:
                            cell = row.get(key, "")
                            if isinstance(cell, (int, float)):
                                formatted_row.append(f"{cell:.2f}")
                            else:
                                formatted_row.append(str(cell))
                        writer.writerow(formatted_row)
        else:
            sheets: List[Tuple[str, List[List[str]]]] = []
            existing_names: Set[str] = set()
            for value, rows in table_blocks:
                sheet_name = self._sanitize_sheet_name(
                    f"{parameter_label}={value:.2f}", existing_names
                )
                column_count = len(columns)
                summary_row: List[str] = [""] * max(column_count, 2)
                summary_row[0] = parameter_label
                summary_row[1] = f"{value:.2f}"
                if column_count > 0:
                    summary_display = summary_row[:column_count]
                else:
                    summary_display = summary_row
                formatted_rows: List[List[str]] = [summary_display]
                formatted_rows.append(list(columns))
                for row in rows:
                    formatted_row: List[str] = []
                    for key in columns:
                        cell = row.get(key, "")
                        if isinstance(cell, (int, float)):
                            formatted_row.append(f"{cell:.2f}")
                        else:
                            formatted_row.append(str(cell))
                    formatted_rows.append(formatted_row)
                sheets.append((sheet_name, formatted_rows))
            self._write_xlsx(path, sheets)
        return path

    def _sanitize_sheet_name(self, name: str, existing: Set[str]) -> str:
        invalid = "[]:*?/\\"
        sanitized = "".join("_" if ch in invalid else ch for ch in name)
        sanitized = sanitized.strip()
        if not sanitized:
            sanitized = "Feuille"
        if len(sanitized) > 31:
            sanitized = sanitized[:31]
        base = sanitized
        counter = 1
        while sanitized in existing:
            suffix = f"_{counter}"
            sanitized = f"{base[: max(0, 31 - len(suffix))]}{suffix}" or f"Feuille_{counter}"
            counter += 1
        existing.add(sanitized)
        return sanitized

    def _write_xlsx(self, path: str, sheets: List[Tuple[str, List[List[str]]]]):
        if not sheets:
            raise ValueError("Aucune donnée à écrire dans le classeur")
        with ZipFile(path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", self._xlsx_content_types(len(sheets)))
            archive.writestr("_rels/.rels", self._xlsx_root_rels())
            archive.writestr("xl/workbook.xml", self._xlsx_workbook_xml(sheets))
            archive.writestr("xl/_rels/workbook.xml.rels", self._xlsx_workbook_rels(len(sheets)))
            for index, (_, rows) in enumerate(sheets, start=1):
                archive.writestr(
                    f"xl/worksheets/sheet{index}.xml",
                    self._xlsx_sheet_xml(rows),
                )

    def _xlsx_content_types(self, sheet_count: int) -> str:
        overrides = [
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        ]
        for index in range(1, sheet_count + 1):
            overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        parts = "".join(overrides)
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
            "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
            "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            f"{parts}" "</Types>"
        )

    def _xlsx_root_rels(self) -> str:
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
            "</Relationships>"
        )

    def _xlsx_workbook_xml(self, sheets: List[Tuple[str, List[List[str]]]]) -> str:
        sheet_entries = []
        for index, (name, _) in enumerate(sheets, start=1):
            sheet_entries.append(
                f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
            )
        sheets_xml = "".join(sheet_entries)
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
            "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
            f"<sheets>{sheets_xml}</sheets>"
            "</workbook>"
        )

    def _xlsx_workbook_rels(self, sheet_count: int) -> str:
        relationships = []
        for index in range(1, sheet_count + 1):
            relationships.append(
                f'<Relationship Id=\"rId{index}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet{index}.xml\"/>'
            )
        relations_xml = "".join(relationships)
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            f"{relations_xml}" "</Relationships>"
        )

    def _xlsx_sheet_xml(self, rows: List[List[str]]) -> str:
        lines = [
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">",
            "<sheetData>",
        ]
        for row_index, row in enumerate(rows, start=1):
            lines.append(f'<row r="{row_index}">')
            for column_index, value in enumerate(row, start=1):
                column_letter = self._xlsx_column_letter(column_index)
                cell_ref = f"{column_letter}{row_index}"
                text = escape(str(value)) if value is not None else ""
                lines.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
            lines.append("</row>")
        lines.extend(["</sheetData>", "</worksheet>"])
        return "".join(lines)

    def _xlsx_column_letter(self, index: int) -> str:
        result = ""
        current = index
        while current > 0:
            current, remainder = divmod(current - 1, 26)
            result = chr(65 + remainder) + result
        return result or "A"

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

    def on_toggle_cursor(self):
        enabled = self.plot_panel.toggle_cursor()
        self.set_message("Curseur activé" if enabled else "Curseur désactivé", success=True)

    def on_reload_config(self):
        self.params = self.load_params()
        self.simulation.update_parameters(self.params)
        self.planner = TrajectoryPlanner(self.params)
        self.editor.params = self.params
        self.editor.update_from_params()
        self.table_panel.sync_params(self.params)
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
            toolbar_modal = any(dropdown.open for dropdown in self.toolbar_dropdowns)
            if toolbar_modal:
                for dropdown in self.toolbar_dropdowns:
                    dropdown.handle_event(event)
                continue
            dropdown_consumed = False
            for dropdown in self.toolbar_dropdowns:
                if dropdown.handle_event(event):
                    dropdown_consumed = True
            if dropdown_consumed:
                continue
            for button in self.toolbar_buttons:
                button.handle_event(event)
            if self.mode == "edit":
                if self.editor.has_open_dropdown():
                    self.editor.handle_event(event)
                    continue
                self.editor.handle_event(event)
            elif self.mode == "table":
                if self.table_panel.has_open_dropdown():
                    self.table_panel.handle_event(event)
                    continue
                self.table_panel.handle_event(event)
            else:
                self.plot_panel.handle_event(event)
        return True

    def draw_toolbar(self):
        pygame.draw.rect(self.screen, TOOLBAR_BG, pygame.Rect(0, 0, self.window_width, TOOLBAR_HEIGHT))
        for button in self.toolbar_buttons:
            button.draw(self.screen)
        for dropdown in self.toolbar_dropdowns:
            dropdown.draw(self.screen)
        if self.message:
            if time.time() - self.message_time > 4.0:
                self.message = ""
            else:
                text = self.small_font.render(self.message, True, self.message_color)
                self.screen.blit(text, (self.window_width - text.get_width() - 20, 20))

    def draw_dropdown_modals(self):
        for dropdown in self.toolbar_dropdowns:
            dropdown.draw_overlay(self.screen)
        if self.mode == "edit":
            self.editor.draw_overlays(self.screen)
        elif self.mode == "table":
            self.table_panel.draw_overlays(self.screen)

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
        elif self.mode == "table":
            self.table_panel.draw(self.screen)
        else:
            self.plot_panel.draw(self.screen)
        self.draw_dropdown_modals()
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
