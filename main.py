"""
Bras robot 1‑axe – moteur de simulation (Pygame)
-------------------------------------------------
Hugo, voici une base propre pour simuler un bras rigide en rotation autour d'un pivot,
avec masse du bras + charge en bout, gravité, amortissement visqueux, et un couple
moteur calculé à partir d'une accélération angulaire demandée (alpha_cmd).

Contrôles clavier
-----------------
 ↑ / ↓ : augmenter / diminuer l'accélération demandée (alpha_cmd)
 A / Z  : augmenter / diminuer le couple moteur directement (mode manuel)
 M      : basculer mode contrôle (accélération -> couple manuel)
 R      : reset (θ, ω = 0)
 P      : pause / reprise
 G      : geler/dégeler la gravité
 ESC    : quitter

Affichages
----------
- θ (rad & °), ω (rad/s), α (rad/s²)
- alpha_cmd (rad/s²)
- Couple moteur τ_motor (N·m)
- Couple gravité τ_g (N·m)
- Couple amortissement τ_d (N·m)
- Inertie équivalente I (kg·m²)
- Puissance instantanée P = τ_motor * ω (W)

Modèle physique
---------------
I_total = I_bras + I_charge
I_bras (tige mince pivotée à une extrémité) = (1/3) * m_bras * L^2
I_charge (masse ponctuelle en bout) = m_charge * L^2

Équation du mouvement : I * α = τ_motor + τ_g + τ_d
  où τ_g = - (m_bras*g*L/2 + m_charge*g*L) * sin(θ)
      τ_d = - b * ω (amortissement visqueux simple)

Intégration numérique : semi‑implicite (symplectic) Euler
  ω ← ω + α * dt
  θ ← θ + ω * dt

Tu peux ajuster facilement les constantes dans la section CONFIG.

Prochaines étapes possibles
---------------------------
- Limites de couple et de vitesse moteur réalistes
- Réducteur (ratio, inerties ramenées) et frottements secs (Coulomb)
- Commande en position (PD / PID) et profils S‑courbe (accélération limitée)
- Export CSV pour post‑traitement
- Champs d'inertie variables (charge à mi‑bras, etc.)
"""
import math
import pygame
from dataclasses import dataclass

# ===================== CONFIG =====================
WIDTH, HEIGHT = 960, 600
FPS_TARGET = 120               # fréquence de rendu (Hz)
PHYSICS_HZ = 600               # fréquence d'intégration physique (Hz)

# Paramètres physiques (unité SI)
G = 9.81                       # gravité (m/s²)
L = 0.6                        # longueur du bras (m)
M_BRAS = 1.0                   # masse du bras (kg)
M_CHARGE = 0.5                 # masse en bout (kg)
B_DAMP = 0.15                  # coefficient d'amortissement visqueux (N·m·s/rad)

# Limites de sécurité / confort
THETA_MIN = -math.radians(170) # bornes d'angle (évite un tour complet en visu)
THETA_MAX =  math.radians(170)
OMEGA_MAX =  20.0              # rad/s (limitation numérique simple)
TAU_LIMIT =  25.0              # N·m (limite du couple moteur appliqué)
ALPHA_STEP =  2.0              # incrément de alpha_cmd par appui (rad/s²)
TAU_STEP   =  1.0              # incrément de couple manuel par appui (N·m)

# Rendu
ARM_COLOR = (240, 240, 255)
PIVOT_COLOR = (200, 200, 220)
MASS_COLOR = (170, 200, 255)
BACKGROUND = (20, 24, 28)
TEXT_COLOR = (230, 230, 235)

# Échelle pixels ↔ mètres (pour l'affichage uniquement)
PX_PER_M = 350

# ===================== MODÈLE =====================
@dataclass
class ArmState:
    theta: float = 0.0   # angle (rad), 0 = bras horizontal à droite, sens CCW positif
    omega: float = 0.0   # vitesse angulaire (rad/s)
    alpha: float = 0.0   # accélération angulaire (rad/s²)

@dataclass
class ArmParams:
    L: float
    m_bras: float
    m_charge: float
    g: float
    b: float

    @property
    def I(self) -> float:
        # inertie bras + masse ponctuelle en bout
        I_bras = (1.0/3.0) * self.m_bras * self.L**2
        I_charge = self.m_charge * self.L**2
        return I_bras + I_charge

    @property
    def tau_g_mag(self) -> float:
        # magnitude du terme gravitaire combiné
        return (self.m_bras * self.g * self.L/2.0) + (self.m_charge * self.g * self.L)

class ArmSim:
    def __init__(self, params: ArmParams):
        self.p = params
        self.s = ArmState()
        self.alpha_cmd = 0.0   # mode contrôle par accélération demandée
        self.tau_manual = 0.0  # mode contrôle par couple direct
        self.use_accel_mode = True
        self.gravity_on = True
        self.paused = False

    def reset(self):
        self.s = ArmState()
        self.alpha_cmd = 0.0
        self.tau_manual = 0.0

    def physics_step(self, dt: float):
        if self.paused:
            return
        # Couples
        tau_g = - self.p.tau_g_mag * math.sin(self.s.theta) if self.gravity_on else 0.0
        tau_d = - self.p.b * self.s.omega
        if self.use_accel_mode:
            # τ_motor qui permet d'atteindre alpha_cmd : I*α = τ_motor + τ_g + τ_d
            tau_motor = self.p.I * self.alpha_cmd - tau_g - tau_d
        else:
            tau_motor = self.tau_manual

        # Saturation couple moteur
        tau_motor = max(-TAU_LIMIT, min(TAU_LIMIT, tau_motor))

        # Dynamique
        alpha = (tau_motor + tau_g + tau_d) / self.p.I
        # Limitation vitesse (stabilité num.)
        omega = self.s.omega + alpha * dt
        omega = max(-OMEGA_MAX, min(OMEGA_MAX, omega))
        theta = self.s.theta + omega * dt
        # bornes d'angle simples (rebond doux)
        if theta < THETA_MIN:
            theta = THETA_MIN
            omega = 0.0
        elif theta > THETA_MAX:
            theta = THETA_MAX
            omega = 0.0

        # mise à jour état
        self.s.alpha = alpha
        self.s.omega = omega
        self.s.theta = theta
        return {
            'tau_motor': tau_motor,
            'tau_g': tau_g,
            'tau_d': tau_d,
            'power': tau_motor * self.s.omega,
        }

# ===================== VISUALISATION =====================
class Viewer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Bras robot 1‑axe – moteur de simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.big = pygame.font.SysFont("consolas", 24, bold=True)
        self.center = (WIDTH//2, HEIGHT//2 + 120)

    def draw_arm(self, theta: float, L: float):
        cx, cy = self.center
        x_tip = cx + int(L * PX_PER_M * math.cos(theta))
        y_tip = cy - int(L * PX_PER_M * math.sin(theta))

        # bras
        pygame.draw.line(self.screen, ARM_COLOR, (cx, cy), (x_tip, y_tip), 6)
        # pivot
        pygame.draw.circle(self.screen, PIVOT_COLOR, (cx, cy), 10)
        # masse en bout
        pygame.draw.circle(self.screen, MASS_COLOR, (x_tip, y_tip), 12)

    def draw_textblock(self, lines, x, y):
        for i, txt in enumerate(lines):
            surf = self.font.render(txt, True, TEXT_COLOR)
            self.screen.blit(surf, (x, y + 20*i))

    def draw_header(self, text, x, y):
        surf = self.big.render(text, True, TEXT_COLOR)
        self.screen.blit(surf, (x, y))

# ===================== BOUCLE PRINCIPALE =====================
def main():
    params = ArmParams(L=L, m_bras=M_BRAS, m_charge=M_CHARGE, g=G, b=B_DAMP)
    sim = ArmSim(params)
    view = Viewer()

    running = True
    physics_dt = 1.0 / PHYSICS_HZ
    physics_accum = 0.0

    last_stats = {'tau_motor': 0.0, 'tau_g': 0.0, 'tau_d': 0.0, 'power': 0.0}

    while running:
        # ----- Events -----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_r:
                    sim.reset()
                elif event.key == pygame.K_g:
                    sim.gravity_on = not sim.gravity_on
                elif event.key == pygame.K_m:
                    sim.use_accel_mode = not sim.use_accel_mode
                elif event.key == pygame.K_UP:
                    if sim.use_accel_mode:
                        sim.alpha_cmd += ALPHA_STEP
                    else:
                        sim.tau_manual = min(TAU_LIMIT, sim.tau_manual + TAU_STEP)
                elif event.key == pygame.K_DOWN:
                    if sim.use_accel_mode:
                        sim.alpha_cmd -= ALPHA_STEP
                    else:
                        sim.tau_manual = max(-TAU_LIMIT, sim.tau_manual - TAU_STEP)
                elif event.key == pygame.K_a:
                    sim.tau_manual = min(TAU_LIMIT, sim.tau_manual + TAU_STEP)
                    sim.use_accel_mode = False
                elif event.key == pygame.K_z:
                    sim.tau_manual = max(-TAU_LIMIT, sim.tau_manual - TAU_STEP)
                    sim.use_accel_mode = False

        # ----- Physics fixed‑step -----
        frame_dt = view.clock.tick(FPS_TARGET) / 1000.0
        physics_accum += frame_dt
        # éviter une spirale de la mort si le rendu lagge
        physics_accum = min(0.25, physics_accum)
        while physics_accum >= physics_dt:
            stats = sim.physics_step(physics_dt)
            if stats:
                last_stats = stats
            physics_accum -= physics_dt

        # ----- Render -----
        view.screen.fill(BACKGROUND)
        view.draw_arm(sim.s.theta, sim.p.L)

        mode = "ACCEL (alpha_cmd)" if sim.use_accel_mode else "COUPLE manuel"
        status = [
            f"Mode: {mode} | Pause: {'ON' if sim.paused else 'OFF'} | Gravité: {'ON' if sim.gravity_on else 'OFF'}",
        ]
        view.draw_header("Bras robot 1‑axe – moteur de simulation", 24, 20)
        view.draw_textblock(status, 24, 60)

        lines = [
            f"θ = {sim.s.theta: .3f} rad  ({math.degrees(sim.s.theta): .1f}°)",
            f"ω = {sim.s.omega: .3f} rad/s",
            f"α = {sim.s.alpha: .3f} rad/s²",
            f"alpha_cmd = {sim.alpha_cmd: .3f} rad/s²",
            f"τ_motor = {last_stats['tau_motor']: .3f} N·m (lim {TAU_LIMIT} N·m)",
            f"τ_g = {last_stats['tau_g']: .3f} N·m",
            f"τ_d = {last_stats['tau_d']: .3f} N·m",
            f"I = {sim.p.I: .4f} kg·m²",
            f"P = {last_stats['power']: .2f} W",
        ]
        view.draw_textblock(lines, 24, 96)

        help_lines = [
            "Contrôles :",
            " ↑ / ↓ : alpha_cmd ± (ou couple ± en mode manuel)",
            " A / Z  : couple ± et passe en mode manuel",
            " M : bascule mode ACCEL ↔ COUPLE",
            " G : gravité ON/OFF  |  P : pause  |  R : reset  |  ESC : quitter",
        ]
        view.draw_textblock(help_lines, 24, HEIGHT - 120)

        # barre horizontale zéro
        pygame.draw.line(view.screen, (80, 90, 100), (0, view.center[1]), (WIDTH, view.center[1]), 1)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
