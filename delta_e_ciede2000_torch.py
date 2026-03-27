"""
CIEDE2000 Color Difference - PyTorch Implementation
Versione NUMERICAMENTE STABILE per backpropagation con bf16/AMP.

Modifiche rispetto all'originale per evitare NaN nei gradienti:
1. safe_sqrt() con epsilon per evitare grad infiniti quando x→0
2. safe_atan2() con gradiente stabile
3. Clamp dei valori intermedi per evitare overflow
4. Epsilon in tutte le divisioni
"""

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# COSTANTI NUMERICHE per stabilità
# ═══════════════════════════════════════════════════════════════════════════════
EPS = 1e-6  # Epsilon per sqrt e divisioni
EPS_ATAN = 1e-8  # Epsilon più piccolo per atan2
GRAD_CLAMP = 1e4  # Limite per gradienti


def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """
    Sqrt numericamente stabile.
    Problema: sqrt(0) ha gradiente infinito → NaN
    Soluzione: sqrt(max(x, eps)) 
    """
    return torch.sqrt(torch.clamp(x, min=EPS))


def safe_pow7(x: torch.Tensor) -> torch.Tensor:
    """
    x^7 stabile - evita overflow/underflow.
    """
    # Clamp per evitare che valori estremi causino overflow
    x_clamped = torch.clamp(x, min=0, max=1000)
    return x_clamped ** 7


def rgb2xyz(rgb_image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """RGB → XYZ con gestione numerica stabile."""
    mt = torch.tensor([[0.4124, 0.3576, 0.1805], 
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9504]], 
                      dtype=rgb_image.dtype, device=device)
    
    # Clamp RGB in [0, 1] per sicurezza
    rgb_image = torch.clamp(rgb_image, 0, 1)
    
    mask1 = (rgb_image > 0.04045).float()
    mask1_no = 1 - mask1
    
    # Gamma expansion con stabilità
    temp_img = mask1 * (torch.clamp((rgb_image + 0.055) / 1.055, min=EPS) ** 2.4)
    temp_img = temp_img + mask1_no * (rgb_image / 12.92)
    temp_img = 100 * temp_img

    res = torch.matmul(mt, temp_img.permute(1, 0, 2, 3).contiguous().view(3, -1))
    res = res.view(3, rgb_image.size(0), rgb_image.size(2), rgb_image.size(3)).permute(1, 0, 2, 3)
    return res


def xyz_lab(xyz_image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """XYZ → LAB f(t) con gestione valori zero."""
    # Evita divisione per zero nei gradienti
    xyz_safe = torch.clamp(torch.abs(xyz_image), min=EPS)
    
    mask1 = (xyz_image > 0.008856).float()
    mask1_no = 1 - mask1
    
    # Cuberoot stabile
    res = mask1 * (xyz_safe ** (1.0 / 3.0))
    res = res + mask1_no * ((7.787 * xyz_safe) + (16.0 / 116.0))
    
    return res


def rgb2lab_diff(rgb_image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    RGB → CIELAB conversion per batch di immagini.
    Illuminante: D65 standard (Y = 100)
    """
    rgb_image = rgb_image.to(device)
    res = torch.zeros_like(rgb_image)
    xyz_image = rgb2xyz(rgb_image, device)
    
    # Reference white D65
    xn = 95.0489
    yn = 100.0
    zn = 108.8840
    
    x = xyz_image[:, 0, :, :]
    y = xyz_image[:, 1, :, :]
    z = xyz_image[:, 2, :, :]

    fx = xyz_lab(x / xn, device)
    fy = xyz_lab(y / yn, device)
    fz = xyz_lab(z / zn, device)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    res[:, 0, :, :] = L
    res[:, 1, :, :] = a
    res[:, 2, :, :] = b
  
    return res


def degrees(n: torch.Tensor) -> torch.Tensor:
    """Radianti → Gradi"""
    return n * (180.0 / np.pi)


def radians(n: torch.Tensor) -> torch.Tensor:
    """Gradi → Radianti"""
    return n * (np.pi / 180.0)


def safe_atan2_degrees(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    atan2 stabile che restituisce gradi in [0, 360).
    
    Problema: atan2(0, 0) è undefined e causa gradienti NaN.
    Soluzione: Aggiungi piccolo offset e usa where per casi speciali.
    """
    # Maschera per (x=0, y=0) - achromatic
    is_achromatic = (torch.abs(x) < EPS_ATAN) & (torch.abs(y) < EPS_ATAN)
    
    # Aggiungi piccolo offset per evitare atan2(0,0)
    x_safe = torch.where(is_achromatic, torch.ones_like(x) * EPS_ATAN, x)
    y_safe = torch.where(is_achromatic, torch.zeros_like(y), y)
    
    # Calcola angolo in radianti, poi converti in gradi
    angle_rad = torch.atan2(y_safe, x_safe)
    angle_deg = degrees(angle_rad)
    
    # Converti in range [0, 360)
    angle_deg = torch.where(angle_deg < 0, angle_deg + 360.0, angle_deg)
    
    # Per punti acromatici, restituisci 0
    angle_deg = torch.where(is_achromatic, torch.zeros_like(angle_deg), angle_deg)
    
    return angle_deg


def ciede2000_diff(lab1: torch.Tensor, lab2: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    CIEDE2000 metric - versione numericamente stabile.
    
    Input: lab1, lab2 - tensors (B, 3, H, W) in LAB color space
    Output: color distance map (B, H, W)
    
    Modifiche per stabilità numerica:
    - safe_sqrt() ovunque ci sia sqrt
    - Epsilon in tutte le divisioni
    - Clamp dei valori intermedi
    - Gestione esplicita casi acromatici
    """
    lab1 = lab1.to(device)
    lab2 = lab2.to(device)
    
    L1 = lab1[:, 0, :, :]
    A1 = lab1[:, 1, :, :]
    B1 = lab1[:, 2, :, :]
    L2 = lab2[:, 0, :, :]
    A2 = lab2[:, 1, :, :]
    B2 = lab2[:, 2, :, :]
    
    kL = 1.0
    kC = 1.0
    kH = 1.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 1: Calcola C1, C2 (chroma) con sqrt stabile
    # ═══════════════════════════════════════════════════════════════════════════
    C1 = safe_sqrt(A1 ** 2 + B1 ** 2)
    C2 = safe_sqrt(A2 ** 2 + B2 ** 2)
    
    # Maschera per punti acromatici (C ≈ 0)
    is_achromatic1 = C1 < EPS
    is_achromatic2 = C2 < EPS
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 2: Calcola a' (a prime) con fattore G
    # ═══════════════════════════════════════════════════════════════════════════
    aC1C2 = (C1 + C2) / 2.0
    aC1C2_pow7 = safe_pow7(aC1C2)
    pow25_7 = 25.0 ** 7  # Costante: 6103515625
    
    G = 0.5 * (1.0 - safe_sqrt(aC1C2_pow7 / (aC1C2_pow7 + pow25_7 + EPS)))
    
    a1P = (1.0 + G) * A1
    a2P = (1.0 + G) * A2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 3: Calcola C' (chroma prime) con sqrt stabile
    # ═══════════════════════════════════════════════════════════════════════════
    c1P = safe_sqrt(a1P ** 2 + B1 ** 2)
    c2P = safe_sqrt(a2P ** 2 + B2 ** 2)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 4: Calcola h' (hue angle) con atan2 stabile
    # ═══════════════════════════════════════════════════════════════════════════
    h1P = safe_atan2_degrees(B1, a1P)
    h2P = safe_atan2_degrees(B2, a2P)
    
    # Azzera hue per punti acromatici
    h1P = torch.where(is_achromatic1, torch.zeros_like(h1P), h1P)
    h2P = torch.where(is_achromatic2, torch.zeros_like(h2P), h2P)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 5: Calcola ΔL', ΔC', ΔH'
    # ═══════════════════════════════════════════════════════════════════════════
    dLP = L2 - L1
    dCP = c2P - c1P
    
    # Δh' con gestione wrap-around
    dhP = h2P - h1P
    dhP = torch.where(dhP > 180.0, dhP - 360.0, dhP)
    dhP = torch.where(dhP < -180.0, dhP + 360.0, dhP)
    
    # Se uno dei due è acromatico, Δh' = 0
    either_achromatic = is_achromatic1 | is_achromatic2
    dhP = torch.where(either_achromatic, torch.zeros_like(dhP), dhP)
    
    # ΔH' = 2 * sqrt(C1' * C2') * sin(Δh'/2)
    # Usa safe_sqrt e clamp per sin
    c1c2_prod = torch.clamp(c1P * c2P, min=0)
    dHP = 2.0 * safe_sqrt(c1c2_prod) * torch.sin(radians(dhP / 2.0))
    dHP = torch.where(either_achromatic, torch.zeros_like(dHP), dHP)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 6: Calcola medie L̄', C̄', H̄'
    # ═══════════════════════════════════════════════════════════════════════════
    aL = (L1 + L2) / 2.0
    aCP = (c1P + c2P) / 2.0
    
    # H̄' con gestione casi speciali
    h_sum = h1P + h2P
    h_diff_abs = torch.abs(h1P - h2P)
    
    # Caso 1: |h1' - h2'| <= 180 → media semplice
    # Caso 2: |h1' - h2'| > 180 e somma < 360 → media + 180
    # Caso 3: |h1' - h2'| > 180 e somma >= 360 → media - 180
    aHP = torch.where(
        h_diff_abs <= 180.0,
        h_sum / 2.0,
        torch.where(
            h_sum < 360.0,
            (h_sum + 360.0) / 2.0,
            (h_sum - 360.0) / 2.0
        )
    )
    
    # Se entrambi acromatici, H̄' = 0
    both_achromatic = is_achromatic1 & is_achromatic2
    aHP = torch.where(both_achromatic, torch.zeros_like(aHP), aHP)
    # Se solo uno acromatico, H̄' = hue del cromatico
    aHP = torch.where(is_achromatic1 & ~is_achromatic2, h2P, aHP)
    aHP = torch.where(~is_achromatic1 & is_achromatic2, h1P, aHP)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 7: Calcola T (hue weighting function)
    # ═══════════════════════════════════════════════════════════════════════════
    aHP_rad = radians(aHP)
    T = (1.0 
         - 0.17 * torch.cos(aHP_rad - radians(torch.tensor(30.0, device=device)))
         + 0.24 * torch.cos(2.0 * aHP_rad)
         + 0.32 * torch.cos(3.0 * aHP_rad + radians(torch.tensor(6.0, device=device)))
         - 0.20 * torch.cos(4.0 * aHP_rad - radians(torch.tensor(63.0, device=device))))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 8: Calcola SL, SC, SH, RT
    # ═══════════════════════════════════════════════════════════════════════════
    # SL
    aL_minus_50_sq = (aL - 50.0) ** 2
    sL = 1.0 + (0.015 * aL_minus_50_sq) / safe_sqrt(20.0 + aL_minus_50_sq)
    
    # SC
    sC = 1.0 + 0.045 * aCP
    
    # SH
    sH = 1.0 + 0.015 * aCP * T
    
    # RT (rotation term)
    dRO = 30.0 * torch.exp(-((aHP - 275.0) / 25.0) ** 2)
    aCP_pow7 = safe_pow7(aCP)
    rC = 2.0 * safe_sqrt(aCP_pow7 / (aCP_pow7 + pow25_7 + EPS))
    rT = -rC * torch.sin(radians(2.0 * dRO))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Step 9: Calcola ΔE00 finale
    # ═══════════════════════════════════════════════════════════════════════════
    # Termini normalizzati (con epsilon per evitare div by zero)
    term_L = dLP / (sL * kL + EPS)
    term_C = dCP / (sC * kC + EPS)
    term_H = dHP / (sH * kH + EPS)
    
    # Formula completa
    res_square = (term_L ** 2 
                  + term_C ** 2 
                  + term_H ** 2 
                  + rT * term_C * term_H)
    
    # Assicura che res_square >= 0 (può essere leggermente negativo per errori numerici)
    res_square = torch.clamp(res_square, min=0)
    
    # Sqrt finale stabile
    result = safe_sqrt(res_square)
    
    # Clamp finale per sicurezza (ΔE00 tipicamente < 100)
    result = torch.clamp(result, min=0, max=200)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# WRAPPER per compatibilità con l'architettura esistente
# ═══════════════════════════════════════════════════════════════════════════════

def delta_e_ciede2000_torch(Lab1: torch.Tensor, Lab2: torch.Tensor, 
                            kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> torch.Tensor:
    """
    Wrapper per compatibilità con l'interfaccia esistente.
    
    Args:
        Lab1: torch.Tensor di forma (B, H, W, 3) - primo set di colori LAB
        Lab2: torch.Tensor di forma (B, H, W, 3) - secondo set di colori LAB
        kL, kC, kH: Fattori parametrici (default = 1)
    
    Returns:
        delta_E: torch.Tensor di forma (B, H, W) - mappa delle differenze ΔE₀₀
    """
    device = Lab1.device
    original_dtype = Lab1.dtype
    
    # ═══════════════════════════════════════════════════════════════════════════
    # IMPORTANTE: Converti SEMPRE a float32 per il calcolo CIEDE2000
    # bf16 non ha abbastanza precisione per le operazioni trigonometriche
    # ═══════════════════════════════════════════════════════════════════════════
    Lab1_f32 = Lab1.float()
    Lab2_f32 = Lab2.float()
    
    # L'interfaccia esistente usa (B, H, W, 3), la funzione interna usa (B, 3, H, W)
    # Converti: (B, H, W, 3) -> (B, 3, H, W)
    lab1_bchw = Lab1_f32.permute(0, 3, 1, 2).contiguous()
    lab2_bchw = Lab2_f32.permute(0, 3, 1, 2).contiguous()
    
    # Chiama la funzione stabile
    result = ciede2000_diff(lab1_bchw, lab2_bchw, device)
    
    # Converti il risultato al dtype originale
    if original_dtype != torch.float32:
        result = result.to(original_dtype)
    
    return result
