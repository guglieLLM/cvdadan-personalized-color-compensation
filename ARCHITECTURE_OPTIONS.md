# Architecture Options & Configuration Reference

> Documento tecnico di dettaglio. Per l'approccio effettivamente adottato, vedere il [README](README.md).

Questo documento elenca **tutte** le opzioni architetturali, le varianti di loss, e le strategie di normalizzazione disponibili nel codice, incluse quelle **non utilizzate** nella configurazione finale.

---

## 1. Modello: `CVDCompensationModelAdaIN`

L'unica architettura implementata. Encoder-decoder con condizionamento CVD-AdaIN a ogni livello di normalizzazione.

### Opzioni configurabili

| Parametro | Tipo | Default | **Valore scelto** | Alternative disponibili |
|-----------|------|---------|-------------------|------------------------|
| `y_preserving` | bool | `False` | **`True`** | `False` → modalità RGB-only (predice ΔRGB a 3 canali, rischio artefatti bianchi) |
| `delta_rgb_scale` | float | 0.9 | **0.9** | Fattore di scala per i delta crominanza in output |
| `use_skip_connection` | bool | `False` | **`False`** | `True` → usa `PLCFDecoderCVDWithSkip` (skip connection dallo stage 0 dell'encoder) |
| `stop_at_stage` | int | 2 | **2** | 0 (shallow, 96ch), 1 (medium, 192ch) |
| `pretrained_encoder` | bool | `True` | **`True`** | `False` → encoder random init |
| `freeze_encoder_except_adain` | bool | `True` | **`True`** | `False` → fine-tuning completo encoder |
| `target_resolution` | int | 256 | **256** | Qualsiasi potenza di 2 |
| `cvd_dim` | int | 3 | **3** | Dimensione del vettore profilo CVD |

### Y-Preserving vs RGB-Only

| Modalità | Canali output decoder | Comportamento |
|----------|----------------------|---------------|
| **Y-Preserving** (`True`) — SCELTO | 2 (ΔCb, ΔCr) | Luma Y' (BT.601) copiata dall'input, solo crominanza modificata |
| RGB-Only (`False`) | 3 (ΔR, ΔG, ΔB) | Delta diretto in RGB, luma non garantita |

### Decoder con Skip Connection

| Variante | Classe | Vantaggio | Svantaggio |
|----------|--------|-----------|------------|
| **Standard** — SCELTO | `PLCFDecoderCVD` | Architettura più semplice | — |
| Con Skip | `PLCFDecoderCVDWithSkip` | Feature ad alta risoluzione dallo stage 0 | Complessità aggiuntiva, miglioramento non significativo |

---

## 2. Encoder: `PLCFEncoderCVD`

Basato su **ConvNeXt-Tiny** (pretrained ImageNet-1k) con CVDAdaIN integrato in ogni LayerNorm.

| Parametro | Tipo | Default | **Valore scelto** | Note |
|-----------|------|---------|-------------------|------|
| `encoder_type` | str | `"convnext_tiny"` | **`"convnext_tiny"`** | Unico tipo implementato |
| `stop_at_stage` | int | 2 | **2** | Stage dims: 0→[96,64,64], 1→[192,32,32], 2→[384,16,16] |
| `pretrained` | bool | `True` | **`True`** | Pesi ImageNet-1k |
| `freeze_stem` | bool | `True` | **`True`** | Congela il patchify layer iniziale |
| `drop_path_rate` | float | 0.1 | **0.1** | Stochastic depth |

Struttura ConvNeXt-Tiny: 4 stage con depths `[3, 3, 9, 3]` e dims `[96, 192, 384, 768]`. Con `stop_at_stage=2` si usano i primi 3 stage (output: 384 canali, 16×16 per input 256×256).

---

## 3. Decoder: `PLCFDecoderCVD`

Upsampling progressivo con CVDAdaIN in ogni ChannelLayerNorm.

| Parametro | Tipo | Default | **Valore scelto** | Alternative |
|-----------|------|---------|-------------------|-------------|
| `upsample_mode` | str | `"nearest"` | **`"nearest"`** | `"bilinear"` |
| `output_channels` | int | dinamico | **2** (y_preserving) | 3 (RGB-only) |

---

## 4. Condizionamento: `CVDAdaIN`

Moduli di Adaptive Instance Normalization condizionati dal profilo CVD.

| Classe | Contesto | Formato tensore |
|--------|----------|-----------------|
| `CVDAdaINChannelLast` | Encoder (ConvNeXt) | `[B, H, W, C]` |
| `CVDAdaINChannelFirst` | Decoder (PLCF) | `[B, C, H, W]` |

Ogni modulo proietta il vettore CVD `[3]` → `[2C]` tramite un Linear layer, producendo parametri di scala `γ` e shift `β` che modulano la normalizzazione.

Punti di applicazione:
- **Encoder**: 15 blocchi ConvNeXt (1 CVDAdaIN per blocco) + 2 downsample
- **Decoder**: 9 punti di normalizzazione (1 bottleneck + 6 block norms + 1 block0 + 1 head)

---

## 5. Funzione di Loss: `CVDLoss`

Formula generale a 3 componenti con normalizzazione statica:

$$\mathcal{L} = \lambda_{\text{mse}} \cdot \frac{\text{MSE}_{a^*b^*}}{M_{\text{mse}}} + \lambda_{\Delta E} \cdot \frac{\Delta E_{00}}{M_{\Delta E}} + \lambda_{\text{ssim}} \cdot \frac{(1 - \text{MS-SSIM}_{\text{RGB}})}{M_{\text{ssim}}}$$

### Componenti

| Componente | Spazio | Range | Descrizione |
|------------|--------|-------|-------------|
| MSE a\*b\* | CIELAB | [0, ∞) | Errore quadratico medio solo sui canali crominanza a\*, b\* |
| ΔE2000 | CIELAB | [0, 100) | CIEDE2000 — distanza percettiva standard industriale |
| MS-SSIM | sRGB | [0, 1] | Multi-Scale Structural Similarity — preservazione struttura |

### Costanti di calibrazione M

Calcolate automaticamente sui primi `cvd_warmup_samples` (200) campioni di training. Servono a normalizzare le tre componenti sulla stessa scala prima della pesatura con i λ. Salvate in `calibration_constants_*.json`.

### Configurazioni loss generate

| Config | λ_mse | λ_ΔE | λ_ssim | Tipo | Stato |
|--------|-------|------|--------|------|-------|
| **`config_01_no_delta_e`** | **0.7** | **0.0** | **0.3** | **2 componenti** | **SCELTO** |
| `config_02_balanced` | 0.45 | 0.30 | 0.25 | 3 componenti | Non usato |
| `config_03_perceptual` | 0.40 | 0.35 | 0.25 | 3 componenti (ΔE focus) | Non usato |
| `config_04_conservative` | 0.32 | 0.36 | 0.32 | 3 componenti (stabile) | Non usato |

La scelta della configurazione a 2 componenti (senza ΔE2000 nella loss) è stata fatta perché:
- Il calcolo di ΔE2000 è costoso (conversione Lab per ogni batch)
- La ΔE00 viene comunque calcolata come **metrica di validazione** per monitorarne l'andamento
- La combinazione MSE a\*b\* + MS-SSIM RGB ha prodotto i migliori risultati sperimentali

### Feature avanzate (non usate)

| Feature | Flag | Descrizione | Stato |
|---------|------|-------------|-------|
| Severity Weighting | `cvd_severity_dynamic_weighting` | λ dinamico in base alla severità del CVD del campione | **OFF** |
| Edge-Aware Weighting | `cvd_edge_aware_weighting` | Peso spaziale basato su edge detection (Sobel) | **OFF** |

---

## 6. Normalizzazione Profilo CVD

Strategia **ibrida** (`hybrid_theta_global_cs_per_type`):

| Dimensione | Normalizzazione | Motivazione |
|------------|----------------|-------------|
| θ (theta) | **Globale** | Preserva la distinzione tra tipi CVD (Protan θ>5°, Deutan −30°<θ≤5°, Tritan θ≤−30°) |
| C (Confusion) | **Per-tipo CVD** | Distribuzioni di severità diverse per tipo |
| S (Scatter) | **Per-tipo CVD** | Distribuzioni di dispersione diverse per tipo |

Le statistiche sono calcolate sul dataset di training e salvate nel checkpoint sotto `profile_normalization`.

---

## 7. Training — Parametri completi

| Parametro | Valore scelto | Descrizione |
|-----------|--------------|-------------|
| `epochs` | 2500 (max) | Epoche massime |
| `patience` | 20 | Early stopping su Val ΔE00 |
| `min_epochs` | 20 | Minimo prima di attivare early stopping |
| `warmup_epochs` | 5 | LR warmup lineare |
| `learning_rate` | 3e-5 | LR decoder |
| `encoder_learning_rate` | 1e-4 | LR encoder (fine-tuning) |
| `lr_factor` | 0.7 | `ReduceLROnPlateau` decay factor |
| `lr_patience` | 15 | Epoche senza miglioramento prima del decay |
| `min_lr` | 5e-6 | LR minimo |
| `weight_decay` | 1e-4 | L2 regularization |
| `batch_size` | 32 | |
| `max_norm` | 0.5 | Gradient clipping |
| `use_amp` | `true` | Mixed precision |
| `amp_dtype` | `bfloat16` | Tipo AMP |
| `use_torch_compile` | `false` | `torch.compile()` disabilitato |
| `auto_resume` | `true` | Riprende da ultimo checkpoint |
| `seed` | 42 | Riproducibilità |
| `target_resolution` | 256 | Resize + CenterCrop |
| `FRACTION_IMAGES_TRAIN` | 0.15 | Frazione del dataset Places365 per training |
| `FRACTION_IMAGES_VAL` | 1.0 | Validazione completa |

---

## 8. Data Pipeline

| Parametro | Valore | Note |
|-----------|--------|------|
| Dataset sorgente | Places365 subsets | Immagini naturali diverse |
| Preprocessing | Resize → CenterCrop → ToTensor → Normalize ImageNet | Standard ConvNeXt |
| Normalizzazione input | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | ImageNet stats |
| Data augmentation | Nessuna | Solo resize + crop |
| `num_workers` | 4 | DataLoader workers |
| `prefetch_factor` | 2 | Prefetch per worker |
| `pin_memory` | `true` | CUDA pinned memory |
| `persistent_workers` | `true` | Workers persistenti |
