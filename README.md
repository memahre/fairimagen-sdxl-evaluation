# Stress-Testing Bias Mitigation in Text-to-Image Models
### Evaluating FairImagen on Stable Diffusion XL

This project evaluates [FairImagen](https://github.com/fuzihaofzh/FairImagen) (Fu et al., NeurIPS 2025), a post-hoc bias mitigation framework for text-to-image models, applied to Stable Diffusion XL. The original paper validated FairImagen primarily on Stable Diffusion 3. This project extends that evaluation to a different diffusion architecture (SDXL) across four custom prompt categories, examining whether gender fairness improvements generalize across architectures and prompt domains.

---

## Key Findings

- **Gender fairness improvements do not consistently generalize to SDXL.** Only female-stereotyped roles showed improvement (0.38 → 0.62). Authority and male-stereotyped roles showed no change under either condition.
- **Age fairness improved across all four categories**, including those where gender fairness was unchanged — suggesting FairPCA projection affects demographic attributes beyond its intended targets.
- **Prompt-image alignment (CLIPScore) remained stable** across baseline and FairImagen conditions, indicating semantic fidelity is preserved.
- **Hyperparameter sensitivity is binary by category** — within the tested range, effectiveness is determined by prompt category, not parameter tuning.

---

## Project Structure

```
├── COGS185_FinalProject_Code.ipynb   # Full evaluation pipeline
├── COGS185_FinalProject_Paper.pdf    # Final project paper
└── README.md
```

---

## Prompt Categories

| Category | Example Roles |
|---|---|
| Female-stereotyped | Teacher, hairdresser, secretary, baker |
| Male-stereotyped | Sheriff, farmer, firefighter, pilot |
| Authority / Leadership | CEO, senator, project manager, Supreme Court justice |
| Historically grounded | Pope, medieval blacksmith, Viking warrior, Shinto priest |

The historically grounded category serves as a control. FairImagen should not introduce demographic diversity for roles where specificity is culturally or historically grounded.

---

## Setup and Usage

**Environment:** Google Colab with GPU runtime (T4 or better recommended)

**Step 1 — Install dependencies**

Run cell 1 to install all required packages, then restart the runtime before continuing.

**Step 2 — Configure output path**

In cell 3, update `DRIVE_ROOT` to your own output directory:

```python
DRIVE_ROOT = Path("/content/drive/MyDrive/your_output_folder")
```

If using Google Drive, mount it before running cell 3:

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 3 — Run the pipeline**

Run cells in order. The notebook will:
1. Clone the FairImagen repository
2. Define prompt categories and save to JSON
3. Run baseline and FairImagen image generation for each category
4. Score each image using DeepFace (fairness) and CLIP (alignment)
5. Run a 2×2 hyperparameter sensitivity sweep
6. Aggregate results into CSVs and generate quantitative figures
7. Produce side-by-side qualitative comparison figures

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Gender fairness | Normalized deviation from equal gender representation across generated images (0 = one group only, 1 = perfectly balanced) |
| Age fairness | Same metric applied to age buckets (young / middle / old) |
| Race fairness | Same metric applied to racial categories detected by DeepFace |
| CLIPScore | Cosine similarity between prompt text embedding and image embedding, scaled by 2.5 (measures prompt-image semantic alignment) |

---

## Dependencies

Key packages installed in cell 1:

- `diffusers`, `transformers`, `accelerate`, `safetensors` — image generation
- `openai/CLIP` — CLIP-based alignment scoring
- `deepface` — demographic attribute classification
- `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn` — core ML and visualization

---

## Limitations

- 4 images per prompt (vs. 12 in the original paper) due to compute constraints
- Hyperparameter sweep covers only the upper range (hdim: 1750, 2000; enoise: 0.4, 0.6)
- MUSIQ perceptual quality metric unavailable due to library compatibility issues — image quality evaluated qualitatively
- No direct SD3 architecture comparison — results reflect SDXL behavior only

---

## Reference

Fu, Z., Brown, R., Shao, S., Rawal, K., Delaney, E., & Russell, C. (2025). FairImagen: Post-processing for bias mitigation in text-to-image models. *39th Conference on Neural Information Processing Systems (NeurIPS 2025)*. https://openreview.net/pdf?id=uaQWgFk2Pf

---
