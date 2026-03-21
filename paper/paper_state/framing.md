# Framing Registry: Where to Search (GAINS)

**Last Updated**: 2026-03-16

---

## Locked Terminology

| Term | Definition | Context | Status |
|------|-----------|---------|--------|
| GAINS | Global Adaptive Inference-time Noise Scheduling | Method name | Locked |
| noise trajectory search | Modifying noise variables along denoising trajectory under fixed budget | Problem definition | Locked |
| local search operator | Per-timestep noise refinement mechanism ($\mathcal{L}_t$) | Framework component | Locked |
| global scheduler | Cross-timestep compute allocation policy ($\mathcal{G}$) | Framework component | Locked |
| offline profiling | Pre-compute per-step sensitivity from calibration runs | GAINS component | Locked |
| online control | Instance-adaptive early stopping via gain/variance feedback | GAINS component | Locked |
| NFE | Number of function evaluations (denoising network forward passes) | Budget unit | Locked |
| verifier | External quality metric $v(x_0, c)$ | Given, not trained | Locked |
| connectivity measure | Expected weight of tail point on score function | From OR reference paper | For OR framing |

## Venue-Specific Framing

| Aspect | Current (ML) | Target (OR) |
|--------|-------------|-------------|
| Core problem | Inference-time scaling for image generation | Computational budget allocation for simulation optimization |
| Audience | ML researchers | OR/OM researchers + practitioners |
| Application | Text-to-image, class-conditional generation | Stochastic simulation, digital twins, Monte Carlo methods |
| Value prop | Better images with fewer NFE | More efficient use of computational resources |
| Theory role | Supporting empirical claims | Central contribution |
