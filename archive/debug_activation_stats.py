"""
Debug script to check activation statistics across datasets and layers.
"""

from pathlib import Path
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "outputs"
MODEL_PREFIX = "Llama-3.3-70B-Instruct"

def check_activations():
    # Find all MC activation files
    mc_files = sorted(OUTPUT_DIR.glob(f"{MODEL_PREFIX}_*_mc_activations.npz"))

    print(f"Found {len(mc_files)} MC activation files\n")

    for npz_path in mc_files:
        dataset = npz_path.stem.replace(f"{MODEL_PREFIX}_", "").replace("_mc_activations", "")
        print(f"{'='*70}")
        print(f"Dataset: {dataset}")
        print(f"File: {npz_path.name}")
        print(f"{'='*70}")

        data = np.load(npz_path)

        # Get layer keys
        layer_keys = sorted([k for k in data.files if k.startswith("layer_")],
                           key=lambda x: int(x.split("_")[1]))

        print(f"\nLayers: {len(layer_keys)}")
        print(f"Keys: {data.files[:5]}... (showing first 5)\n")

        # Sample a few layers: first, middle, last
        sample_layers = [0, len(layer_keys)//4, len(layer_keys)//2, 3*len(layer_keys)//4, len(layer_keys)-1]
        sample_layers = sorted(set(sample_layers))

        print(f"{'Layer':<8} {'dtype':<10} {'shape':<20} {'min':<12} {'max':<12} {'mean':<12} {'std':<12} {'zeros%':<8}")
        print("-" * 100)

        for idx in sample_layers:
            key = layer_keys[idx]
            arr = data[key]
            layer_num = int(key.split("_")[1])

            zeros_pct = 100 * (arr == 0).sum() / arr.size

            print(f"{layer_num:<8} {str(arr.dtype):<10} {str(arr.shape):<20} {arr.min():<12.2e} {arr.max():<12.2e} {arr.mean():<12.2e} {arr.std():<12.2e} {zeros_pct:<8.1f}")

        # Check inter-sample vs per-dimension variance for a middle layer
        mid_key = layer_keys[len(layer_keys)//2]
        mid_arr = data[mid_key].astype(np.float32)  # Cast for precision
        mid_layer = int(mid_key.split("_")[1])

        print(f"\n--- Variance Analysis (Layer {mid_layer}) ---")
        print(f"Shape: {mid_arr.shape} (samples, dimensions)")

        # Inter-sample variance: how much do samples differ from each other?
        # Compute variance across samples for each dimension, then average
        per_dim_var = np.var(mid_arr, axis=0)  # variance across samples, per dimension
        mean_inter_sample_var = per_dim_var.mean()

        # Per-dimension variance: how much does each dimension vary?
        # This tells us if variance is concentrated in few dimensions
        n_dims_with_variance = (per_dim_var > 1e-10).sum()
        n_dims_total = mid_arr.shape[1]

        # What fraction of dimensions have >1% of max variance?
        max_var = per_dim_var.max()
        n_dims_significant = (per_dim_var > 0.01 * max_var).sum()

        # Top 10 dimensions by variance
        top_var_dims = np.argsort(per_dim_var)[-10:][::-1]
        top_var_values = per_dim_var[top_var_dims]

        print(f"Mean inter-sample variance per dim: {mean_inter_sample_var:.2e}")
        print(f"Dimensions with any variance (>1e-10): {n_dims_with_variance}/{n_dims_total} ({100*n_dims_with_variance/n_dims_total:.1f}%)")
        print(f"Dimensions with >1% of max variance: {n_dims_significant}/{n_dims_total} ({100*n_dims_significant/n_dims_total:.1f}%)")
        print(f"Max per-dim variance: {max_var:.2e}")
        print(f"Top 10 dims by variance: {top_var_dims.tolist()}")
        print(f"Their variances: {[f'{v:.2e}' for v in top_var_values]}")

        # Sample-to-sample similarity: pick 5 random pairs and compute cosine similarity
        n_samples = mid_arr.shape[0]
        if n_samples >= 10:
            rng = np.random.RandomState(42)
            pair_sims = []
            for _ in range(10):
                i, j = rng.choice(n_samples, 2, replace=False)
                a, b = mid_arr[i], mid_arr[j]
                cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                pair_sims.append(cos_sim)
            print(f"Random sample-pair cosine similarities: mean={np.mean(pair_sims):.4f}, std={np.std(pair_sims):.4f}")
            print(f"  (High similarity = samples are nearly identical)")

        # Check for problematic patterns
        print(f"\nChecking all layers for issues...")
        issues = []
        for key in layer_keys:
            arr = data[key]
            layer_num = int(key.split("_")[1])

            # Check for tiny values
            if arr.std() < 1e-6:
                issues.append(f"  Layer {layer_num}: very small std ({arr.std():.2e})")

            # Check for all zeros
            if (arr == 0).all():
                issues.append(f"  Layer {layer_num}: ALL ZEROS")

            # Check for NaN/Inf
            if not np.isfinite(arr).all():
                n_nan = np.isnan(arr).sum()
                n_inf = np.isinf(arr).sum()
                issues.append(f"  Layer {layer_num}: {n_nan} NaN, {n_inf} Inf")

        if issues:
            print("ISSUES FOUND:")
            for issue in issues[:20]:  # Limit output
                print(issue)
            if len(issues) > 20:
                print(f"  ... and {len(issues) - 20} more")
        else:
            print("No issues found.")

        print()


if __name__ == "__main__":
    check_activations()
