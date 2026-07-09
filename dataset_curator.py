#!/usr/bin/env python3
"""
Dataset Curator: Quality Control and Curation for Image Crop Datasets
Detects duplicates, outliers, and mislabeled crops using embedding analysis.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class DatasetCurator:
    def __init__(self, embeddings, labels, image_paths, ids=None):
        """
        Initialize the curator with dataset embeddings, labels, and image paths.
        
        Parameters:
        - embeddings: numpy array of shape (N, D) containing crop embeddings
        - labels: list/array of shape (N,) containing class labels
        - image_paths: list/array of shape (N,) containing image file paths
        - ids: optional list of unique IDs for each crop. If None, indices are used.
        """
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.labels = np.array(labels)
        self.image_paths = np.array(image_paths)
        self.num_samples = len(embeddings)
        
        if ids is not None:
            self.ids = np.array(ids)
        else:
            self.ids = np.array([f"crop_{i:04d}" for i in range(self.num_samples)])
            
        # Normalize embeddings for cosine similarity calculations
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.norm_embeddings = self.embeddings / norms
        
        # Precompute similarity matrix
        self.sim_matrix = np.dot(self.norm_embeddings, self.norm_embeddings.T)
        
    def detect_duplicates(self, threshold=0.95):
        """
        Flag pairs of crops with cosine similarity above a threshold.
        Leaked duplicates inflate evaluation metrics.
        """
        duplicates = []
        flagged_indices = set()
        
        # Find pairs with similarity above threshold (upper triangle of similarity matrix)
        for i in range(self.num_samples):
            for j in range(i + 1, self.num_samples):
                sim = float(self.sim_matrix[i, j])
                if sim >= threshold:
                    duplicates.append({
                        "item_a": {
                            "index": int(i),
                            "id": str(self.ids[i]),
                            "label": str(self.labels[i]),
                            "path": str(self.image_paths[i])
                        },
                        "item_b": {
                            "index": int(j),
                            "id": str(self.ids[j]),
                            "label": str(self.labels[j]),
                            "path": str(self.image_paths[j])
                        },
                        "similarity": sim
                    })
                    flagged_indices.add(i)
                    flagged_indices.add(j)
                    
        return duplicates, flagged_indices

    def detect_outliers_centroid(self, threshold=0.85, quantile=None):
        """
        Flag crops far from their class centroid.
        Likely junk crops, bad bounding boxes, or blurred images.
        
        If quantile is provided (e.g. 0.05), flags the bottom X% of crops in each class.
        Otherwise, flags crops with similarity below the absolute threshold.
        """
        outliers = []
        flagged_indices = set()
        unique_classes = np.unique(self.labels)
        
        self.class_centroids = {}
        self.centroid_similarities = np.zeros(self.num_samples)
        
        for cls in unique_classes:
            class_mask = (self.labels == cls)
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
                
            # Compute centroid as the mean of normalized embeddings in this class
            centroid = np.mean(self.norm_embeddings[class_indices], axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            self.class_centroids[cls] = centroid
            
            # Compute similarity of each class member to the class centroid
            similarities = np.dot(self.norm_embeddings[class_indices], centroid)
            self.centroid_similarities[class_indices] = similarities
            
            # Determine threshold
            if quantile is not None:
                class_threshold = np.quantile(similarities, quantile)
            else:
                class_threshold = threshold
                
            # Flag outliers
            for idx, sim in zip(class_indices, similarities):
                if sim < class_threshold:
                    outliers.append({
                        "index": int(idx),
                        "id": str(self.ids[idx]),
                        "label": str(self.labels[idx]),
                        "path": str(self.image_paths[idx]),
                        "centroid_similarity": float(sim),
                        "class_threshold": float(class_threshold),
                        "reason": f"Low similarity ({sim:.3f}) to class '{cls}' centroid (threshold: {class_threshold:.3f})"
                    })
                    flagged_indices.add(idx)
                    
        return outliers, flagged_indices

    def detect_outliers_knn(self, k=5, threshold=0.8):
        """
        Flag crops with low mean similarity to their k-nearest neighbors in the same class.
        Useful when a class has multiple distinct sub-clusters, making a single centroid less representative.
        """
        outliers = []
        flagged_indices = set()
        unique_classes = np.unique(self.labels)
        
        self.knn_in_class_similarities = np.zeros(self.num_samples)
        
        for cls in unique_classes:
            class_mask = (self.labels == cls)
            class_indices = np.where(class_mask)[0]
            n_class_samples = len(class_indices)
            
            if n_class_samples <= 1:
                continue
                
            # Extract sub-similarity matrix for this class
            class_sims = self.sim_matrix[class_indices][:, class_indices]
            
            # For each sample, find top k neighbors of the same class (excluding itself)
            k_val = min(k, n_class_samples - 1)
            
            for local_idx, global_idx in enumerate(class_indices):
                # Copy similarity row and set self-similarity to -inf
                row = class_sims[local_idx].copy()
                row[local_idx] = -np.inf
                
                # Get the similarities of top k neighbors
                top_k_sims = np.partition(row, -k_val)[-k_val:]
                mean_sim = float(np.mean(top_k_sims))
                
                self.knn_in_class_similarities[global_idx] = mean_sim
                
                if mean_sim < threshold:
                    outliers.append({
                        "index": int(global_idx),
                        "id": str(self.ids[global_idx]),
                        "label": str(self.labels[global_idx]),
                        "path": str(self.image_paths[global_idx]),
                        "knn_similarity": mean_sim,
                        "reason": f"Low mean similarity ({mean_sim:.3f}) to {k_val} nearest class neighbors (threshold: {threshold:.3f})"
                    })
                    flagged_indices.add(global_idx)
                    
        return outliers, flagged_indices

    def detect_mislabels(self, k=5, own_label_threshold=0.4):
        """
        Flag crops whose nearest neighbors are mostly of a different class.
        Suspected wrong labels.
        
        If the fraction of neighbors with the crop's own label is below own_label_threshold,
        or if the majority class among neighbors differs from the current label, it is flagged.
        """
        mislabels = []
        flagged_indices = set()
        
        for i in range(self.num_samples):
            # Get similarities to all other samples, set self to -inf
            sims = self.sim_matrix[i].copy()
            sims[i] = -np.inf
            
            # Find indices of top k nearest neighbors
            neighbor_indices = np.argsort(sims)[-k:][::-1]
            neighbor_labels = self.labels[neighbor_indices]
            neighbor_sims = sims[neighbor_indices]
            
            # Count class frequencies
            classes, counts = np.unique(neighbor_labels, return_counts=True)
            class_counts = dict(zip(classes, counts))
            
            own_label = self.labels[i]
            own_count = class_counts.get(own_label, 0)
            own_fraction = own_count / k
            
            # Find majority class
            majority_label = classes[np.argmax(counts)]
            majority_count = class_counts[majority_label]
            majority_fraction = majority_count / k
            
            # Flag conditions:
            # 1. Majority label among neighbors differs from own label AND own fraction is low
            # 2. Or, own label fraction is extremely low (below threshold)
            is_flagged = False
            reason = ""
            
            if majority_label != own_label and majority_fraction >= 0.5:
                is_flagged = True
                reason = f"Majority of {k} nearest neighbors ({majority_fraction*100:.0f}%) are '{majority_label}', while current label is '{own_label}'."
            elif own_fraction < own_label_threshold:
                is_flagged = True
                reason = f"Only {own_fraction*100:.0f}% of {k} nearest neighbors share the label '{own_label}' (threshold: {own_label_threshold*100:.0f}%). Top class is '{majority_label}'."
                
            if is_flagged:
                # Format neighbor info for dashboard representation
                neighbors_info = []
                for idx, sim in zip(neighbor_indices, neighbor_sims):
                    neighbors_info.append({
                        "index": int(idx),
                        "id": str(self.ids[idx]),
                        "label": str(self.labels[idx]),
                        "similarity": float(sim),
                        "path": str(self.image_paths[idx])
                    })
                    
                mislabels.append({
                    "index": int(i),
                    "id": str(self.ids[i]),
                    "label": str(own_label),
                    "path": str(self.image_paths[i]),
                    "suggested_label": str(majority_label),
                    "own_label_fraction": own_fraction,
                    "majority_label_fraction": majority_fraction,
                    "reason": reason,
                    "neighbors": neighbors_info
                })
                flagged_indices.add(i)
                
        return mislabels, flagged_indices

    def run_all(self, duplicate_threshold=0.95, outlier_threshold_centroid=0.85, outlier_threshold_knn=0.8, mislabel_k=5, outlier_k=5):
        """
        Run all detection pipelines and return a structured dictionary of reports.
        """
        duplicates, dup_indices = self.detect_duplicates(threshold=duplicate_threshold)
        outliers_c, out_c_indices = self.detect_outliers_centroid(threshold=outlier_threshold_centroid)
        outliers_k, out_k_indices = self.detect_outliers_knn(k=outlier_k, threshold=outlier_threshold_knn)
        mislabels, mis_indices = self.detect_mislabels(k=mislabel_k)
        
        # Combine outlier methods
        outliers_all = []
        out_indices_all = out_c_indices.union(out_k_indices)
        
        # Helper lookup
        out_c_dict = {item["index"]: item for item in outliers_c}
        out_k_dict = {item["index"]: item for item in outliers_k}
        
        for idx in sorted(out_indices_all):
            c_info = out_c_dict.get(idx)
            k_info = out_k_dict.get(idx)
            
            centroid_sim = float(self.centroid_similarities[idx])
            knn_sim = float(self.knn_in_class_similarities[idx])
            
            reasons = []
            if c_info: reasons.append(f"Centroid outlier (sim: {centroid_sim:.3f})")
            if k_info: reasons.append(f"KNN outlier (sim: {knn_sim:.3f})")
            
            outliers_all.append({
                "index": int(idx),
                "id": str(self.ids[idx]),
                "label": str(self.labels[idx]),
                "path": str(self.image_paths[idx]),
                "centroid_similarity": centroid_sim,
                "knn_similarity": knn_sim,
                "reason": " & ".join(reasons)
            })
            
        # Complete summary list of items with their status flags
        items_summary = []
        for i in range(self.num_samples):
            is_dup = i in dup_indices
            is_out = i in out_indices_all
            is_mis = i in mis_indices
            
            # Find matching mislabel suggested class
            suggested = ""
            mis_reason = ""
            for m in mislabels:
                if m["index"] == i:
                    suggested = m["suggested_label"]
                    mis_reason = m["reason"]
                    break
                    
            items_summary.append({
                "index": int(i),
                "id": str(self.ids[i]),
                "label": str(self.labels[i]),
                "path": str(self.image_paths[i]),
                "centroid_similarity": float(self.centroid_similarities[i]),
                "knn_similarity": float(self.knn_in_class_similarities[i]),
                "is_duplicate": is_dup,
                "is_outlier": is_out,
                "is_mislabeled": is_mis,
                "suggested_label": suggested,
                "mislabel_reason": mis_reason
            })
            
        return {
            "summary": {
                "total_samples": self.num_samples,
                "duplicate_count": len(dup_indices),
                "duplicate_pairs_count": len(duplicates),
                "outlier_count": len(outliers_all),
                "mislabel_count": len(mislabels),
                "clean_count": self.num_samples - len(dup_indices.union(out_indices_all).union(mis_indices)),
                "duplicate_threshold": duplicate_threshold,
                "outlier_threshold_centroid": outlier_threshold_centroid,
                "outlier_threshold_knn": outlier_threshold_knn,
                "mislabel_k": mislabel_k
            },
            "duplicates": duplicates,
            "outliers": outliers_all,
            "mislabels": mislabels,
            "items": items_summary,
            "embeddings": self.norm_embeddings.tolist()
        }

def generate_demo_dataset(output_dir="demo_dataset"):
    """
    Generates a simulated dataset with crop images and synthetic cluster embeddings,
    including injected duplicates, outliers, and mislabeled items.
    """
    print(f"Generating synthetic crop dataset in '{output_dir}'...")
    
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    # 5 classes with specific visual styles
    class_styles = {
        "apple": {"color": (220, 40, 40), "shape": "circle", "label": "Apple"},
        "orange": {"color": (240, 130, 20), "shape": "circle", "label": "Orange"},
        "banana": {"color": (230, 210, 40), "shape": "banana", "label": "Banana"},
        "leaf": {"color": (40, 180, 80), "shape": "leaf", "label": "Leaf"},
        "weed": {"color": (120, 90, 60), "shape": "weed", "label": "Weed"}
    }
    
    classes_list = list(class_styles.keys())
    
    # Generate class centroids in embedding space (128-dimensional)
    np.random.seed(42)
    embedding_dim = 128
    class_centroids = {}
    for cls in classes_list:
        v = np.random.randn(embedding_dim)
        class_centroids[cls] = v / np.linalg.norm(v)
        
    num_clean_samples = 120
    
    embeddings = []
    labels = []
    image_paths = []
    ids = []
    
    # Help helper function to draw shapes
    def draw_crop(path, cls, noise_level=0, label_text=None, is_bad_box=False):
        img = Image.new('RGB', (100, 100), color=(240, 235, 225))
        draw = ImageDraw.Draw(img)
        style = class_styles[cls]
        color = style["color"]
        shape = style["shape"]
        
        # Add color noise
        if noise_level > 0:
            color = (
                max(0, min(255, int(color[0] + np.random.normal(0, noise_level)))),
                max(0, min(255, int(color[1] + np.random.normal(0, noise_level)))),
                max(0, min(255, int(color[2] + np.random.normal(0, noise_level)))),
            )
            
        # Draw shapes
        if is_bad_box:
            # Bad box: shape is cut off or very off center
            offset_x = np.random.randint(60, 90)
            offset_y = np.random.randint(60, 90)
            draw.rectangle([offset_x, offset_y, offset_x + 40, offset_y + 40], fill=color, outline=(30, 30, 30))
        elif shape == "circle":
            draw.ellipse([20, 20, 80, 80], fill=color, outline=(30, 30, 30), width=2)
            # Add a small stem
            draw.line([50, 10, 50, 20], fill=(60, 40, 20), width=3)
        elif shape == "banana":
            # Draw arc
            draw.arc([15, 20, 110, 80], start=30, end=180, fill=color, width=16)
        elif shape == "leaf":
            # Draw diamond/polygon leaf
            draw.polygon([(50, 15), (80, 50), (50, 85), (20, 50)], fill=color, outline=(30, 30, 30), width=2)
            draw.line([50, 15, 50, 85], fill=(30, 100, 50), width=2)
        elif shape == "weed":
            # Irregular star shape
            points = [(50,15), (60,40), (85,35), (70,60), (80,85), (50,75), (20,85), (30,60), (15,35), (40,40)]
            draw.polygon(points, fill=color, outline=(30, 30, 30), width=2)
            
        # If it's a junk crop (outlier), draw heavy noise over it
        if noise_level > 80:
            # Draw cross lines or messy boxes
            draw.line([0, 0, 100, 100], fill=(130, 130, 130), width=8)
            draw.line([0, 100, 100, 0], fill=(130, 130, 130), width=8)
            # Add gray salt and pepper
            pixels = img.load()
            for _ in range(300):
                x = np.random.randint(0, 100)
                y = np.random.randint(0, 100)
                pixels[x, y] = (128, 128, 128)
                
        # Draw small text for visual guidance
        try:
            draw.text((10, 80), label_text or cls.capitalize(), fill=(0,0,0))
        except Exception:
            pass
            
        img.save(path)

    # 1. Generate clean clustered samples
    print("Generating clean samples...")
    sample_idx = 0
    for i in range(num_clean_samples):
        cls = classes_list[i % len(classes_list)]
        img_path = f"crops/crop_{sample_idx:04d}.jpg"
        abs_img_path = os.path.join(output_dir, img_path)
        
        # Draw clean image
        draw_crop(abs_img_path, cls)
        
        # Embedding: centroid + noise (reduced to 0.03 for tight clusters)
        emb = class_centroids[cls] + np.random.normal(0, 0.03, embedding_dim)
        emb = emb / np.linalg.norm(emb)
        
        embeddings.append(emb)
        labels.append(cls)
        image_paths.append(img_path)
        ids.append(f"crop_{sample_idx:04d}")
        sample_idx += 1
        
    # 2. Inject exact and near-duplicates
    print("Injecting duplicates...")
    for i in range(6): # 6 pairs of duplicates
        # Target a random clean sample to copy
        target_idx = np.random.randint(0, num_clean_samples)
        src_cls = labels[target_idx]
        src_path = os.path.join(output_dir, image_paths[target_idx])
        
        # Pair A: Duplicate clone
        img_path_a = f"crops/crop_{sample_idx:04d}.jpg"
        abs_img_path_a = os.path.join(output_dir, img_path_a)
        # Load and save copy
        src_img = Image.open(src_path)
        src_img.save(abs_img_path_a)
        
        # Near duplicate embedding (extremely close, noise 0.001)
        emb_a = embeddings[target_idx] + np.random.normal(0, 0.001, embedding_dim)
        emb_a = emb_a / np.linalg.norm(emb_a)
        
        embeddings.append(emb_a)
        labels.append(src_cls)
        image_paths.append(img_path_a)
        ids.append(f"crop_{sample_idx:04d}")
        sample_idx += 1

    # 3. Inject outliers (Junk crops / Bad boxes)
    print("Injecting outliers...")
    # Junk Crop Outlier
    for i in range(4):
        cls = classes_list[i % len(classes_list)]
        img_path = f"crops/crop_{sample_idx:04d}.jpg"
        abs_img_path = os.path.join(output_dir, img_path)
        
        # Heavy noise / junk drawing
        draw_crop(abs_img_path, cls, noise_level=100, label_text="Junk")
        
        # Embedding is extremely noisy or random (far from centroid)
        emb = np.random.randn(embedding_dim)
        emb = emb / np.linalg.norm(emb)
        
        embeddings.append(emb)
        labels.append(cls)
        image_paths.append(img_path)
        ids.append(f"crop_{sample_idx:04d}")
        sample_idx += 1
        
    # Bad Bounding Box Outlier
    for i in range(4):
        cls = classes_list[i % len(classes_list)]
        img_path = f"crops/crop_{sample_idx:04d}.jpg"
        abs_img_path = os.path.join(output_dir, img_path)
        
        # Offset box drawing
        draw_crop(abs_img_path, cls, is_bad_box=True, label_text="BadBox")
        
        # Embedding is shifted halfway between centroid and random noise
        emb = class_centroids[cls] * 0.4 + np.random.normal(0, 0.35, embedding_dim)
        emb = emb / np.linalg.norm(emb)
        
        embeddings.append(emb)
        labels.append(cls)
        image_paths.append(img_path)
        ids.append(f"crop_{sample_idx:04d}")
        sample_idx += 1

    # 4. Inject mislabeled crops (mislabeled class)
    print("Injecting mislabeled crops...")
    # Generate apple crop, but label it as "orange"
    for i in range(5):
        # We take a source class (e.g. apple) and a wrong label (e.g. orange)
        src_cls = "apple" if i % 2 == 0 else "leaf"
        wrong_label = "orange" if i % 2 == 0 else "weed"
        
        img_path = f"crops/crop_{sample_idx:04d}.jpg"
        abs_img_path = os.path.join(output_dir, img_path)
        
        # Draw the visual representation of the SOURCE class (e.g. Apple)
        # But write the wrong text for visual confusion in the demo
        draw_crop(abs_img_path, src_cls, label_text=wrong_label.capitalize())
        
        # Embedding belongs to the SOURCE class cluster (so neighbors will be source class)
        emb = class_centroids[src_cls] + np.random.normal(0, 0.03, embedding_dim)
        emb = emb / np.linalg.norm(emb)
        
        embeddings.append(emb)
        labels.append(wrong_label) # Wrong label assigned!
        image_paths.append(img_path)
        ids.append(f"crop_{sample_idx:04d}")
        sample_idx += 1
        
    # Convert lists to array structure
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    image_paths = np.array(image_paths)
    ids = np.array(ids)
    
    # Save the input data for reference
    data_dict = {
        "labels": labels.tolist(),
        "image_paths": image_paths.tolist(),
        "ids": ids.tolist(),
        "embeddings": embeddings.tolist()
    }
    with open(os.path.join(output_dir, "dataset_data.json"), "w") as f:
        json.dump(data_dict, f)
        
    # Run the curator
    curator = DatasetCurator(embeddings, labels, image_paths, ids)
    report = curator.run_all()
    
    # Save report
    report_path = os.path.join(output_dir, "curation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    report_js_path = os.path.join(output_dir, "curation_report.js")
    with open(report_js_path, "w") as f:
        f.write(f"window.CURATION_REPORT = {json.dumps(report, indent=2)};\n")
        
    print(f"Dataset generated successfully! Curation report saved to '{report_path}' and '{report_js_path}'.")
    print(f"Summary: {report['summary']}")
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Curator - Quality Control for Crop Datasets")
    parser.add_argument("--embeddings", type=str, help="Path to JSON file containing embeddings list of shape (N, D)")
    parser.add_argument("--labels", type=str, help="Path to JSON file containing labels list of shape (N,)")
    parser.add_argument("--images", type=str, help="Path to JSON file containing image paths list of shape (N,)")
    parser.add_argument("--output", type=str, default="curation_report.json", help="Path to output curation report JSON")
    parser.add_argument("--dup-threshold", type=float, default=0.95, help="Cosine similarity threshold for duplicate detection")
    parser.add_argument("--out-threshold-centroid", type=float, default=0.85, help="Cosine similarity threshold for centroid outlier detection")
    parser.add_argument("--out-threshold-knn", type=float, default=0.8, help="Mean cosine similarity threshold for KNN outlier detection")
    parser.add_argument("--knn-k", type=int, default=5, help="Number of nearest neighbors to look at for mislabel & outlier detection")
    parser.add_argument("--generate-demo", action="store_true", help="Generate a synthetic demo dataset and report")
    parser.add_argument("--demo-dir", type=str, default="demo_dataset", help="Directory where the generated demo dataset should be stored")
    
    args = parser.parse_args()
    
    if args.generate_demo:
        generate_demo_dataset(args.demo_dir)
    elif args.embeddings and args.labels and args.images:
        # Load user data
        with open(args.embeddings, 'r') as f:
            embs = np.array(json.load(f))
        with open(args.labels, 'r') as f:
            lbls = json.load(f)
        with open(args.images, 'r') as f:
            imgs = json.load(f)
            
        print(f"Loaded dataset with {len(embs)} samples.")
        curator = DatasetCurator(embs, lbls, imgs)
        report = curator.run_all(
            duplicate_threshold=args.dup_threshold,
            outlier_threshold_centroid=args.out_threshold_centroid,
            outlier_threshold_knn=args.out_threshold_knn,
            mislabel_k=args.knn_k,
            outlier_k=args.knn_k
        )
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
            
        js_output = args.output.replace(".json", ".js") if args.output.endswith(".json") else args.output + ".js"
        with open(js_output, 'w') as f:
            f.write(f"window.CURATION_REPORT = {json.dumps(report, indent=2)};\n")
            
        print(f"Curation complete. Reports saved to {args.output} and {js_output}")
        print(f"Summary: {report['summary']}")
    else:
        parser.print_help()
        print("\nNote: To run a quick test, execute with: python3 dataset_curator.py --generate-demo")
