import argparse
import sys
import json
import os

from vyntri.dataset.engine import DatasetIntelligenceEngine
from vyntri.core.config import Config
from vyntri.core.pipeline import VyntriPipeline

def main():
    parser = argparse.ArgumentParser(description="Vyntri: Training-Less Vision Intelligence")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze Command
    parser_analyze = subparsers.add_parser("analyze", help="Analyze a dataset")
    parser_analyze.add_argument("dataset_path", type=str, help="Path to the dataset directory")

    # Train Command
    parser_train = subparsers.add_parser("train", help="Train a model on a dataset")
    parser_train.add_argument("dataset_path", type=str, help="Path to dataset folder")
    parser_train.add_argument("--save_path", type=str, default="model.pkl", help="Path to save trained model")
    parser_train.add_argument("--solver", type=str, default="anacp", choices=["anacp", "fly", "wisard"], help="Solver algorithm to use")

    # Predict Command
    parser_predict = subparsers.add_parser("predict", help="Predict class for an image")
    parser_predict.add_argument("image_path", type=str, help="Path to image")
    parser_predict.add_argument("model_path", type=str, help="Path to trained model file")

    # Update Command (Continual Learning)
    parser_update = subparsers.add_parser("update", help="Update model with new data (Continual Learning)")
    parser_update.add_argument("dataset_path", type=str, help="Path to new data")
    parser_update.add_argument("model_path", type=str, help="Path to existing model .pkl")

    args = parser.parse_args()

    if args.command == "analyze":
        if not os.path.isdir(args.dataset_path):
            print(f"Error: Dataset path '{args.dataset_path}' does not exist or is not a directory.")
            sys.exit(1)
        config = Config()
        engine = DatasetIntelligenceEngine(config)
        try:
            fingerprint = engine.analyze(args.dataset_path)
            print("\n=== Dataset Analysis Result ===")
            print(json.dumps(fingerprint, indent=2, default=str))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "train":
        if not os.path.isdir(args.dataset_path):
            print(f"Error: Dataset path '{args.dataset_path}' does not exist or is not a directory.")
            sys.exit(1)
        config = Config()
        pipeline = VyntriPipeline(config)
        # Use ContinualAnaCP by default for the 'train' command if we want to enable updates later?
        # Or we can swap the solver in the pipeline.
        # For Phase 6, let's force the pipeline to use ContinualAnaCP if we want extensibility.
        # But 'train' uses the pipeline which hardcodes 'AnalyticContrastiveProjection'.
        # We need to hack the pipeline or subclass it, OR just swap the solver class in pipeline.py
        # For now, let's keep 'train' as is, and 'update' will handle the conversion/usage.
        try:
            result = pipeline.run(args.dataset_path, solver_type=args.solver)
            
            # --- PHASE 6 MODIFICATION: Upgrade to Continual Wrapper if possible or save as is ---
            # Actually, to update, we need the G and C matrices which standard AnaCP doesn't save (it computes them locally).
            # We need to modify AnaCP or Pipeline to use ContinualAnaCP.
            # HACK: Let's modify pipeline.py to use ContinualAnaCP by default in the next step.
            # Assuming pipeline uses ContinualAnaCP now:
            
            print("\n=== Training Complete ===")
            print(f"Backbone: {result['selected_backbone']}")
            print(f"Samples: {result['num_samples']}")
            
            from vyntri.core.io import save_model
            save_model(result, args.save_path)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


    elif args.command == "update":
        from vyntri.core.io import load_model, save_model
        from vyntri.solvers.continual import ContinualAnaCP
        from vyntri.core.pipeline import VyntriPipeline 
        from vyntri.core.config import Config
        import numpy as np
        import torch
        
        if not os.path.isdir(args.dataset_path):
            print(f"Error: Dataset path '{args.dataset_path}' does not exist or is not a directory.")
            sys.exit(1)
        if not os.path.isfile(args.model_path):
            print(f"Error: Model file '{args.model_path}' not found.")
            sys.exit(1)
        
        try:
            print(f"Loading model from {args.model_path}...")
            model_ctx = load_model(args.model_path)
            
            solver = model_ctx['solver']
            # Check if solver is capable of update
            if not hasattr(solver, 'update') or not hasattr(solver, 'G') or solver.G is None:
                print("Error: The loaded model does not support continual updates (G/C matrices missing).")
                print("Please retrain the model with the latest version of Vyntri.")
                sys.exit(1)
                
            print(f"Updating model with data from {args.dataset_path}...")
            
            # Setup pipeline for feature extraction
            config = Config()
            pipeline = VyntriPipeline(config)
            
            # Manually inject the loaded backbone into the pipeline
            pipeline.model_backbone = model_ctx['backbone']
            pipeline.model_backbone.to(pipeline.device)
            # We don't need to load_backbone again because we have it from model_ctx
            
            # Extract features
            # Note: _extract_features expects the backbone to be set in self.model_backbone
            features, labels, class_names = pipeline._extract_features(args.dataset_path)
            
            if len(features) == 0:
                print("No valid images found in the update dataset.")
                sys.exit(1)

            print(f"Extracted features for {len(labels)} new samples.")
            
            # Map labels (which are ints 0..K relative to this batch) to real class names
            real_labels = np.array([class_names[i] for i in labels])
            
            # Update the solver
            print("Running incremental update (Recursive Least Squares)...")
            solver.update(features, real_labels)
            
            # Save back
            print("Saving updated model...")
            
            # Reconstruct the result dictionary expected by save_model
            # We need to preserve original fingerprint/stats but maybe update sample count?
            
            # Update class names list in the result to include new ones
            # solver.classes_ now contains all unique classes seen so far
            final_class_names = solver.classes_.tolist()
            
            updated_result = {
                "selected_backbone": model_ctx['backbone_name'],
                "solver": solver,
                "fingerprint": model_ctx.get('fingerprint'), # Keep original fingerprint for reference
                "num_samples": "unknown (updated)", # valid metric but tracking it requires the solver to track n_seen
                "class_names": final_class_names
            }
            
            save_model(updated_result, args.model_path)
            print(f"Successfully updated model at {args.model_path}")
            print(f"Total classes now: {len(final_class_names)}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.command == "predict":
        from vyntri.core.io import load_model
        from vyntri.backbones.loader import get_transform
        from PIL import Image
        import torch
        import numpy as np
        
        if not os.path.isfile(args.image_path):
            print(f"Error: Image file '{args.image_path}' not found.")
            sys.exit(1)
        if not os.path.isfile(args.model_path):
            print(f"Error: Model file '{args.model_path}' not found.")
            sys.exit(1)
        
        try:
            print(f"Loading model from {args.model_path}...")
            model_ctx = load_model(args.model_path)
            backbone = model_ctx['backbone']
            solver = model_ctx['solver']
            
            # Setup inference
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            backbone.to(device)
            transform = get_transform()
            
            # Process Image
            img = Image.open(args.image_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = backbone(img_t).cpu().numpy().flatten()
            
            # Predict
            pred_label = solver.predict(feat.reshape(1, -1))[0]
            print(f"Prediction: {pred_label}")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
