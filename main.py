import data_collection
import data_preprocessing
import feature_selection
import model_training
import model_evaluation
import visualization
import variance_correlation_viz

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def run_data_collection():
    """Step 1: Data Collection"""
    print_header("STEP 1: DATA COLLECTION")
    collector = data_collection.GDCDataCollector()
    clinical_data = collector.collect_all_data()
    return clinical_data

def run_data_preprocessing():
    """Step 2: Data Preprocessing"""
    print_header("STEP 2: DATA PREPROCESSING")
    preprocessor = data_preprocessing.DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline()
    return processed_data

def run_feature_selection():
    """Step 3: Feature Selection"""
    print_header("STEP 3: FEATURE SELECTION")
    selector = feature_selection.FeatureSelector()
    final_data, selected_features = selector.feature_selection_pipeline()
    return final_data, selected_features

def run_model_training():
    """Step 4: Model Training"""
    print_header("STEP 4: MODEL TRAINING")
    trainer = model_training.ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.training_pipeline()
    return trainer

def run_model_evaluation():
    """Step 5: Model Evaluation"""
    print_header("STEP 5: MODEL EVALUATION")
    evaluator = model_evaluation.ModelEvaluator()
    metrics, comparison = evaluator.evaluation_pipeline()
    return evaluator, metrics, comparison

def run_visualization():
    """Step 6: Visualization"""
    print_header("STEP 6: VISUALIZATION")
    
    # Standard visualizations
    visualizer = visualization.Visualizer()
    visualizer.generate_all_plots()
    
    # Variance-correlation specific visualizations
    vc_visualizer = variance_correlation_viz.VarianceCorrelationVisualizer()
    vc_visualizer.generate_all_plots()
    
    return visualizer, vc_visualizer

def run_full_pipeline(skip_collection=False):
    """
    Run the complete pipeline
    """
    start_time = datetime.now()
    
    print_header("PDAC DETECTION PIPELINE")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {config.PROJECT_ID}")
    print(f"Output Directory: {config.RESULTS_DIR}")
    
    try:
        # Step 1: Data Collection (optional)
        if not skip_collection:
            clinical_data = run_data_collection()
        else:
            print_header("STEP 1: DATA COLLECTION (SKIPPED)")
            print("Using existing data files...")
        
        # Step 2: Data Preprocessing
        processed_data = run_data_preprocessing()
        
        # Step 3: Feature Selection
        final_data, selected_features = run_feature_selection()
        
        # Step 4: Model Training
        trainer = run_model_training()
        
        # Step 5: Model Evaluation
        evaluator, metrics, comparison = run_model_evaluation()
        
        # Step 6: Visualization
        visualizer, vc_visualizer = run_visualization()
        
        # Print final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print_header("PIPELINE EXECUTION COMPLETE")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {duration}")
        print(f"\nBest Model: {evaluator.best_model_name}")
        print(f"ROC-AUC: {metrics[evaluator.best_model_name]['roc_auc']:.4f}")
        print(f"F1 Score: {metrics[evaluator.best_model_name]['f1']:.4f}")
        print(f"\nSelected Features: {len(selected_features)}")
        print(f"\nResults saved to: {config.RESULTS_DIR}")
        print(f"Models saved to: {config.MODELS_DIR}")
        print(f"Visualizations saved to: {visualizer.figures_dir}")
        print("\n" + "="*70)
        
        return {
            'trainer': trainer,
            'evaluator': evaluator,
            'visualizer': visualizer,
            'vc_visualizer': vc_visualizer,
            'metrics': metrics,
            'comparison': comparison,
            'selected_features': selected_features,
            'duration': duration
        }
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR: Pipeline execution failed")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_single_step(step_name):
    """
    Run a single step of the pipeline
    """
    steps = {
        'collect': run_data_collection,
        'preprocess': run_data_preprocessing,
        'select': run_feature_selection,
        'train': run_model_training,
        'evaluate': run_model_evaluation,
        'visualize': run_visualization
    }
    
    if step_name not in steps:
        print(f"Error: Unknown step '{step_name}'")
        print(f"Available steps: {', '.join(steps.keys())}")
        return None
    
    print_header(f"RUNNING STEP: {step_name.upper()}")
    result = steps[step_name]()
    print_header(f"STEP COMPLETE: {step_name.upper()}")
    
    return result

def main():
    """
    Main entry point with command-line argument parsing
    """
    parser = argparse.ArgumentParser(
        description='PDAC Detection Pipeline - TCGA-PAAD Analysis'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['collect', 'preprocess', 'select', 'train', 'evaluate', 'visualize', 'all'],
        default='all',
        help='Pipeline step to run (default: all)'
    )
    
    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip data collection step (use existing data)'
    )
    
    args = parser.parse_args()
    
    if args.step == 'all':
        result = run_full_pipeline(skip_collection=args.skip_collection)
    else:
        result = run_single_step(args.step)
    
    if result is not None:
        print("\nExecution successful!")
        return 0
    else:
        print("\nExecution failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
