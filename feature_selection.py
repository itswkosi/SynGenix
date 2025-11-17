import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr, pearsonr
import config

class FeatureSelector:
    """
    Selects most informative features for PDAC detection using variance and correlation
    """
    
    def __init__(self):
        self.selected_features = None
        self.feature_scores = None
        self.variance_selector = None
        self.variance_scores = {}
        self.correlation_scores = {}
        self.feature_correlations = None
    
    def load_processed_data(self, filepath=None):
        """
        Load preprocessed data
        """
        if filepath is None:
            filepath = config.INTEGRATED_DATA_FILE
        
        print(f"Loading processed data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {df.shape[0]} samples with {df.shape[1]} features")
        
        return df
    
    def prepare_data(self, df, target_col='is_tumor'):
        """
        Separate features and target
        """
        # Remove ID columns
        id_cols = ['case_id', 'submitter_id']
        feature_cols = [col for col in df.columns if col not in id_cols + [target_col]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"\nFeatures: {X.shape[1]}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def calculate_feature_variance(self, X):
        """
        Calculate variance for each feature
        """
        print("\n" + "="*70)
        print("VARIANCE ANALYSIS")
        print("="*70)
        
        # Calculate variance for each feature
        variances = X.var()
        
        # Create variance dataframe
        variance_df = pd.DataFrame({
            'feature': X.columns,
            'variance': variances,
            'std': X.std()
        }).sort_values('variance', ascending=False)
        
        print(f"\nVariance statistics:")
        print(f"  Mean variance: {variances.mean():.4f}")
        print(f"  Median variance: {variances.median():.4f}")
        print(f"  Std variance: {variances.std():.4f}")
        print(f"  Min variance: {variances.min():.4f}")
        print(f"  Max variance: {variances.max():.4f}")
        
        # Calculate percentiles
        percentile_value = np.percentile(variances, config.VARIANCE_PERCENTILE)
        print(f"\n{config.VARIANCE_PERCENTILE}th percentile variance: {percentile_value:.4f}")
        
        print(f"\nTop 15 features by variance:")
        print(variance_df.head(15).to_string(index=False))
        
        self.variance_scores = variance_df
        
        return variance_df
    
    def variance_threshold_selection(self, X, threshold=None):
        """
        Select features above variance threshold
        """
        if threshold is None:
            threshold = config.VARIANCE_THRESHOLD_SELECTION
        
        print(f"\nApplying variance threshold: {threshold}")
        
        # Use sklearn's VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        print(f"Features selected: {len(selected_features)} / {X.shape[1]}")
        print(f"Features removed: {X.shape[1] - len(selected_features)}")
        
        self.variance_selector = selector
        
        return selected_features, X[selected_features]
    
    def variance_percentile_selection(self, X, percentile=None):
        """
        Select top N% features by variance
        """
        if percentile is None:
            percentile = config.VARIANCE_PERCENTILE
        
        print(f"\nSelecting top {percentile}th percentile features by variance...")
        
        variances = X.var()
        threshold = np.percentile(variances, percentile)
        
        selected_features = variances[variances >= threshold].index.tolist()
        
        print(f"Features selected: {len(selected_features)} / {X.shape[1]}")
        
        return selected_features, X[selected_features]
    
    def calculate_correlation_with_target(self, X, y, method='pearson'):
        """
        Calculate correlation of each feature with target variable
        """
        print("\n" + "="*70)
        print("CORRELATION WITH TARGET (PDAC STATUS)")
        print("="*70)
        
        correlations = []
        p_values = []
        
        for col in X.columns:
            if method == 'pearson':
                corr, p_val = pearsonr(X[col], y)
            elif method == 'spearman':
                corr, p_val = spearmanr(X[col], y)
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            correlations.append(corr)
            p_values.append(p_val)
        
        # Create correlation dataframe
        corr_df = pd.DataFrame({
            'feature': X.columns,
            'correlation': correlations,
            'abs_correlation': np.abs(correlations),
            'p_value': p_values,
            'significant': np.array(p_values) < 0.05
        }).sort_values('abs_correlation', ascending=False)
        
        print(f"\nCorrelation statistics ({method}):")
        print(f"  Mean |correlation|: {np.mean(np.abs(correlations)):.4f}")
        print(f"  Median |correlation|: {np.median(np.abs(correlations)):.4f}")
        print(f"  Max |correlation|: {np.max(np.abs(correlations)):.4f}")
        print(f"  Significant features (p<0.05): {corr_df['significant'].sum()}")
        
        print(f"\nTop 15 features by absolute correlation with PDAC:")
        print(corr_df.head(15)[['feature', 'correlation', 'p_value']].to_string(index=False))
        
        self.correlation_scores = corr_df
        
        return corr_df
    
    def correlation_target_selection(self, X, y, threshold=None, method='pearson'):
        """
        Select features with significant correlation to target
        """
        if threshold is None:
            threshold = config.CORRELATION_WITH_TARGET_THRESHOLD
        
        print(f"\nSelecting features with |correlation| > {threshold}...")
        
        corr_df = self.calculate_correlation_with_target(X, y, method)
        
        # Select features above threshold AND statistically significant
        selected_features = corr_df[
            (corr_df['abs_correlation'] >= threshold) & 
            (corr_df['significant'] == True)
        ]['feature'].tolist()
        
        print(f"Features selected: {len(selected_features)}")
        
        return selected_features, X[selected_features]
    
    def calculate_feature_correlation_matrix(self, X):
        """
        Calculate correlation matrix between features
        """
        print("\n" + "="*70)
        print("FEATURE INTER-CORRELATION ANALYSIS")
        print("="*70)
        
        # Calculate correlation matrix
        corr_matrix = X.corr(method=config.CORRELATION_METHOD).abs()
        
        # Get upper triangle (avoid duplicates)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for column in upper_triangle.columns:
            high_corr = upper_triangle[column][upper_triangle[column] > config.FEATURE_CORRELATION_THRESHOLD]
            for idx, corr_val in high_corr.items():
                high_corr_pairs.append({
                    'feature1': column,
                    'feature2': idx,
                    'correlation': corr_val
                })
        
        high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
        
        print(f"\nHighly correlated feature pairs (>{config.FEATURE_CORRELATION_THRESHOLD}):")
        print(f"  Found {len(high_corr_df)} pairs")
        
        if len(high_corr_df) > 0:
            print(f"\nTop 10 correlated pairs:")
            print(high_corr_df.head(10).to_string(index=False))
        
        self.feature_correlations = corr_matrix
        
        return corr_matrix, high_corr_df
    
    def remove_correlated_features(self, X, y, threshold=None):
        """
        Remove highly correlated features, keeping the one with higher target correlation
        """
        if threshold is None:
            threshold = config.FEATURE_CORRELATION_THRESHOLD
        
        print(f"\nRemoving features with inter-correlation > {threshold}...")
        
        # Calculate feature correlation matrix
        corr_matrix = X.corr(method=config.CORRELATION_METHOD).abs()
        
        # Calculate correlation with target
        target_corr = X.corrwith(y, method=config.CORRELATION_METHOD).abs()
        
        # Find features to drop
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = set()
        
        for column in upper_triangle.columns:
            high_corr_features = upper_triangle[column][upper_triangle[column] > threshold].index
            
            for corr_feature in high_corr_features:
                # Keep feature with higher target correlation
                if target_corr[column] >= target_corr[corr_feature]:
                    to_drop.add(corr_feature)
                else:
                    to_drop.add(column)
        
        to_drop = list(to_drop)
        
        print(f"Features to remove: {len(to_drop)}")
        if len(to_drop) > 0 and len(to_drop) <= 20:
            print(f"Removed features: {', '.join(to_drop)}")
        
        # Remove correlated features
        X_reduced = X.drop(columns=to_drop)
        selected_features = X_reduced.columns.tolist()
        
        print(f"Remaining features: {len(selected_features)}")
        
        return selected_features, X_reduced
    
    def variance_correlation_pipeline(self, X, y):
        """
        Combined variance and correlation-based selection
        """
        print("\n" + "="*70)
        print("VARIANCE-CORRELATION FEATURE SELECTION PIPELINE")
        print("="*70)
        
        original_features = X.shape[1]
        print(f"\nStarting with {original_features} features")
        
        # Step 1: Calculate variance for all features
        variance_df = self.calculate_feature_variance(X)
        
        # Step 2: Remove very low variance features (baseline threshold)
        low_var_features, X_var = self.variance_threshold_selection(
            X, threshold=config.VARIANCE_THRESHOLD
        )
        
        # Step 3: Calculate correlation with target
        corr_df = self.calculate_correlation_with_target(
            X_var, y, method=config.CORRELATION_METHOD
        )
        
        # Step 4: Select features by target correlation
        corr_features, X_corr = self.correlation_target_selection(
            X_var, y, threshold=config.CORRELATION_WITH_TARGET_THRESHOLD
        )
        
        # Step 5: Analyze inter-feature correlation
        corr_matrix, high_corr_pairs = self.calculate_feature_correlation_matrix(X_corr)
        
        # Step 6: Remove highly correlated features
        final_features, X_final = self.remove_correlated_features(
            X_corr, y, threshold=config.FEATURE_CORRELATION_THRESHOLD
        )
        
        # Step 7: Select top features by variance if still too many
        if len(final_features) > config.N_TOP_FEATURES:
            print(f"\nReducing to top {config.N_TOP_FEATURES} features by variance...")
            variances = X_final.var().sort_values(ascending=False)
            final_features = variances.head(config.N_TOP_FEATURES).index.tolist()
            X_final = X_final[final_features]
        
        print("\n" + "="*70)
        print("FEATURE SELECTION SUMMARY")
        print("="*70)
        print(f"Original features: {original_features}")
        print(f"After variance filter: {len(low_var_features)}")
        print(f"After correlation filter: {len(corr_features)}")
        print(f"After removing redundancy: {len(final_features)}")
        print(f"Final selected features: {len(final_features)}")
        
        self.selected_features = final_features
        
        return final_features, X_final
    
    def create_feature_ranking(self, X, y):
        """
        Create comprehensive feature ranking combining variance and correlation
        """
        print("\nCreating feature ranking...")
        
        # Get variance scores
        variances = X.var()
        
        # Get correlation with target
        target_corr = X.corrwith(y, method=config.CORRELATION_METHOD).abs()
        
        # Normalize to 0-1 scale
        variance_norm = (variances - variances.min()) / (variances.max() - variances.min())
        corr_norm = target_corr
        
        # Combined score (weighted average)
        variance_weight = 0.3
        correlation_weight = 0.7
        
        combined_score = (variance_weight * variance_norm + 
                         correlation_weight * corr_norm)
        
        # Create ranking dataframe
        ranking_df = pd.DataFrame({
            'feature': X.columns,
            'variance': variances,
            'variance_norm': variance_norm,
            'target_correlation': target_corr,
            'combined_score': combined_score,
            'selected': X.columns.isin(self.selected_features)
        }).sort_values('combined_score', ascending=False)
        
        print(f"\nTop 20 features by combined score:")
        print(ranking_df.head(20)[['feature', 'variance', 'target_correlation', 
                                    'combined_score', 'selected']].to_string(index=False))
        
        return ranking_df
    
    def save_results(self, ranking_df):
        """
        Save feature selection results
        """
        output_path = config.FEATURE_IMPORTANCE_FILE
        ranking_df.to_csv(output_path, index=False)
        print(f"\nFeature selection results saved to: {output_path}")
        
        # Save variance and correlation details
        if hasattr(self, 'variance_scores') and isinstance(self.variance_scores, pd.DataFrame):
            variance_path = output_path.replace('.csv', '_variance.csv')
            self.variance_scores.to_csv(variance_path, index=False)
            print(f"Variance scores saved to: {variance_path}")
        
        if hasattr(self, 'correlation_scores') and isinstance(self.correlation_scores, pd.DataFrame):
            corr_path = output_path.replace('.csv', '_correlation.csv')
            self.correlation_scores.to_csv(corr_path, index=False)
            print(f"Correlation scores saved to: {corr_path}")
    
    def create_final_dataset(self, df, selected_features):
        """
        Create final dataset with selected features
        """
        print(f"\nCreating final dataset with {len(selected_features)} features...")
        
        # Include target and ID columns
        final_cols = ['case_id', 'is_tumor'] + selected_features
        
        # Some features might not exist, filter them
        available_cols = [col for col in final_cols if col in df.columns]
        final_df = df[available_cols]
        
        print(f"Final dataset shape: {final_df.shape}")
        
        return final_df
    
    def feature_selection_pipeline(self):
        """
        Complete feature selection pipeline
        """
        print("="*70)
        print("VARIANCE-CORRELATION FEATURE SELECTION PIPELINE")
        print("="*70)
        
        # Step 1: Load processed data
        df = self.load_processed_data()
        
        # Step 2: Prepare data
        X, y, feature_names = self.prepare_data(df)
        
        # Step 3: Variance-correlation selection
        selected_features, X_selected = self.variance_correlation_pipeline(X, y)
        
        # Step 4: Create feature ranking
        ranking_df = self.create_feature_ranking(X_selected, y)
        
        # Step 5: Save results
        self.save_results(ranking_df)
        
        # Step 6: Create final dataset
        final_df = self.create_final_dataset(df, selected_features)
        
        # Save final dataset
        output_path = config.INTEGRATED_DATA_FILE.replace('.csv', '_selected.csv')
        final_df.to_csv(output_path, index=False)
        print(f"\nFinal dataset saved to: {output_path}")
        
        print("\n" + "="*70)
        print("FEATURE SELECTION COMPLETE")
        print("="*70)
        
        return final_df, selected_features


def main():
    """
    Main execution function
    """
    selector = FeatureSelector()
    final_data, selected_features = selector.feature_selection_pipeline()
    
    print("\n" + "="*70)
    print("SELECTED FEATURES FOR PDAC DETECTION")
    print("="*70)
    
    print(f"\nTotal selected features: {len(selected_features)}")
    print("\nFeature list:")
    for i, feat in enumerate(selected_features, 1):
        print(f"{i:2d}. {feat}")


if __name__ == "__main__":
    main()
