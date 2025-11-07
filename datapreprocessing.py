import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import config
import os

class DataPreprocessor:
    """
    Preprocesses TCGA-PAAD genomic and clinical data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
    
    def load_clinical_data(self, filepath=None):
        """
        Load and preprocess clinical data
        """
        if filepath is None:
            filepath = config.CLINICAL_DATA_FILE
        
        print(f"Loading clinical data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        
        return df
    
    def clean_clinical_data(self, df):
        """
        Clean clinical data
        """
        print("\nCleaning clinical data...")
        
        # Convert age to numeric
        if 'age_at_diagnosis' in df.columns:
            df['age_at_diagnosis'] = pd.to_numeric(df['age_at_diagnosis'], errors='coerce')
        
        # Create binary vital status
        if 'vital_status' in df.columns:
            df['is_deceased'] = df['vital_status'].apply(
                lambda x: 1 if str(x).lower() == 'dead' else 0
            )
        
        # Extract stage information
        if 'tumor_stage' in df.columns:
            df['stage_numeric'] = df['tumor_stage'].apply(self._extract_stage_number)
        
        # Fill missing is_tumor values
        if 'is_tumor' not in df.columns:
            df['is_tumor'] = 1  # Assume all samples are tumor if not specified
        
        print(f"After cleaning: {df.shape}")
        return df
    
    def _extract_stage_number(self, stage):
        """
        Convert tumor stage to numeric
        """
        if pd.isna(stage):
            return np.nan
        
        stage = str(stage).lower()
        if 'i' in stage:
            if 'iv' in stage:
                return 4
            elif 'iii' in stage:
                return 3
            elif 'ii' in stage:
                return 2
            else:
                return 1
        return np.nan
    
    def load_gene_expression_data(self, filepath=None):
        """
        Load gene expression data
        For real TCGA data, this would parse multiple HTSeq files
        For now, we'll create a synthetic version
        """
        print("\nLoading gene expression data...")
        
        # In real scenario, you would:
        # 1. Read all HTSeq count files from downloaded data
        # 2. Merge them into a single matrix
        # 3. Normalize (TPM or FPKM)
        
        # For demonstration, create synthetic data
        clinical_df = self.load_clinical_data()
        n_samples = len(clinical_df)
        
        # Focus on key genes
        all_genes = (config.KEY_MUTATION_GENES + 
                    config.KEY_EXPRESSION_GENES + 
                    config.DNA_REPAIR_GENES)
        
        # Create synthetic expression matrix
        expr_data = {}
        expr_data['case_id'] = clinical_df['case_id'].values
        
        for gene in all_genes:
            # Simulate log2(TPM+1) values
            expr_data[f'{gene}_expr'] = np.random.lognormal(3, 1.5, n_samples)
        
        expr_df = pd.DataFrame(expr_data)
        print(f"Expression data shape: {expr_df.shape}")
        
        return expr_df
    
    def load_mutation_data(self, filepath=None):
        """
        Load and process mutation data
        For real TCGA data, this would parse MAF files
        """
        print("\nLoading mutation data...")
        
        # In real scenario, you would:
        # 1. Read MAF (Mutation Annotation Format) files
        # 2. Extract mutations in key genes
        # 3. Create binary mutation matrix
        
        # For demonstration, create synthetic data
        clinical_df = self.load_clinical_data()
        n_samples = len(clinical_df)
        
        mut_data = {}
        mut_data['case_id'] = clinical_df['case_id'].values
        
        # KRAS mutation frequency ~90% in PDAC
        mut_data['KRAS_mut'] = np.random.binomial(1, 0.90, n_samples)
        mut_data['TP53_mut'] = np.random.binomial(1, 0.70, n_samples)
        mut_data['CDKN2A_mut'] = np.random.binomial(1, 0.60, n_samples)
        mut_data['SMAD4_mut'] = np.random.binomial(1, 0.55, n_samples)
        
        for gene in config.KEY_MUTATION_GENES[4:]:  # Remaining genes
            mut_data[f'{gene}_mut'] = np.random.binomial(1, 0.15, n_samples)
        
        for gene in config.DNA_REPAIR_GENES:
            mut_data[f'{gene}_mut'] = np.random.binomial(1, 0.05, n_samples)
        
        mut_df = pd.DataFrame(mut_data)
        print(f"Mutation data shape: {mut_df.shape}")
        
        return mut_df
    
    def integrate_data(self, clinical_df, expr_df, mut_df):
        """
        Integrate all data sources
        """
        print("\nIntegrating data sources...")
        
        # Merge on case_id
        integrated = clinical_df.copy()
        integrated = integrated.merge(expr_df, on='case_id', how='left')
        integrated = integrated.merge(mut_df, on='case_id', how='left')
        
        print(f"Integrated data shape: {integrated.shape}")
        
        return integrated
    
    def handle_missing_values(self, df):
        """
        Handle missing values
        """
        print("\nHandling missing values...")
        
        # Calculate missing percentage
        missing_pct = (df.isnull().sum() / len(df)) * 100
        
        # Remove columns with too many missing values
        cols_to_drop = missing_pct[missing_pct > config.MISSING_VALUE_THRESHOLD * 100].index
        if len(cols_to_drop) > 0:
            print(f"Dropping {len(cols_to_drop)} columns with >{config.MISSING_VALUE_THRESHOLD*100}% missing")
            df = df.drop(columns=cols_to_drop)
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove ID columns from processing
        id_cols = ['case_id', 'submitter_id']
        numeric_cols = [col for col in numeric_cols if col not in id_cols]
        categorical_cols = [col for col in categorical_cols if col not in id_cols]
        
        # Impute numeric columns
        if numeric_cols:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Fill categorical with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        print(f"After handling missing values: {df.shape}")
        return df
    
    def encode_categorical(self, df):
        """
        Encode categorical variables
        """
        print("\nEncoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        id_cols = ['case_id', 'submitter_id']
        categorical_cols = [col for col in categorical_cols if col not in id_cols]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        return df
    
    def remove_low_variance(self, df, target_col='is_tumor'):
        """
        Remove very low variance features (only extremely low variance)
        Keep most features for variance-based selection later
        """
        print("\nRemoving extremely low variance features...")
        
        # Get numeric columns excluding target and IDs
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['case_id', target_col]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate variance
        variances = df[feature_cols].var()
        
        # Only remove features with essentially zero variance
        zero_var_threshold = 1e-10
        low_var_cols = variances[variances < zero_var_threshold].index.tolist()
        
        if low_var_cols:
            print(f"Removing {len(low_var_cols)} zero-variance features")
            df = df.drop(columns=low_var_cols)
        else:
            print("No zero-variance features found")
        
        print(f"Variance statistics for remaining features:")
        print(f"  Min: {variances[variances >= zero_var_threshold].min():.6f}")
        print(f"  Max: {variances.max():.6f}")
        print(f"  Mean: {variances.mean():.6f}")
        
        return df
    
    def detect_outliers(self, df, target_col='is_tumor'):
        """
        Detect and cap outliers using Z-score method
        """
        print("\nDetecting and capping outliers...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['case_id', target_col]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in feature_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > config.OUTLIER_ZSCORE_THRESHOLD
            
            if outliers.any():
                # Cap at threshold
                lower_bound = df[col].mean() - config.OUTLIER_ZSCORE_THRESHOLD * df[col].std()
                upper_bound = df[col].mean() + config.OUTLIER_ZSCORE_THRESHOLD * df[col].std()
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def normalize_features(self, df, target_col='is_tumor'):
        """
        Normalize numeric features using StandardScaler
        """
        print("\nNormalizing features...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['case_id', target_col]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        return df
    
    def preprocess_pipeline(self):
        """
        Complete preprocessing pipeline
        """
        print("="*70)
        print("DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Load clinical data
        clinical_df = self.load_clinical_data()
        clinical_df = self.clean_clinical_data(clinical_df)
        
        # Step 2: Load expression data
        expr_df = self.load_gene_expression_data()
        
        # Step 3: Load mutation data
        mut_df = self.load_mutation_data()
        
        # Step 4: Integrate all data
        integrated_df = self.integrate_data(clinical_df, expr_df, mut_df)
        
        # Step 5: Handle missing values
        integrated_df = self.handle_missing_values(integrated_df)
        
        # Step 6: Encode categorical variables
        integrated_df = self.encode_categorical(integrated_df)
        
        # Step 7: Remove low variance features
        integrated_df = self.remove_low_variance(integrated_df)
        
        # Step 8: Detect and cap outliers
        integrated_df = self.detect_outliers(integrated_df)
        
        # Step 9: Normalize features
        integrated_df = self.normalize_features(integrated_df)
        
        # Save processed data
        output_path = config.INTEGRATED_DATA_FILE
        integrated_df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")
        print(f"Final shape: {integrated_df.shape}")
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        
        return integrated_df


def main():
    """
    Main execution function
    """
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline()
    
    print("\nProcessed Data Summary:")
    print(processed_data.info())
    print("\nTarget distribution:")
    print(processed_data['is_tumor'].value_counts())
    print("\nFirst few records:")
    print(processed_data.head())


if __name__ == "__main__":
    main()
