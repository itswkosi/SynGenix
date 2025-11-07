import requests
import json
import os
import pandas as pd
from tqdm import tqdm
import time
import config

class GDCDataCollector:
    """
    Collects data from GDC Data Portal for TCGA-PAAD project
    """
    
    def __init__(self):
        self.api_base = config.GDC_API_BASE
        self.project_id = config.PROJECT_ID
        self.session = requests.Session()
    
    def query_cases(self, size=1000):
        """
        Query all cases in TCGA-PAAD project
        """
        endpoint = f"{self.api_base}/cases"
        
        filters = {
            "op": "in",
            "content": {
                "field": "project.project_id",
                "value": [self.project_id]
            }
        }
        
        params = {
            "filters": json.dumps(filters),
            "format": "json",
            "size": size,
            "fields": "case_id,submitter_id,demographic,diagnoses,samples"
        }
        
        print(f"Querying cases from {self.project_id}...")
        response = self.session.get(endpoint, params=params)
        
        if response.status_code == 200:
            data = response.json()
            cases = data['data']['hits']
            print(f"Found {len(cases)} cases")
            return cases
        else:
            print(f"Error: {response.status_code}")
            return []
    
    def extract_clinical_data(self, cases):
        """
        Extract clinical information from case data
        """
        clinical_records = []
        
        for case in tqdm(cases, desc="Extracting clinical data"):
            record = {
                'case_id': case.get('case_id'),
                'submitter_id': case.get('submitter_id')
            }
            
            # Demographic data
            if 'demographic' in case and case['demographic']:
                demo = case['demographic']
                record['age_at_diagnosis'] = demo.get('age_at_index')
                record['gender'] = demo.get('gender')
                record['race'] = demo.get('race')
                record['ethnicity'] = demo.get('ethnicity')
                record['vital_status'] = demo.get('vital_status')
            
            # Diagnosis data
            if 'diagnoses' in case and case['diagnoses']:
                diag = case['diagnoses'][0]  # Take first diagnosis
                record['tumor_stage'] = diag.get('tumor_stage')
                record['primary_diagnosis'] = diag.get('primary_diagnosis')
                record['tissue_or_organ_of_origin'] = diag.get('tissue_or_organ_of_origin')
                record['morphology'] = diag.get('morphology')
                record['days_to_death'] = diag.get('days_to_death')
                record['days_to_last_follow_up'] = diag.get('days_to_last_follow_up')
            
            # Sample information
            if 'samples' in case and case['samples']:
                sample = case['samples'][0]
                record['sample_type'] = sample.get('sample_type')
                record['is_tumor'] = 1 if 'Tumor' in str(sample.get('sample_type', '')) else 0
            
            clinical_records.append(record)
        
        df = pd.DataFrame(clinical_records)
        return df
    
    def query_files(self, data_type, size=5000):
        """
        Query files of specific data type
        """
        endpoint = f"{self.api_base}/files"
        
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [self.project_id]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_type",
                        "value": [data_type]
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "format": "json",
            "size": size,
            "fields": "file_id,file_name,data_type,cases.case_id,cases.submitter_id"
        }
        
        print(f"Querying {data_type} files...")
        response = self.session.get(endpoint, params=params)
        
        if response.status_code == 200:
            data = response.json()
            files = data['data']['hits']
            print(f"Found {len(files)} {data_type} files")
            return files
        else:
            print(f"Error: {response.status_code}")
            return []
    
    def download_file(self, file_id, output_path):
        """
        Download a single file from GDC
        """
        endpoint = f"{self.api_base}/data/{file_id}"
        
        response = self.session.get(endpoint, stream=True)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"Failed to download {file_id}: {response.status_code}")
            return False
    
    def create_manifest(self, files, output_file):
        """
        Create a manifest file for batch download
        """
        manifest_data = []
        
        for file_info in files:
            manifest_data.append({
                'file_id': file_info['file_id'],
                'file_name': file_info['file_name'],
                'data_type': file_info.get('data_type', '')
            })
        
        df = pd.DataFrame(manifest_data)
        df.to_csv(output_file, index=False)
        print(f"Manifest saved to {output_file}")
        return df
    
    def collect_all_data(self):
        """
        Main method to collect all required data
        """
        print("="*70)
        print("TCGA-PAAD Data Collection")
        print("="*70)
        
        # Step 1: Query cases
        cases = self.query_cases()
        
        if not cases:
            print("No cases found. Exiting.")
            return
        
        # Step 2: Extract and save clinical data
        print("\nExtracting clinical data...")
        clinical_df = self.extract_clinical_data(cases)
        clinical_df.to_csv(config.CLINICAL_DATA_FILE, index=False)
        print(f"Clinical data saved: {clinical_df.shape}")
        print(f"Tumor samples: {clinical_df['is_tumor'].sum()}")
        print(f"Normal samples: {(clinical_df['is_tumor'] == 0).sum()}")
        
        # Step 3: Query gene expression files
        print("\nQuerying gene expression data...")
        gene_expr_files = self.query_files(config.DATA_TYPES['gene_expression'])
        
        if gene_expr_files:
            manifest_path = os.path.join(config.RAW_DATA_DIR, 'gene_expression_manifest.csv')
            self.create_manifest(gene_expr_files, manifest_path)
        
        # Step 4: Query mutation files
        print("\nQuerying mutation data...")
        mutation_files = self.query_files(config.DATA_TYPES['mutations'])
        
        if mutation_files:
            manifest_path = os.path.join(config.RAW_DATA_DIR, 'mutation_manifest.csv')
            self.create_manifest(mutation_files, manifest_path)
        
        print("\n" + "="*70)
        print("Data collection complete!")
        print(f"Clinical data: {config.CLINICAL_DATA_FILE}")
        print(f"Manifests saved in: {config.RAW_DATA_DIR}")
        print("\nNote: For gene expression and mutation files, use GDC Data Transfer Tool")
        print("with the generated manifest files for efficient batch download.")
        print("="*70)
        
        return clinical_df


def main():
    """
    Main execution function
    """
    collector = GDCDataCollector()
    clinical_data = collector.collect_all_data()
    
    if clinical_data is not None:
        print("\nClinical Data Summary:")
        print(clinical_data.info())
        print("\nFirst few records:")
        print(clinical_data.head())


if __name__ == "__main__":
    main()
