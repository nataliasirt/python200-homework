"""
World Happiness Pipeline Project - Assignment 01
Prefect-based end-to-end analysis of World Happiness data (2015-2024)

This pipeline performs:
1. Multi-year data loading and merging
2. Descriptive statistics analysis
3. Visual exploration with 4 plots
4. Hypothesis testing (pandemic impact)
5. Correlation analysis with Bonferroni correction
6. Summary report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from prefect import task, flow
from prefect.logging import get_run_logger
import urllib.request
import os


# =============================================================================
# Data Loading Helper Functions
# =============================================================================

def download_data_files():
    """Download World Happiness data files from GitHub if not present locally."""
    base_url = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/main/assignments/resources/happiness_project"
    data_dir = Path("assignments/resources/happiness_project")
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_run_logger()
    
    for year in range(2015, 2025):
        filename = f"world_happiness_{year}.csv"
        filepath = data_dir / filename
        
        if not filepath.exists():
            url = f"{base_url}/{filename}"
            try:
                logger.info(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                logger.info(f"Successfully downloaded {filename}")
            except Exception as e:
                logger.warning(f"Could not download {filename}: {e}")
        else:
            logger.info(f"{filename} already exists locally")


# =============================================================================
# Task 1: Load and Merge Data
# =============================================================================

@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    """
    Load World Happiness data from CSV files for all years (2015-2024).
    
    Returns:
        pd.DataFrame: Combined dataset with all years merged
    """
    logger = get_run_logger()
    
    # Try to download files if needed
    download_data_files()
    
    data_dir = Path("assignments/resources/happiness_project")
    dfs = []
    
    for year in range(2015, 2025):
        filepath = data_dir / f"world_happiness_{year}.csv"
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue
        
        try:
            # Load CSV with semicolon separator and comma as decimal
            df = pd.read_csv(
                filepath,
                sep=';',
                decimal=','
            )
            
            # Add year column
            df['Year'] = year
            
            # Standardize column names (remove extra spaces, lowercase)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            dfs.append(df)
            logger.info(f"Loaded {len(df)} records from {year}")
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    
    if not dfs:
        raise ValueError("No data files were successfully loaded!")
    
    # Merge all years
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged data
    output_path = Path("assignments_01/outputs/merged_happiness.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    logger.info(f"Merged dataset: {len(merged_df)} total records, {len(merged_df['year'].unique())} years")
    logger.info(f"Saved merged data to {output_path}")
    
    return merged_df


# =============================================================================
# Task 2: Descriptive Statistics
# =============================================================================

@task
def compute_descriptive_stats(df):
    """
    Compute descriptive statistics for happiness score.
    
    Returns:
        dict: Dictionary containing statistics
    """
    logger = get_run_logger()
    
    # Overall statistics
    happiness_col = 'happiness_score' if 'happiness_score' in df.columns else 'ladder_score'
    
    mean_happiness = df[happiness_col].mean()
    median_happiness = df[happiness_col].median()
    std_happiness = df[happiness_col].std()
    
    logger.info(f"Overall Happiness Statistics:")
    logger.info(f"  Mean: {mean_happiness:.3f}")
    logger.info(f"  Median: {median_happiness:.3f}")
    logger.info(f"  Std Dev: {std_happiness:.3f}")
    
    # By year
    logger.info(f"\nMean Happiness by Year:")
    by_year = df.groupby('year')[happiness_col].mean()
    for year, score in by_year.items():
        logger.info(f"  {int(year)}: {score:.3f}")
    
    # By region
    region_col = 'regional_indicator' if 'regional_indicator' in df.columns else 'region'
    logger.info(f"\nMean Happiness by Region:")
    by_region = df.groupby(region_col)[happiness_col].mean().sort_values(ascending=False)
    for region, score in by_region.items():
        logger.info(f"  {region}: {score:.3f}")
    
    stats_dict = {
        'mean': mean_happiness,
        'median': median_happiness,
        'std': std_happiness,
        'by_year': by_year,
        'by_region': by_region
    }
    
    return stats_dict


# =============================================================================
# Task 3: Visual Exploration
# =============================================================================

@task
def create_visualizations(df):
    """
    Create and save visualization plots.
    """
    logger = get_run_logger()
    output_dir = Path("assignments_01/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    happiness_col = 'happiness_score' if 'happiness_score' in df.columns else 'ladder_score'
    region_col = 'regional_indicator' if 'regional_indicator' in df.columns else 'region'
    gdp_col = 'gdp_per_capita' if 'gdp_per_capita' in df.columns else 'gdp'
    
    # Plot 1: Histogram of all happiness scores
    plt.figure(figsize=(10, 6))
    plt.hist(df[happiness_col].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Happiness Scores (2015-2024)', fontsize=14, fontweight='bold')
    plt.xlabel('Happiness Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'happiness_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved happiness_histogram.png")
    
    # Plot 2: Boxplot by year
    plt.figure(figsize=(12, 6))
    df.boxplot(column=happiness_col, by='year', figsize=(12, 6))
    plt.title('Happiness Score Distribution by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Happiness Score')
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.savefig(output_dir / 'happiness_by_year.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved happiness_by_year.png")
    
    # Plot 3: Scatter plot GDP vs Happiness
    if gdp_col in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[gdp_col], df[happiness_col], alpha=0.6, s=50)
        plt.title('GDP per Capita vs Happiness Score', fontsize=14, fontweight='bold')
        plt.xlabel('GDP per Capita')
        plt.ylabel('Happiness Score')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'gdp_vs_happiness.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved gdp_vs_happiness.png")
    
    # Plot 4: Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix - Happiness Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved correlation_heatmap.png")


# =============================================================================
# Task 4: Hypothesis Testing
# =============================================================================

@task
def hypothesis_testing(df):
    """
    Test pandemic impact on happiness (2019 vs 2020).
    """
    logger = get_run_logger()
    
    happiness_col = 'happiness_score' if 'happiness_score' in df.columns else 'ladder_score'
    
    # Test 1: 2019 vs 2020 (pandemic)
    data_2019 = df[df['year'] == 2019][happiness_col].dropna()
    data_2020 = df[df['year'] == 2020][happiness_col].dropna()
    
    t_stat, p_value = stats.ttest_ind(data_2019, data_2020)
    
    logger.info(f"\n=== Test 1: Pandemic Impact (2019 vs 2020) ===")
    logger.info(f"2019 Mean Happiness: {data_2019.mean():.3f}")
    logger.info(f"2020 Mean Happiness: {data_2020.mean():.3f}")
    logger.info(f"T-statistic: {t_stat:.3f}")
    logger.info(f"P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        direction = "decreased" if data_2020.mean() < data_2019.mean() else "increased"
        logger.info(f"SIGNIFICANT: Global happiness {direction} from 2019 to 2020 (p < 0.05)")
    else:
        logger.info(f"NOT SIGNIFICANT: No significant change in happiness from 2019 to 2020")
    
    # Test 2: Custom test - comparing regions
    region_col = 'regional_indicator' if 'regional_indicator' in df.columns else 'region'
    regions = df[region_col].value_counts().head(2).index
    
    if len(regions) >= 2:
        region1_data = df[df[region_col] == regions[0]][happiness_col].dropna()
        region2_data = df[df[region_col] == regions[1]][happiness_col].dropna()
        
        t_stat2, p_value2 = stats.ttest_ind(region1_data, region2_data)
        
        logger.info(f"\n=== Test 2: Regional Comparison ===")
        logger.info(f"{regions[0]} Mean: {region1_data.mean():.3f}")
        logger.info(f"{regions[1]} Mean: {region2_data.mean():.3f}")
        logger.info(f"T-statistic: {t_stat2:.3f}")
        logger.info(f"P-value: {p_value2:.6f}")
        
        if p_value2 < 0.05:
            logger.info(f"SIGNIFICANT: Happiness differs significantly between regions")
        else:
            logger.info(f"NOT SIGNIFICANT: No significant difference between regions")


# =============================================================================
# Task 5: Correlation Analysis with Bonferroni Correction
# =============================================================================

@task
def correlation_analysis(df):
    """
    Compute correlations with happiness and apply Bonferroni correction.
    """
    logger = get_run_logger()
    
    happiness_col = 'happiness_score' if 'happiness_score' in df.columns else 'ladder_score'
    
    # Select numeric columns (excluding year and happiness)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if happiness_col in numeric_cols:
        numeric_cols.remove(happiness_col)
    if 'year' in numeric_cols:
        numeric_cols.remove('year')
    if 'ranking' in numeric_cols:
        numeric_cols.remove('ranking')
    
    num_tests = len(numeric_cols)
    adjusted_alpha = 0.05 / num_tests
    
    logger.info(f"\n=== Correlation Analysis with Bonferroni Correction ===")
    logger.info(f"Number of correlation tests: {num_tests}")
    logger.info(f"Adjusted alpha (Bonferroni): {adjusted_alpha:.6f}")
    logger.info(f"\nCorrelations with {happiness_col}:")
    
    correlations = []
    
    for col in numeric_cols:
        # Remove NaN values
        valid_data = df[[col, happiness_col]].dropna()
        
        if len(valid_data) > 2:
            corr_coef, p_val = stats.pearsonr(valid_data[col], valid_data[happiness_col])
            
            sig_original = "YES" if p_val < 0.05 else "NO"
            sig_corrected = "YES" if p_val < adjusted_alpha else "NO"
            
            logger.info(f"\n{col}:")
            logger.info(f"  Correlation: {corr_coef:.4f}")
            logger.info(f"  P-value: {p_val:.6f}")
            logger.info(f"  Significant (α=0.05): {sig_original}")
            logger.info(f"  Significant (Bonferroni): {sig_corrected}")
            
            correlations.append({
                'variable': col,
                'correlation': corr_coef,
                'p_value': p_val,
                'sig_original': sig_original,
                'sig_bonferroni': sig_corrected
            })
    
    return correlations


# =============================================================================
# Task 6: Summary Report
# =============================================================================

@task
def generate_summary_report(df, stats_dict, correlations_list):
    """
    Generate comprehensive summary report of findings.
    """
    logger = get_run_logger()
    
    happiness_col = 'happiness_score' if 'happiness_score' in df.columns else 'ladder_score'
    region_col = 'regional_indicator' if 'regional_indicator' in df.columns else 'region'
    
    logger.info(f"\n{'='*70}")
    logger.info(f"WORLD HAPPINESS ANALYSIS - SUMMARY REPORT")
    logger.info(f"{'='*70}\n")
    
    # Dataset overview
    num_countries = df['country'].nunique() if 'country' in df.columns else df.iloc[:, 1].nunique()
    num_years = int(df['year'].max() - df['year'].min() + 1)
    
    logger.info(f"DATASET OVERVIEW:")
    logger.info(f"  Total Countries: {num_countries}")
    logger.info(f"  Years Covered: {num_years} years (2015-2024)")
    logger.info(f"  Total Records: {len(df)}")
    
    # Top and bottom regions
    logger.info(f"\nTOP 3 REGIONS BY MEAN HAPPINESS:")
    for i, (region, score) in enumerate(stats_dict['by_region'].head(3).items(), 1):
        logger.info(f"  {i}. {region}: {score:.3f}")
    
    logger.info(f"\nBOTTOM 3 REGIONS BY MEAN HAPPINESS:")
    for i, (region, score) in enumerate(stats_dict['by_region'].tail(3).items(), 1):
        logger.info(f"  {i}. {region}: {score:.3f}")
    
    # Pandemic impact
    data_2019 = df[df['year'] == 2019][happiness_col].mean()
    data_2020 = df[df['year'] == 2020][happiness_col].mean()
    
    logger.info(f"\nPANDEMIC IMPACT:")
    logger.info(f"  2019 Mean Happiness: {data_2019:.3f}")
    logger.info(f"  2020 Mean Happiness: {data_2020:.3f}")
    logger.info(f"  Change: {data_2020 - data_2019:+.3f}")
    
    # Most correlated variable
    if correlations_list:
        # Find most correlated after Bonferroni correction
        significant_after = [c for c in correlations_list if c['sig_bonferroni'] == 'YES']
        
        if significant_after:
            most_corr = max(significant_after, key=lambda x: abs(x['correlation']))
            logger.info(f"\nMOST STRONGLY CORRELATED VARIABLE (Bonferroni-corrected):")
            logger.info(f"  Variable: {most_corr['variable']}")
            logger.info(f"  Correlation: {most_corr['correlation']:.4f}")
            logger.info(f"  P-value: {most_corr['p_value']:.6f}")
        else:
            logger.info(f"\nNOTE: No variables remain significant after Bonferroni correction")
    
    logger.info(f"\n{'='*70}\n")


# =============================================================================
# Main Prefect Flow
# =============================================================================

@flow
def happiness_pipeline():
    """
    Main Prefect flow orchestrating all analysis tasks.
    """
    logger = get_run_logger()
    logger.info("Starting World Happiness Analysis Pipeline")
    
    # Task 1
    logger.info("\n--- TASK 1: Loading and Merging Data ---")
    df = load_and_merge_data()
    
    # Task 2
    logger.info("\n--- TASK 2: Descriptive Statistics ---")
    stats_dict = compute_descriptive_stats(df)
    
    # Task 3
    logger.info("\n--- TASK 3: Visual Exploration ---")
    create_visualizations(df)
    
    # Task 4
    logger.info("\n--- TASK 4: Hypothesis Testing ---")
    hypothesis_testing(df)
    
    # Task 5
    logger.info("\n--- TASK 5: Correlation Analysis ---")
    correlations = correlation_analysis(df)
    
    # Task 6
    logger.info("\n--- TASK 6: Summary Report ---")
    generate_summary_report(df, stats_dict, correlations)
    
    logger.info("\nPipeline execution completed successfully!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    happiness_pipeline()
