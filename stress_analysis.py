"""
Undefined X (Metastability) Analysis Script
============================================
This script analyzes how increasing input frequency causes undefined X states
in digital circuit outputs. When inputs change too quickly, outputs become
undefined (0.5 value = X state) instead of settling to stable logic levels (0 or 1).

Pins Configuration:
- Input Pins: 2 (B0), 3 (S0), 6 (S1), 7 (C1)
- Output Pins: 1 (A0), 4 (A1), 5 (B1), 8 (C in)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class UndefinedXAnalyzer:
    def __init__(self, csv_files):
        """Initialize with list of CSV file paths"""
        self.csv_files = csv_files
        self.data = {}
        self.frequencies = []
        self.output_pins = [1, 4, 5, 8]  # Pin indices for outputs (A0, A1, B1, C_in)
        self.output_names = ['Pin 1 (A0)', 'Pin 4 (A1)', 'Pin 5 (B1)', 'Pin 8 (C in)']
        
    def load_data(self):
        """Load all CSV files and extract frequency information"""
        for csv_file in self.csv_files:
            # Read metadata to get baud rate
            with open(csv_file, 'r') as f:
                for line in f:
                    if 'Pattern Generator 1' in line and 'baud rate' in line:
                        # Extract baud rate from line like "% Pattern Generator 1: 10 values, divider 200,000, baud rate 625"
                        parts = line.split('baud rate')
                        baud_rate = int(parts[1].strip().replace(',', ''))
                        self.frequencies.append(baud_rate)
                        break
            
            # Load the actual data
            df = pd.read_csv(csv_file, comment='%')
            self.data[baud_rate] = df
            
        # Sort by frequency
        self.frequencies.sort()
        
    def calculate_metastability_metrics(self):
        """Calculate undefined X percentage for each frequency"""
        results = []
        
        for freq in self.frequencies:
            df = self.data[freq]
            
            # Count undefined X values (0.5) in output pins
            undefined_counts = {}
            stable_counts = {}
            total_samples = len(df)
            
            for pin_idx, pin_name in zip(self.output_pins, self.output_names):
                values = df.iloc[:, pin_idx]
                undefined = (values == 0.5).sum()
                stable = ((values == 0) | (values == 1)).sum()
                
                undefined_counts[pin_name] = undefined
                stable_counts[pin_name] = stable
                
                results.append({
                    'Frequency (Hz)': freq,
                    'Pin': pin_name,
                    'Undefined X Samples': undefined,
                    'Stable Samples': stable,
                    'Undefined X %': (undefined / total_samples) * 100
                })
        
        return pd.DataFrame(results)
    
    def plot_metastability_vs_frequency(self, metrics_df):
        """Create visualization of undefined X states"""
        fig = plt.figure(figsize=(16, 6))
        
        # 1. Trend Graph: Average increase in undefined X (0.5) values
        ax1 = plt.subplot(1, 2, 1)
        avg_undefined = metrics_df.groupby('Frequency (Hz)')['Undefined X %'].mean()
        
        # Create trend line with markers
        ax1.plot(self.frequencies, avg_undefined.values, 
                marker='o', linewidth=3, markersize=12, color='#e74c3c',
                label='Average Undefined X', markerfacecolor='#c0392b', 
                markeredgecolor='white', markeredgewidth=2)
        
        # Fill area under curve to emphasize growth
        ax1.fill_between(self.frequencies, 0, avg_undefined.values, 
                        alpha=0.3, color='#e74c3c')
        
        # Add value labels on points
        for freq, val in zip(self.frequencies, avg_undefined.values):
            ax1.annotate(f'{val:.1f}%', 
                        xy=(freq, val), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=11, 
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        ax1.set_xlabel('Input Frequency (Hz)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Undefined X (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Trend: Increase in Undefined X (0.5) Values', fontsize=16, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.set_ylim(-5, 105)
        
        # Add frequency labels
        ax1.set_xticks(self.frequencies)
        ax1.set_xticklabels([f'{freq:,}' for freq in self.frequencies], rotation=45, ha='right')
        
        # Add warning zones
        ax1.axhspan(0, 10, alpha=0.1, color='green', label='Safe Zone')
        ax1.axhspan(10, 50, alpha=0.1, color='yellow', label='Warning Zone')
        ax1.axhspan(50, 100, alpha=0.1, color='red', label='Critical Zone')
        ax1.legend(loc='upper left', fontsize=10)
        
        # 2. Bar plot: Average undefined X across all output pins
        ax2 = plt.subplot(1, 2, 2)
        colors = plt.cm.RdYlGn_r(avg_undefined / 100)
        bars = ax2.bar(range(len(avg_undefined)), avg_undefined, 
                      color=colors, edgecolor='black', linewidth=2)
        ax2.set_xticks(range(len(avg_undefined)))
        ax2.set_xticklabels([f'{freq:,}\nHz' for freq in avg_undefined.index], 
                           fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average Undefined X (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Undefined X Comparison by Frequency', fontsize=16, fontweight='bold')
        ax2.set_ylim(0, 110)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        # Add status labels
        for i, (bar, val) in enumerate(zip(bars, avg_undefined.values)):
            if val < 10:
                status = "✓ STABLE"
                color = 'green'
            elif val < 50:
                status = "⚠ UNSTABLE"
                color = 'orange'
            else:
                status = "✗ CRITICAL"
                color = 'red'
            
            ax2.text(bar.get_x() + bar.get_width()/2., 5,
                    status, ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color=color,
                    rotation=90)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_table(self, metrics_df):
        """Generate a summary table"""
        print("\n" + "="*100)
        print("UNDEFINED X (METASTABILITY) ANALYSIS SUMMARY")
        print("="*100)
        
        summary = metrics_df.groupby('Frequency (Hz)').agg({
            'Undefined X %': ['mean', 'min', 'max']
        }).round(2)
        
        print("\nAverage Undefined X by Frequency:")
        print("-" * 100)
        print(f"{'Frequency (Hz)':<20} {'Avg Undefined X %':<25} {'Min %':<15} {'Max %':<15}")
        print("-" * 100)
        
        for freq in self.frequencies:
            freq_data = metrics_df[metrics_df['Frequency (Hz)'] == freq]
            avg = freq_data['Undefined X %'].mean()
            min_val = freq_data['Undefined X %'].min()
            max_val = freq_data['Undefined X %'].max()
            status = "✓ STABLE" if avg < 10 else "⚠ UNSTABLE" if avg < 50 else "✗ CRITICAL"
            print(f"{freq:>15,} Hz     {avg:>8.2f}% {status:<12}   {min_val:>8.2f}%      {max_val:>8.2f}%")
        
        print("-" * 100)
        
        # Detailed breakdown by pin
        print("\n\nDetailed Undefined X by Pin and Frequency:")
        print("-" * 100)
        pivot = metrics_df.pivot(index='Pin', columns='Frequency (Hz)', values='Undefined X %')
        print(pivot.to_string())
        print("-" * 100)
        
        # Key observations
        print("\n\nKEY OBSERVATIONS:")
        print("-" * 100)
        print("1. At LOW frequencies (625 Hz):")
        low_freq_avg = metrics_df[metrics_df['Frequency (Hz)'] == self.frequencies[0]]['Undefined X %'].mean()
        print(f"   - Average undefined X: {low_freq_avg:.2f}%")
        print("   - Outputs settle to stable logic levels (0 or 1)")
        print("   - Circuit has sufficient time to respond to input changes")
        
        print("\n2. At MEDIUM frequencies (6,250 - 62,500 Hz):")
        if len(self.frequencies) > 2:
            mid_freqs = self.frequencies[1:-1]
            mid_avg = metrics_df[metrics_df['Frequency (Hz)'].isin(mid_freqs)]['Undefined X %'].mean()
            print(f"   - Average undefined X: {mid_avg:.2f}%")
            print("   - Partial undefined X states observed")
            print("   - Some outputs fail to settle before next input change")
        
        print("\n3. At HIGH frequencies (625,000 Hz):")
        high_freq_avg = metrics_df[metrics_df['Frequency (Hz)'] == self.frequencies[-1]]['Undefined X %'].mean()
        print(f"   - Average undefined X: {high_freq_avg:.2f}%")
        print("   - Severe undefined X states - outputs stuck at intermediate level (0.5)")
        print("   - Input changes too fast for circuit to stabilize")
        print("   - Critical failure condition for digital logic")
        
        print("\n4. CONCLUSION:")
        print("   - Undefined X states increase exponentially with input frequency")
        print("   - Circuit has a maximum operating frequency beyond which it fails")
        print("   - Proper timing analysis and frequency limits are critical for reliable operation")
        print("="*100 + "\n")


def main():
    """Main execution function"""
    # Define CSV files in order
    csv_files = [
        Path('1.csv'),  # 625 Hz
        Path('2.csv'),  # 6,250 Hz
        Path('3.csv'),  # 62,500 Hz
        Path('4.csv'),  # 625,000 Hz
    ]
    
    # Check if files exist
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"Error: {csv_file} not found!")
            return
    
    print("Loading data from CSV files...")
    analyzer = UndefinedXAnalyzer(csv_files)
    analyzer.load_data()
    
    print("Calculating undefined X metrics...")
    metrics_df = analyzer.calculate_metastability_metrics()
    
    print("Generating visualizations...")
    fig = analyzer.plot_metastability_vs_frequency(metrics_df)
    
    print("Creating summary report...")
    analyzer.generate_summary_table(metrics_df)
    
    # Save the plot
    output_file = 'undefined_x_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_file}")
    
    # Save metrics to CSV
    metrics_file = 'undefined_x_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved as: {metrics_file}")
    
    plt.show()


if __name__ == "__main__":
    main()
