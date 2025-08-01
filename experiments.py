#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
#   "scikit-learn",
#   "matplotlib",
#   "scipy",
#   "seaborn",
# ]
# ///
"""
MDL vs Cross-Validation: A Rigorous Comparison
==============================================

Minimal implementation comparing Minimum Description Length (MDL) 
and Cross-Validation (CV) for model selection across 4 experiments.

Usage: uv run experiments.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoCV, LassoLarsIC
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")


def compute_effect_size(data1, data2):
    """Cohen's d for paired samples: d = mean(differences) / std(differences)"""
    differences = np.array(data1) - np.array(data2)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    return mean_diff / std_diff if std_diff != 0 else 0.0


def statistical_test(data1, data2, method1_name, method2_name):
    """Perform statistical comparison with effect size."""
    t_stat, p_val = stats.ttest_rel(data1, data2)
    effect_size = compute_effect_size(data1, data2)
    
    # Interpret effect size
    if abs(effect_size) < 0.2:
        interpretation = "negligible"
    elif abs(effect_size) < 0.5:
        interpretation = "small"
    elif abs(effect_size) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        'methods': f"{method1_name} vs {method2_name}",
        'p_value': p_val,
        'effect_size': effect_size,
        'interpretation': interpretation,
        'significant': p_val < 0.05,
        'mean1': np.mean(data1),
        'mean2': np.mean(data2)
    }


def enhanced_bic(y_true, y_pred, n_params):
    """BIC computation as MDL proxy."""
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    if mse <= 0:
        mse = 1e-10
    return n * np.log(mse) + n_params * np.log(n)


def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin, mean + margin


def print_experiment_results(name1, name2, data1, data2, comparison):
    """Print standardized experiment results with confidence intervals."""
    mean1, mean2 = np.mean(data1), np.mean(data2)
    ci1_low, ci1_high = compute_confidence_interval(data1)
    ci2_low, ci2_high = compute_confidence_interval(data2)
    
    print(f"{name1}: {mean1:.4f} (95% CI: [{ci1_low:.4f}, {ci1_high:.4f}])")
    print(f"{name2}: {mean2:.4f} (95% CI: [{ci2_low:.4f}, {ci2_high:.4f}])")
    print(f"Effect size (Cohen's d): {comparison['effect_size']:.3f} ({comparison['interpretation']})")
    print(f"Statistical significance: {'Yes' if comparison['significant'] else 'No'} (p={comparison['p_value']:.4f})")


def polynomial_experiment(n_runs=30):
    """Polynomial degree selection (synthetic cubic data)."""
    print("\n🔬 Polynomial Degree Selection")
    print("-" * 50)
    
    results = {'CV': [], 'BIC': []}
    
    for run in range(n_runs):
        # Generate cubic polynomial data with noise
        np.random.seed(run)
        X = np.random.uniform(-2, 2, size=(200, 1))
        y = 1 - 2*X.flatten() + 0.5*X.flatten()**2 + 0.3*X.flatten()**3 + np.random.normal(0, 1, 200)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=run)
        
        # Test polynomial degrees 0-7
        cv_scores = []
        bic_scores = []
        
        for degree in range(8):
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            
            # CV score
            cv_score = -np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
            cv_scores.append(cv_score)
            
            # BIC score
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            bic_score = enhanced_bic(y_train, y_pred_train, degree + 1)
            bic_scores.append(bic_score)
        
        # Select best degrees
        best_cv_degree = np.argmin(cv_scores)
        best_bic_degree = np.argmin(bic_scores)
        
        # Evaluate on test set
        for method, degree in [('CV', best_cv_degree), ('BIC', best_bic_degree)]:
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            model.fit(X_train, y_train)
            test_mse = mean_squared_error(y_test, model.predict(X_test))
            results[method].append(test_mse)
    
    # Statistical analysis
    comparison = statistical_test(results['CV'], results['BIC'], 'CV', 'BIC')
    print_experiment_results('CV MSE', 'BIC MSE', results['CV'], results['BIC'], comparison)
    
    return results, comparison


def feature_selection_experiment(n_runs=30):
    """Feature selection with Lasso (diabetes dataset)."""
    print("\n🔬 Feature Selection with Lasso")
    print("-" * 50)
    
    # Load real-world dataset
    data = load_diabetes()
    X_full, y_full = data.data, data.target
    
    results = {'CV': [], 'BIC': []}
    
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=run
        )
        
        # Lasso with CV
        lasso_cv = LassoCV(cv=5, random_state=run, max_iter=2000)
        lasso_cv.fit(X_train, y_train)
        cv_mse = mean_squared_error(y_test, lasso_cv.predict(X_test))
        
        # Lasso with BIC
        lasso_bic = LassoLarsIC(criterion='bic', max_iter=2000)
        lasso_bic.fit(X_train, y_train)
        bic_mse = mean_squared_error(y_test, lasso_bic.predict(X_test))
        
        results['CV'].append(cv_mse)
        results['BIC'].append(bic_mse)
    
    # Statistical analysis
    comparison = statistical_test(results['CV'], results['BIC'], 'CV', 'BIC')
    print_experiment_results('CV MSE', 'BIC MSE', results['CV'], results['BIC'], comparison)
    
    return results, comparison


def tree_experiment(n_runs=30):
    """Decision tree pruning (California housing)."""
    print("\n🔬 Decision Tree Pruning")
    print("-" * 50)
    
    # Load real-world dataset
    data = fetch_california_housing()
    X_full, y_full = data.data, data.target
    
    results = {'CV': [], 'MDL': []}
    
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=run
        )
        
        # Test tree depths 1-15
        cv_scores = []
        mdl_scores = []
        
        for depth in range(1, 16):
            # CV evaluation
            tree = DecisionTreeRegressor(max_depth=depth, random_state=run, min_samples_leaf=5)
            cv_score = -np.mean(cross_val_score(tree, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
            cv_scores.append(cv_score)
            
            # MDL evaluation
            tree.fit(X_train, y_train)
            y_pred_train = tree.predict(X_train)
            # Use internal nodes as complexity measure
            n_internal_nodes = tree.tree_.node_count - tree.get_n_leaves()
            mdl_score = enhanced_bic(y_train, y_pred_train, n_internal_nodes)
            mdl_scores.append(mdl_score)
        
        # Select best depths
        best_cv_depth = np.argmin(cv_scores) + 1
        best_mdl_depth = np.argmin(mdl_scores) + 1
        
        # Evaluate on test set
        for method, depth in [('CV', best_cv_depth), ('MDL', best_mdl_depth)]:
            tree = DecisionTreeRegressor(max_depth=depth, random_state=run, min_samples_leaf=5)
            tree.fit(X_train, y_train)
            test_mse = mean_squared_error(y_test, tree.predict(X_test))
            results[method].append(test_mse)
    
    # Statistical analysis
    comparison = statistical_test(results['CV'], results['MDL'], 'CV', 'MDL')
    print_experiment_results('CV MSE', 'MDL MSE', results['CV'], results['MDL'], comparison)
    
    return results, comparison


def period_detection_experiment(n_runs=30):
    """Period detection in noisy binary sequences."""
    print("\n🔬 Period Detection in Binary Sequences (with noise)")
    print("-" * 50)
    
    import math
    
    def generate_repeating_pattern(period_length=3, total_length=36, noise_prob=0.1, seed=42):
        """Generate a binary string with a repeating pattern plus noise."""
        np.random.seed(seed)
        pattern = np.random.randint(0, 2, period_length)
        full_sequence = np.tile(pattern, total_length // period_length + 1)[:total_length]
        
        # Add noise
        noise_mask = np.random.random(total_length) < noise_prob
        noisy_sequence = full_sequence.copy()
        noisy_sequence[noise_mask] = 1 - noisy_sequence[noise_mask]  # Flip bits
        
        return noisy_sequence, pattern
    
    def mdl_score_period(sequence, period_length):
        """Calculate MDL score for a period hypothesis with noise handling."""
        n = len(sequence)
        
        # Get the best-fitting pattern for this period
        pattern = np.zeros(period_length)
        for i in range(period_length):
            # Majority vote for each position in pattern
            positions = np.arange(i, n, period_length)
            if len(positions) > 0:
                pattern[i] = np.round(np.mean(sequence[positions]))
        
        # Reconstruct sequence
        reconstructed = np.tile(pattern, n // period_length + 1)[:n]
        
        # Count errors
        errors = np.sum(sequence != reconstructed)
        
        # MDL score components
        pattern_bits = period_length
        complexity_bits = math.log2(period_length) if period_length > 0 else 0
        
        # Encoding the errors using binary entropy
        if errors == 0:
            noise_bits = 0
        elif errors < n:
            p_err = errors / n
            if p_err > 0 and p_err < 1:
                noise_bits = n * (-p_err * math.log2(p_err) - (1-p_err) * math.log2(1-p_err))
            else:
                noise_bits = 0
        else:
            noise_bits = float('inf')
        
        return pattern_bits + complexity_bits + noise_bits
    
    def cv_accuracy_period(sequence, period_length):
        """Leave-one-out CV accuracy for period hypothesis."""
        n = len(sequence)
        correct = 0
        
        for i in range(n):
            # Leave out position i
            train_seq = np.concatenate([sequence[:i], sequence[i+1:]])
            true_val = sequence[i]
            
            # Predict based on period hypothesis
            pattern_pos = i % period_length
            
            # Find matching positions in training
            matching_positions = []
            for j in range(len(train_seq)):
                adj_pos = j if j < i else j + 1
                if adj_pos % period_length == pattern_pos:
                    matching_positions.append(j)
            
            if matching_positions:
                # Majority vote
                votes = [train_seq[j] for j in matching_positions]
                pred_val = 1 if sum(votes) > len(votes) / 2 else 0
                if pred_val == true_val:
                    correct += 1
            else:
                correct += 0.5
        
        return correct / n
    
    # Run experiments
    results = {'MDL': [], 'CV': []}
    true_period = 3
    sequence_length = 36  # Match the standalone version that works
    
    for run in range(n_runs):
        # Generate noisy sequence
        sequence, _ = generate_repeating_pattern(true_period, sequence_length, noise_prob=0.1, seed=run)
        
        # Test all periods 1-15 (not just divisors)
        periods_to_test = list(range(1, 16))
        
        # MDL selection
        mdl_scores = {p: mdl_score_period(sequence, p) for p in periods_to_test}
        mdl_best = min(mdl_scores.keys(), key=lambda p: mdl_scores[p])
        
        # CV selection (find all with perfect accuracy)
        cv_scores = {p: cv_accuracy_period(sequence, p) for p in periods_to_test}
        max_acc = max(cv_scores.values())
        cv_best_candidates = [p for p, acc in cv_scores.items() if acc == max_acc]
        
        # Apply Occam's razor: select smallest period among ties
        cv_best = min(cv_best_candidates)
        
        # MDL identifies correct period
        results['MDL'].append(1.0 if mdl_best == true_period else 0.0)
        
        # CV identifies correct period (using Occam's razor for ties)
        results['CV'].append(1.0 if cv_best == true_period else 0.0)
    
    # Statistical analysis
    comparison = statistical_test(results['MDL'], results['CV'], 'MDL', 'CV')
    print_experiment_results('MDL Success Rate', 'CV Success Rate', results['MDL'], results['CV'], comparison)
    
    return results, comparison


def create_summary_plot(period_results, poly_results, lasso_results, tree_results):
    """Create summary visualization."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    experiments = [
        ('Period\nDetection', period_results, 'Success Rate'),
        ('Polynomial\nRegression', poly_results, 'Test MSE'),
        ('Feature\nSelection', lasso_results, 'Test MSE'),
        ('Decision\nTrees', tree_results, 'Test MSE')
    ]
    
    for idx, (title, (results, comparison), ylabel) in enumerate(experiments):
        ax = axes[idx]
        
        # Create box plots
        data = [results[method] for method in results.keys()]
        bp = ax.boxplot(data, labels=list(results.keys()), patch_artist=True)
        
        # Color by effect size
        effect_size = abs(comparison['effect_size'])
        if effect_size < 0.2:
            color = 'lightgreen'
        elif effect_size < 0.5:
            color = 'yellow'
        else:
            color = 'lightcoral'
        
        for patch in bp['boxes']:
            patch.set_facecolor(color)
        
        ax.set_title(f"{title}\nd={comparison['effect_size']:.3f}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mdl_vs_cv_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Summary plot saved as 'mdl_vs_cv_results.png'")


def main():
    """Run all experiments and generate results."""
    print("🧪 MDL vs Cross-Validation: Comparison Study")
    print("=" * 60)
    print("Running 4 experiments with 30 runs each...\n")
    
    # Run experiments (Period Detection first - most dramatic difference)
    period_results = period_detection_experiment()
    poly_results = polynomial_experiment()
    lasso_results = feature_selection_experiment()
    tree_results = tree_experiment()
    
    # Create visualization
    create_summary_plot(period_results, poly_results, lasso_results, tree_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 KEY FINDINGS")
    print("=" * 60)
    
    comparisons = [
        ("Period Detection", period_results[1]),
        ("Polynomial Regression", poly_results[1]),
        ("Feature Selection", lasso_results[1]),
        ("Decision Trees", tree_results[1])
    ]
    
    for exp_name, comparison in comparisons:
        effect = comparison['interpretation']
        print(f"• {exp_name}: {effect} effect size ({comparison['effect_size']:.3f})")
        if abs(comparison['effect_size']) < 0.2:
            print(f"  → Methods essentially equivalent")
        else:
            winner = 'MDL' if comparison['effect_size'] > 0 else 'CV'
            print(f"  → {winner} performs better")
    
    print(f"\n✅ Results generated successfully!")


if __name__ == "__main__":
    main()