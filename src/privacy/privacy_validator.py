"""
Comprehensive Privacy Validation

Combines all privacy mechanisms to provide overall privacy assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
import json

from .identifier_manager import IdentifierManager
from .k_anonymity import KAnonymityValidator, KAnonymityResult
from .differential_privacy import DifferentialPrivacy
from .generalization import AgeGeneralizer, GeographicGeneralizer, TemporalGeneralizer

logger = logging.getLogger(__name__)


@dataclass
class PrivacyMetrics:
    """Overall privacy metrics"""
    k_anonymity: int
    l_diversity: float
    t_closeness: float
    differential_privacy_epsilon: float
    quasi_identifier_combinations: int
    unique_combination_ratio: float
    suppression_rate: float
    generalization_levels: Dict[str, int]
    risk_score: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PrivacyValidationResult:
    """Comprehensive privacy validation results"""
    overall_metrics: PrivacyMetrics
    k_anonymity_result: Optional[KAnonymityResult]
    l_diversity_scores: Dict[str, float]
    attribute_risks: Dict[str, Dict[str, Any]]
    privacy_guarantees: Dict[str, bool]
    timestamp: datetime
    validation_passed: bool


class PrivacyValidator:
    """
    Comprehensive privacy validation combining multiple privacy mechanisms
    """
    
    def __init__(self, 
                 k_threshold: int = 5,
                 l_threshold: int = 3,
                 epsilon: float = 1.0,
                 max_risk_score: float = 0.2):
        """
        Initialize privacy validator
        
        Args:
            k_threshold: Minimum k-anonymity value
            l_threshold: Minimum l-diversity value
            epsilon: Differential privacy epsilon
            max_risk_score: Maximum acceptable risk score (0-1)
        """
        self.k_threshold = k_threshold
        self.l_threshold = l_threshold
        self.epsilon = epsilon
        self.max_risk_score = max_risk_score
        
        # Initialize sub-validators
        self.k_validator = KAnonymityValidator(k_threshold)
        self.dp = DifferentialPrivacy(epsilon)
        
        # Define quasi-identifiers for NEDIS
        self.quasi_identifiers = [
            'pat_age', 'pat_sex', 'pat_sarea',
            'vst_dt', 'vst_tm', 'ktas_lv',
            'emorg_type', 'emorg_btype'
        ]
        
        # Define sensitive attributes
        self.sensitive_attributes = [
            'ed_diag', 'outcome', 'death_flag',
            'admission_flag', 'ktas_lv'
        ]
    
    def validate(self, df: pd.DataFrame,
                quasi_identifiers: Optional[List[str]] = None,
                sensitive_attributes: Optional[List[str]] = None) -> PrivacyValidationResult:
        """
        Perform comprehensive privacy validation
        
        Args:
            df: DataFrame to validate
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attributes: List of sensitive attribute columns
            
        Returns:
            PrivacyValidationResult with comprehensive metrics
        """
        logger.info("Starting comprehensive privacy validation")
        
        qi = quasi_identifiers or self.quasi_identifiers
        sa = sensitive_attributes or self.sensitive_attributes
        
        # Filter to existing columns
        qi = [col for col in qi if col in df.columns]
        sa = [col for col in sa if col in df.columns]
        
        # 1. K-anonymity validation
        k_result = self._validate_k_anonymity(df, qi)
        
        # 2. L-diversity validation
        l_scores = self._validate_l_diversity(df, qi, sa)
        
        # 3. T-closeness validation
        t_score = self._validate_t_closeness(df, qi, sa)
        
        # 4. Attribute risk analysis
        attr_risks = self._analyze_attribute_risks(df, qi)
        
        # 5. Calculate overall risk score
        risk_score = self._calculate_risk_score(k_result, l_scores, t_score, attr_risks)
        risk_level = self._determine_risk_level(risk_score)
        
        # 6. Generate recommendations
        recommendations = self._generate_recommendations(
            k_result, l_scores, risk_score, attr_risks
        )
        
        # 7. Check privacy guarantees
        guarantees = self._check_privacy_guarantees(k_result, l_scores, risk_score)
        
        # Create overall metrics
        metrics = PrivacyMetrics(
            k_anonymity=k_result.k_value if k_result else 0,
            l_diversity=min(l_scores.values()) if l_scores else 0,
            t_closeness=t_score,
            differential_privacy_epsilon=self.epsilon,
            quasi_identifier_combinations=self._count_unique_combinations(df, qi),
            unique_combination_ratio=self._calculate_unique_ratio(df, qi),
            suppression_rate=0.0,  # Will be calculated if suppression is applied
            generalization_levels={},  # Will be filled if generalization is applied
            risk_score=risk_score,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
        # Create validation result
        result = PrivacyValidationResult(
            overall_metrics=metrics,
            k_anonymity_result=k_result,
            l_diversity_scores=l_scores,
            attribute_risks=attr_risks,
            privacy_guarantees=guarantees,
            timestamp=datetime.now(),
            validation_passed=all(guarantees.values())
        )
        
        logger.info(f"Privacy validation complete: risk_level={risk_level}, "
                   f"k={metrics.k_anonymity}, passed={result.validation_passed}")
        
        return result
    
    def _validate_k_anonymity(self, df: pd.DataFrame,
                             quasi_identifiers: List[str]) -> Optional[KAnonymityResult]:
        """Validate k-anonymity"""
        if not quasi_identifiers:
            return None
        
        return self.k_validator.validate(df, quasi_identifiers)
    
    def _validate_l_diversity(self, df: pd.DataFrame,
                             quasi_identifiers: List[str],
                             sensitive_attributes: List[str]) -> Dict[str, float]:
        """
        Validate l-diversity for each sensitive attribute
        
        L-diversity ensures that each equivalence class has at least l
        distinct values for each sensitive attribute
        """
        l_scores = {}
        
        if not quasi_identifiers or not sensitive_attributes:
            return l_scores
        
        # Create equivalence classes
        for sensitive_attr in sensitive_attributes:
            if sensitive_attr not in df.columns:
                continue
            
            # Group by quasi-identifiers
            groups = df.groupby(quasi_identifiers)[sensitive_attr].apply(
                lambda x: x.nunique()
            ).reset_index(name='diversity')
            
            # Calculate minimum diversity
            min_diversity = groups['diversity'].min()
            l_scores[sensitive_attr] = min_diversity
            
            logger.info(f"L-diversity for {sensitive_attr}: {min_diversity}")
        
        return l_scores
    
    def _validate_t_closeness(self, df: pd.DataFrame,
                             quasi_identifiers: List[str],
                             sensitive_attributes: List[str]) -> float:
        """
        Validate t-closeness
        
        T-closeness ensures that the distribution of sensitive attributes
        in each equivalence class is close to the overall distribution
        """
        if not quasi_identifiers or not sensitive_attributes:
            return 1.0
        
        max_distance = 0.0
        
        for sensitive_attr in sensitive_attributes:
            if sensitive_attr not in df.columns:
                continue
            
            # Overall distribution
            overall_dist = df[sensitive_attr].value_counts(normalize=True)
            
            # Calculate distance for each equivalence class
            groups = df.groupby(quasi_identifiers)
            
            for _, group in groups:
                if len(group) < 2:
                    continue
                
                # Group distribution
                group_dist = group[sensitive_attr].value_counts(normalize=True)
                
                # Calculate Earth Mover's Distance (simplified)
                distance = 0.0
                for value in overall_dist.index:
                    p = overall_dist.get(value, 0)
                    q = group_dist.get(value, 0)
                    distance += abs(p - q)
                
                distance = distance / 2  # Normalize
                max_distance = max(max_distance, distance)
        
        return max_distance
    
    def _analyze_attribute_risks(self, df: pd.DataFrame,
                                quasi_identifiers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Analyze risk for each attribute"""
        risks = {}
        
        for col in quasi_identifiers:
            if col not in df.columns:
                continue
            
            unique_values = df[col].nunique()
            total_values = len(df[col])
            
            # Handle empty dataframe
            if total_values == 0:
                risks[col] = {
                    'unique_values': 0,
                    'uniqueness_ratio': 0.0,
                    'entropy': 0.0,
                    'risk_level': 'UNKNOWN',
                    'most_common': 0,
                    'least_common': 0
                }
                continue
            
            # Calculate entropy
            value_counts = df[col].value_counts()
            probabilities = value_counts / total_values
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # Determine risk level for attribute
            if unique_values / total_values > 0.8:
                attr_risk = "HIGH"
            elif unique_values / total_values > 0.5:
                attr_risk = "MEDIUM"
            else:
                attr_risk = "LOW"
            
            risks[col] = {
                'unique_values': unique_values,
                'uniqueness_ratio': unique_values / total_values,
                'entropy': entropy,
                'risk_level': attr_risk,
                'most_common': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_common': value_counts.iloc[-1] if len(value_counts) > 0 else 0
            }
        
        return risks
    
    def _count_unique_combinations(self, df: pd.DataFrame,
                                  quasi_identifiers: List[str]) -> int:
        """Count unique quasi-identifier combinations"""
        if not quasi_identifiers:
            return 0
        
        valid_qi = [qi for qi in quasi_identifiers if qi in df.columns]
        if not valid_qi:
            return 0
        
        return df[valid_qi].drop_duplicates().shape[0]
    
    def _calculate_unique_ratio(self, df: pd.DataFrame,
                               quasi_identifiers: List[str]) -> float:
        """Calculate ratio of unique combinations to total records"""
        unique_combos = self._count_unique_combinations(df, quasi_identifiers)
        if len(df) == 0:
            return 0.0
        return unique_combos / len(df)
    
    def _calculate_risk_score(self, k_result: Optional[KAnonymityResult],
                             l_scores: Dict[str, float],
                             t_score: float,
                             attr_risks: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate overall risk score (0-1, lower is better)
        """
        risk_components = []
        
        # K-anonymity risk
        if k_result:
            k_risk = 1.0 / max(k_result.k_value, 1)
            risk_components.append(k_risk * 0.4)  # 40% weight
        
        # L-diversity risk
        if l_scores:
            l_risk = 1.0 / max(min(l_scores.values()), 1)
            risk_components.append(l_risk * 0.3)  # 30% weight
        
        # T-closeness risk
        risk_components.append(t_score * 0.2)  # 20% weight
        
        # Attribute risks
        if attr_risks:
            high_risk_attrs = sum(1 for attr in attr_risks.values() 
                                if attr['risk_level'] == 'HIGH')
            attr_risk = high_risk_attrs / max(len(attr_risks), 1)
            risk_components.append(attr_risk * 0.1)  # 10% weight
        
        return sum(risk_components)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score < 0.2:
            return "LOW"
        elif risk_score < 0.5:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _generate_recommendations(self, k_result: Optional[KAnonymityResult],
                                 l_scores: Dict[str, float],
                                 risk_score: float,
                                 attr_risks: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate privacy improvement recommendations"""
        recommendations = []
        
        # K-anonymity recommendations
        if k_result and k_result.k_value < self.k_threshold:
            recommendations.append(
                f"Increase k-anonymity from {k_result.k_value} to at least {self.k_threshold} "
                f"through generalization or suppression"
            )
            if k_result.num_violations > 0:
                recommendations.append(
                    f"Address {k_result.num_violations} k-anonymity violations "
                    f"affecting {len(k_result.violation_records)} records"
                )
        
        # L-diversity recommendations
        if l_scores:
            low_diversity = [attr for attr, score in l_scores.items() 
                           if score < self.l_threshold]
            if low_diversity:
                recommendations.append(
                    f"Improve l-diversity for attributes: {', '.join(low_diversity)}"
                )
        
        # High-risk attribute recommendations
        if attr_risks:
            high_risk = [attr for attr, risk in attr_risks.items() 
                        if risk['risk_level'] == 'HIGH']
            if high_risk:
                recommendations.append(
                    f"Apply generalization to high-risk attributes: {', '.join(high_risk)}"
                )
        
        # Overall risk recommendations
        if risk_score > self.max_risk_score:
            recommendations.append(
                f"Overall risk score ({risk_score:.2f}) exceeds threshold ({self.max_risk_score})"
            )
            recommendations.append(
                "Consider applying differential privacy with appropriate epsilon value"
            )
        
        return recommendations
    
    def _check_privacy_guarantees(self, k_result: Optional[KAnonymityResult],
                                 l_scores: Dict[str, float],
                                 risk_score: float) -> Dict[str, bool]:
        """Check if privacy guarantees are met"""
        guarantees = {}
        
        # K-anonymity guarantee
        guarantees['k_anonymity'] = (
            k_result.satisfied if k_result else False
        )
        
        # L-diversity guarantee
        guarantees['l_diversity'] = (
            all(score >= self.l_threshold for score in l_scores.values())
            if l_scores else False
        )
        
        # Risk score guarantee
        guarantees['risk_score'] = risk_score <= self.max_risk_score
        
        # Differential privacy guarantee
        guarantees['differential_privacy'] = self.epsilon > 0
        
        return guarantees
    
    def generate_report(self, result: PrivacyValidationResult,
                       output_path: Optional[str] = None) -> str:
        """
        Generate detailed privacy validation report
        
        Args:
            result: Privacy validation result
            output_path: Optional path to save report
            
        Returns:
            HTML report as string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Privacy Validation Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }}
                h1 {{
                    color: #2d3748;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 15px;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #4a5568;
                    margin-top: 30px;
                    border-left: 4px solid #764ba2;
                    padding-left: 15px;
                }}
                .risk-level {{
                    display: inline-block;
                    padding: 8px 20px;
                    border-radius: 20px;
                    font-weight: bold;
                    color: white;
                    font-size: 1.2em;
                }}
                .risk-low {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }}
                .risk-medium {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }}
                .risk-high {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
                .metric-card {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    border-radius: 15px;
                    padding: 20px;
                    margin: 15px 0;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .metric-title {{
                    font-weight: bold;
                    color: #2d3748;
                    margin-bottom: 10px;
                    font-size: 1.1em;
                }}
                .metric-value {{
                    font-size: 1.8em;
                    color: #667eea;
                    font-weight: bold;
                }}
                .guarantee {{
                    display: inline-block;
                    margin: 5px;
                    padding: 5px 15px;
                    border-radius: 15px;
                    background: white;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .guarantee-met {{
                    border: 2px solid #48bb78;
                    color: #22543d;
                }}
                .guarantee-not-met {{
                    border: 2px solid #f56565;
                    color: #742a2a;
                }}
                .recommendation {{
                    background: #fef5e7;
                    border-left: 4px solid #f39c12;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #e2e8f0;
                }}
                tr:hover {{
                    background: #f7fafc;
                }}
                .timestamp {{
                    color: #718096;
                    font-size: 0.9em;
                    margin-top: 30px;
                    text-align: right;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔒 Privacy Validation Report</h1>
                
                <div class="metric-card">
                    <div class="metric-title">Overall Risk Assessment</div>
                    <div>
                        <span class="risk-level risk-{result.overall_metrics.risk_level.lower()}">
                            {result.overall_metrics.risk_level} RISK
                        </span>
                        <span style="margin-left: 20px; font-size: 1.2em;">
                            Score: {result.overall_metrics.risk_score:.2%}
                        </span>
                    </div>
                </div>
                
                <h2>📊 Key Privacy Metrics</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                    <div class="metric-card">
                        <div class="metric-title">K-Anonymity</div>
                        <div class="metric-value">{result.overall_metrics.k_anonymity}</div>
                        <div style="color: #718096;">Minimum: {self.k_threshold}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">L-Diversity</div>
                        <div class="metric-value">{result.overall_metrics.l_diversity:.1f}</div>
                        <div style="color: #718096;">Minimum: {self.l_threshold}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">T-Closeness</div>
                        <div class="metric-value">{result.overall_metrics.t_closeness:.3f}</div>
                        <div style="color: #718096;">Lower is better</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">DP Epsilon</div>
                        <div class="metric-value">{result.overall_metrics.differential_privacy_epsilon}</div>
                        <div style="color: #718096;">Privacy budget</div>
                    </div>
                </div>
                
                <h2>✅ Privacy Guarantees</h2>
                <div>
        """
        
        for guarantee, met in result.privacy_guarantees.items():
            status_class = "guarantee-met" if met else "guarantee-not-met"
            status_icon = "✓" if met else "✗"
            html += f"""
                    <span class="guarantee {status_class}">
                        {status_icon} {guarantee.replace('_', ' ').title()}
                    </span>
            """
        
        html += """
                </div>
                
                <h2>🎯 L-Diversity Scores</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Sensitive Attribute</th>
                            <th>L-Diversity Score</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for attr, score in result.l_diversity_scores.items():
            status = "✅ Met" if score >= self.l_threshold else "❌ Not Met"
            html += f"""
                        <tr>
                            <td>{attr}</td>
                            <td>{score:.1f}</td>
                            <td>{status}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
                
                <h2>⚠️ Attribute Risk Analysis</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Attribute</th>
                            <th>Unique Values</th>
                            <th>Uniqueness Ratio</th>
                            <th>Entropy</th>
                            <th>Risk Level</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for attr, risk in result.attribute_risks.items():
            risk_color = {
                'HIGH': '#f56565',
                'MEDIUM': '#ed8936',
                'LOW': '#48bb78'
            }.get(risk['risk_level'], '#718096')
            
            html += f"""
                        <tr>
                            <td>{attr}</td>
                            <td>{risk['unique_values']}</td>
                            <td>{risk['uniqueness_ratio']:.2%}</td>
                            <td>{risk['entropy']:.2f}</td>
                            <td style="color: {risk_color}; font-weight: bold;">
                                {risk['risk_level']}
                            </td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
                
                <h2>💡 Recommendations</h2>
        """
        
        if result.overall_metrics.recommendations:
            for rec in result.overall_metrics.recommendations:
                html += f"""
                <div class="recommendation">
                    {rec}
                </div>
                """
        else:
            html += """
                <div style="color: #48bb78; padding: 20px; background: #f0fff4; border-radius: 10px;">
                    ✅ All privacy requirements are met. No additional actions required.
                </div>
            """
        
        html += f"""
                <div class="timestamp">
                    Report generated at: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Privacy report saved to {output_path}")
        
        return html