# Clinical Rule Validation Report
Generated on: 2025-09-08 17:16:42

## Overall Compliance: 0.999 (Excellent)
Sample Size: 1,172

## Violations Summary
- Total Violations: 29
- Critical Violations: 2
- Warning Violations: 0
- Violation Rate: 0.0014

## Age Diagnosis Incompatibility
Category Compliance Rate: 0.999

| Rule ID | Description | Status | Actual Rate | Expected | Violations |
|---------|-------------|--------|-------------|----------|------------|
| AD001 | 영아 심혈관 질환 배제 | ❌ FAILED | 0.0017 | 0.0000 | 2 |
| AD002 | 소아 퇴행성 질환 배제 | ✅ PASSED | 0.0000 | 0.0000 | 0 |
| AD003 | 고령자 선천성 질환 낮은 빈도 | ✅ PASSED | 0.0000 | 0.0100 | 0 |

## Gender Diagnosis Incompatibility
Category Compliance Rate: 1.000

| Rule ID | Description | Status | Actual Rate | Expected | Violations |
|---------|-------------|--------|-------------|----------|------------|
| GD001 | 남성 임신 관련 진단 배제 | ✅ PASSED | 0.0000 | 0.0000 | 0 |
| GD002 | 남성 부인과 질환 배제 | ✅ PASSED | 0.0000 | 0.0000 | 0 |
| GD003 | 여성 전립선 질환 배제 | ✅ PASSED | 0.0000 | 0.0000 | 0 |

## Ktas Outcome Consistency
Category Compliance Rate: 0.998

| Rule ID | Description | Status | Actual Rate | Expected | Violations |
|---------|-------------|--------|-------------|----------|------------|
| KO001 | KTAS 1급 귀가율 제한 | ✅ PASSED | 0.0009 | 0.0500 | 1 |
| KO002 | KTAS 5급 중환자실 입원 제한 | ✅ PASSED | 0.0009 | 0.0100 | 1 |
| KO003 | KTAS 1급 사망률 범위 | ✅ PASSED | 0.0034 | 0.1500 | 4 |

## Temporal Consistency
Category Compliance Rate: 1.000

| Rule ID | Description | Status | Actual Rate | Expected | Violations |
|---------|-------------|--------|-------------|----------|------------|
| TC001 | 방문시간 < 퇴실시간 | ✅ PASSED | 0.0000 | 0.0000 | 0 |
| TC002 | 과도한 응급실 체류 제한 | ✅ PASSED | 0.0000 | 0.0200 | 0 |
| TC003 | 퇴실시간 < 입원시간 | ✅ PASSED | 0.0000 | 0.0000 | 0 |

## Vital Signs Medical Range
Category Compliance Rate: 0.996

| Rule ID | Description | Status | Actual Rate | Expected | Violations |
|---------|-------------|--------|-------------|----------|------------|
| VS001 | 수축기혈압 의학적 범위 | ✅ PASSED | 0.0179 | 0.0200 | 21 |
| VS002 | 맥박수 의학적 범위 | ✅ PASSED | 0.0000 | 0.0100 | 0 |
| VS003 | 체온 의학적 범위 | ✅ PASSED | 0.0000 | 0.0050 | 0 |
| VS004 | 산소포화도 의학적 범위 | ✅ PASSED | 0.0000 | 0.0100 | 0 |

## Diagnosis Treatment Consistency
Category Compliance Rate: 1.000

| Rule ID | Description | Status | Actual Rate | Expected | Violations |
|---------|-------------|--------|-------------|----------|------------|
| DT001 | 심장 질환 - 내과 치료과 연관성 | ℹ️ INFO | 0.0000 | 0.2000 | 0 |
| DT002 | 외상 - 외과 치료과 연관성 | ℹ️ INFO | 0.0000 | 0.3000 | 0 |

## Recommendations
🔥 **Critical Issues Found:**
- 2 critical violations require immediate attention
- Review data generation logic for affected rules
- Consider additional validation constraints
