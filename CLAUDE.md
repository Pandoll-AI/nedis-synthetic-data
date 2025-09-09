# NEDIS 합성 데이터 생성 시스템 - 개발 가이드라인

## 핵심 원칙: 하드코딩 금지 및 동적 분석 우선

이 시스템은 **하드코딩된 분포나 가정을 완전히 배제**하고 **실제 데이터에서 동적으로 패턴을 학습**하는 것을 핵심 원칙으로 합니다.

---

## 🚫 하드코딩 금지 사항

### 1. **분포 하드코딩 금지**
```python
# ❌ 금지: 하드코딩된 분포
ktas_distribution = {
    '1': 0.05, '2': 0.15, '3': 0.30, '4': 0.35, '5': 0.15
}

# ✅ 허용: 동적 패턴 분석
ktas_probs = pattern_analyzer.get_hierarchical_ktas_distribution(
    region_code, hospital_type
)
```

### 2. **지역별 가중치 하드코딩 금지**
```python
# ❌ 금지: 수동으로 설정한 지역 가중치
region_weights = {
    '11': 1.2,  # 서울 높은 가중치
    '21': 1.1,  # 부산 중간 가중치
}

# ✅ 허용: 실제 데이터에서 학습
regional_patterns = pattern_analyzer.analyze_regional_patterns()
```

### 3. **병원 할당 규칙 하드코딩 금지**
```python
# ❌ 금지: 복잡한 중력 모델이나 거리 기반 가정
def gravity_model(distance, capacity):
    return (capacity ** 1.0) / (distance ** 2.0)

# ✅ 허용: 실제 환자 유동 패턴 분석
hospital_allocation = pattern_analyzer.analyze_hospital_allocation_patterns()
```

### 4. **시간 패턴 가정 금지**
```python
# ❌ 금지: 수동으로 만든 시간별 분포
hourly_weights = [0.3, 0.2, 0.1, ..., 1.5, 1.6]  # 24시간

# ✅ 허용: 실제 내원 시간 패턴 분석
temporal_patterns = pattern_analyzer.analyze_temporal_patterns()
```

---

## ✅ 동적 분석 접근 방법

### 1. **계층적 대안 전략**
모든 패턴 분석에는 계층적 대안을 구현해야 합니다:

```python
def get_hierarchical_ktas_distribution(region_code: str, hospital_type: str) -> Dict[str, float]:
    """
    계층적 KTAS 분포 조회:
    1단계: 소분류 (4자리 지역코드) + 병원유형
    2단계: 대분류 (첫 2자리) + 병원유형
    3단계: 전국 + 병원유형
    4단계: 전체 평균 (최종 대안)
    """
    # 1단계 시도
    detailed_key = f"{region_code}_{hospital_type}"
    if detailed_key in detailed_patterns:
        return detailed_patterns[detailed_key]
    
    # 2단계 시도  
    major_region = region_code[:2]
    major_key = f"{major_region}_{hospital_type}"
    if major_key in major_patterns:
        return major_patterns[major_key]
    
    # 3단계 시도
    if hospital_type in national_patterns:
        return national_patterns[hospital_type]
    
    # 4단계: 최종 대안
    return overall_pattern
```

### 2. **캐싱 전략**
데이터가 동일하면 분석을 재수행하지 않습니다:

```python
class AnalysisCache:
    def get_data_hash(self, db_manager: DatabaseManager, table_name: str) -> str:
        # 데이터 해시 계산 (행 수 + 샘플 데이터)
        pass
    
    def load_cached_analysis(self, analysis_type: str, data_hash: str) -> Optional[Dict]:
        # 캐시된 분석 결과 로드
        pass
    
    def save_analysis_cache(self, analysis_type: str, data_hash: str, results: Dict):
        # 분석 결과 캐시 저장
        pass
```

### 3. **최소 샘플 크기 검증**
통계적 유의성을 보장합니다:

```python
@dataclass
class PatternAnalysisConfig:
    min_sample_size: int = 10  # 최소 샘플 수
    confidence_threshold: float = 0.95
```

---

## 🏗️ 시스템 아키텍처

### 1. **PatternAnalyzer (핵심 모듈)**
```python
class PatternAnalyzer:
    def analyze_all_patterns(self) -> Dict[str, Any]:
        # 모든 패턴 분석 수행
        
    def analyze_hospital_allocation_patterns(self) -> Dict[str, Any]:
        # 지역별 병원 선택 패턴 분석
        
    def analyze_ktas_distributions(self) -> Dict[str, Any]:
        # 계층적 KTAS 분포 분석
        
    def get_hierarchical_ktas_distribution(self, region_code: str, hospital_type: str) -> Dict[str, float]:
        # 계층적 대안을 통한 KTAS 분포 조회
```

### 2. **VectorizedPatientGenerator (업데이트됨)**
- 동적 패턴 사용으로 완전히 리팩토링
- 중력 모델 제거, 지역 기반 할당 사용
- 계층적 KTAS 생성 구현

### 3. **TemporalPatternAssigner (업데이트됨)**
- 동적 시간 패턴 분석 사용
- 하드코딩된 시간 분포 제거

---

## 📋 개발 체크리스트

새로운 기능 개발 시 다음을 확인하세요:

### ✅ 필수 검증 항목
- [ ] 하드코딩된 확률이나 가중치가 없는가?
- [ ] 실제 데이터에서 패턴을 학습하는가?
- [ ] 계층적 대안이 구현되어 있는가?
- [ ] 최소 샘플 크기를 검증하는가?
- [ ] 캐싱 메커니즘을 사용하는가?
- [ ] 에러 처리 및 로깅이 적절한가?

### 🧪 테스트 요구사항
```python
def test_no_hardcoded_distributions():
    """하드코딩된 분포가 없는지 확인"""
    # 코드에서 하드코딩된 확률/가중치 검색
    pass

def test_hierarchical_fallback():
    """계층적 대안이 동작하는지 확인"""
    # 각 단계별 대안이 올바르게 동작하는지 검증
    pass

def test_cache_efficiency():
    """캐싱이 효율적으로 동작하는지 확인"""
    # 동일 데이터에 대해 캐시 사용 여부 검증
    pass
```

---

## 🔧 설정 파일 예시

```python
# config/pattern_analysis.yaml
pattern_analysis:
  cache_dir: "cache/patterns"
  use_cache: true
  min_sample_size: 10
  confidence_threshold: 0.95
  hierarchical_fallback: true
  
hospital_allocation:
  method: "regional_based"  # not "gravity_model"
  fallback_strategy: "hierarchical"
  
ktas_generation:
  hierarchy_levels: 4
  fallback_method: "national_average"
```

---

## 📖 참고 문서

1. **패턴 분석 가이드**: `docs/pattern_analysis_guide.md`
2. **캐싱 전략**: `docs/caching_strategy.md`
3. **계층적 대안**: `docs/hierarchical_fallback.md`
4. **성능 최적화**: `docs/performance_optimization.md`

---

## 🚨 주의사항

1. **절대 하드코딩하지 마세요**: 모든 분포와 패턴은 실제 데이터에서 학습해야 합니다.
2. **계층적 대안은 필수**: 데이터가 부족한 경우를 대비한 대안이 항상 있어야 합니다.
3. **캐싱은 성능의 핵심**: 동일한 분석을 반복하지 않도록 캐싱을 적극 활용하세요.
4. **로깅과 검증**: 모든 패턴 분석 결과를 로깅하고 검증하세요.
5. **테스트 커버리지**: 하드코딩 검증 테스트는 필수입니다.

---

**Remember: 데이터가 진실을 말하게 하고, 가정은 최소화하세요.**