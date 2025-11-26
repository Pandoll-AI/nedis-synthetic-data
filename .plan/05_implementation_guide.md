# NEDIS 합성 데이터 생성 구현 가이드

## 시작하기 전에

이 구현 가이드는 concept.md에 정의된 7개 Phase를 실제로 구현하기 위한 단계별 지침서입니다. 각 Phase별로 구체적인 코드 예제, 테스트 방법, 품질 체크포인트를 제공합니다.

### 선행 조건
- Python 3.9+ 설치
- DuckDB 0.9+ 설치  
- 16GB+ RAM 개발 환경
- NEDIS 2017 원본 데이터 접근 권한

---

## Phase 1: 데이터 프로파일링 및 메타데이터 추출

### 1.1 환경 설정 및 프로젝트 초기화

#### 프로젝트 구조 생성
```bash
# 프로젝트 디렉토리 생성
mkdir nedis-synthetic
cd nedis-synthetic

# 디렉토리 구조 생성
mkdir -p {config,src/{core,population,temporal,allocation,clinical,validation,optimization,utils},data/{raw,processed,synthetic},tests,logs,outputs}

# 가상환경 설정
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 필수 패키지 설치
pip install duckdb pandas numpy scipy scikit-learn scikit-optimize torch torchvision torchaudio matplotlib seaborn pyyaml pytest black flake8 mypy tensorboard
```

#### requirements.txt 생성
```txt
duckdb>=0.9.0
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.1.0
scikit-optimize>=0.9.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
mypy>=0.991
tqdm>=4.64.0
tensorboard>=2.13.0
```

### 1.2 핵심 인프라 구현

#### core/database.py 구현
```python
import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Any, List
import logging

class DatabaseManager:
    def __init__(self, db_path: str = "nedis.duckdb"):
        """DuckDB 연결 관리자 초기화"""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._setup_schemas()
        self.logger = logging.getLogger(__name__)
        
    def _setup_schemas(self):
        """데이터베이스 스키마 초기화"""
        schema_sql = Path("sql/create_schemas.sql").read_text()
        self.conn.execute(schema_sql)
        self.logger.info("Database schemas created successfully")
    
    def execute_query(self, query: str, params: Optional[List] = None) -> Any:
        """SQL 쿼리 실행 with 로깅"""
        try:
            self.logger.debug(f"Executing query: {query[:100]}...")
            if params:
                result = self.conn.execute(query, params)
            else:
                result = self.conn.execute(query)
            return result
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
            
    def fetch_dataframe(self, query: str, params: Optional[List] = None) -> pd.DataFrame:
        """쿼리 결과를 DataFrame으로 반환"""
        result = self.execute_query(query, params)
        return result.fetchdf()
        
    def batch_insert(self, table: str, data: pd.DataFrame, batch_size: int = 10000):
        """대용량 데이터 배치 삽입"""
        total_rows = len(data)
        self.logger.info(f"Inserting {total_rows} rows into {table}")
        
        for i in range(0, total_rows, batch_size):
            batch = data[i:i + batch_size]
            self.conn.execute(f"INSERT INTO {table} SELECT * FROM batch")
            self.logger.debug(f"Inserted batch {i//batch_size + 1}")
```

#### core/config.py 구현
```python
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

class ConfigManager:
    def __init__(self, config_path: str = "config/generation_params.yaml"):
        """설정 관리자 초기화"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'population': {
                'dirichlet_alpha': 1.0,
                'target_total_records': 9_200_000
            },
            'temporal': {
                'nhpp_seasonal_weights': {
                    'spring': 0.25, 'summer': 0.25,
                    'fall': 0.25, 'winter': 0.25
                }
            },
            'allocation': {
                'gravity_gamma': 1.5,
                'ipf_tolerance': 0.001,
                'ipf_max_iterations': 100
            },
            'clinical': {
                'ktas_smoothing': 1.0,
                'diagnosis_min_count': 10
            },
            'validation': {
                'statistical_alpha': 0.05,
                'privacy_k_anonymity': 5
            }
        }
        
    def get(self, key_path: str, default: Any = None) -> Any:
        """점 표기법으로 중첩된 설정값 조회"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, default)
            if value is None:
                return default
        return value
        
    def update(self, key_path: str, value: Any):
        """설정값 런타임 업데이트"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
        self.logger.info(f"Updated config: {key_path} = {value}")
```

### 1.3 Phase 1 구현: 데이터 프로파일링

#### population/profiler.py 구현
```python
import pandas as pd
import numpy as np
from typing import Dict, Any
from core.database import DatabaseManager
from core.config import ConfigManager
import logging

class NEDISProfiler:
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """NEDIS 데이터 프로파일러 초기화"""
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def extract_population_margins(self) -> bool:
        """인구학적 마진 추출 및 저장"""
        self.logger.info("Starting population margins extraction")
        
        query = """
        INSERT INTO nedis_meta.population_margins
        SELECT 
            pat_do_cd,
            pat_age_gr,
            pat_sex,
            COUNT(*) as yearly_visits,
            -- 계절별 가중치 계산
            SUM(CASE WHEN MONTH(CAST(vst_dt AS DATE)) IN (3,4,5) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_spring,
            SUM(CASE WHEN MONTH(CAST(vst_dt AS DATE)) IN (6,7,8) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_summer,
            SUM(CASE WHEN MONTH(CAST(vst_dt AS DATE)) IN (9,10,11) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_fall,
            SUM(CASE WHEN MONTH(CAST(vst_dt AS DATE)) IN (12,1,2) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_winter,
            -- 요일별 가중치 계산  
            SUM(CASE WHEN DAYOFWEEK(CAST(vst_dt AS DATE)) IN (1,7) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as weekend_weight,
            (1 - SUM(CASE WHEN DAYOFWEEK(CAST(vst_dt AS DATE)) IN (1,7) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)) as weekday_weight
        FROM nedis_original.nedis2017
        WHERE pat_do_cd != '' AND pat_age_gr != '' AND pat_sex IN ('M', 'F')
        GROUP BY pat_do_cd, pat_age_gr, pat_sex
        """
        
        try:
            self.db.execute_query(query)
            
            # 결과 검증
            count_query = "SELECT COUNT(*) as total FROM nedis_meta.population_margins"
            total_combinations = self.db.fetch_dataframe(count_query)['total'][0]
            
            self.logger.info(f"Created {total_combinations} population margin combinations")
            return True
            
        except Exception as e:
            self.logger.error(f"Population margins extraction failed: {e}")
            return False
            
    def extract_hospital_statistics(self) -> bool:
        """병원별 용량 통계 추출"""
        self.logger.info("Starting hospital statistics extraction")
        
        # 먼저 기본 병원 정보 삽입
        hospital_info_query = """
        INSERT INTO nedis_meta.hospital_capacity (emorg_cd, hospname, gubun, adr)
        SELECT DISTINCT emorg_cd, hospname, gubun, adr
        FROM nedis_original.nedis2017
        WHERE emorg_cd != ''
        """
        
        # 그 다음 통계 정보 업데이트
        stats_query = """
        UPDATE nedis_meta.hospital_capacity 
        SET 
            daily_capacity_mean = stats.daily_capacity_mean,
            daily_capacity_std = stats.daily_capacity_std,
            ktas1_capacity = stats.ktas1_capacity,
            ktas2_capacity = stats.ktas2_capacity
        FROM (
            WITH daily_counts AS (
                SELECT 
                    emorg_cd,
                    vst_dt,
                    COUNT(*) as daily_visits,
                    SUM(CASE WHEN ktas_fstu = '1' THEN 1 ELSE 0 END) as ktas1_count,
                    SUM(CASE WHEN ktas_fstu = '2' THEN 1 ELSE 0 END) as ktas2_count
                FROM nedis_original.nedis2017
                GROUP BY emorg_cd, vst_dt
            )
            SELECT 
                emorg_cd,
                ROUND(AVG(daily_visits))::INTEGER as daily_capacity_mean,
                ROUND(STDDEV(daily_visits))::INTEGER as daily_capacity_std,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ktas1_count))::INTEGER as ktas1_capacity,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ktas2_count))::INTEGER as ktas2_capacity
            FROM daily_counts
            GROUP BY emorg_cd
        ) AS stats
        WHERE nedis_meta.hospital_capacity.emorg_cd = stats.emorg_cd
        """
        
        try:
            self.db.execute_query(hospital_info_query)
            self.db.execute_query(stats_query)
            
            # 결과 검증
            verify_query = """
            SELECT COUNT(*) as hospitals_with_stats 
            FROM nedis_meta.hospital_capacity 
            WHERE daily_capacity_mean IS NOT NULL
            """
            result = self.db.fetch_dataframe(verify_query)
            hospitals_count = result['hospitals_with_stats'][0]
            
            self.logger.info(f"Updated statistics for {hospitals_count} hospitals")
            return True
            
        except Exception as e:
            self.logger.error(f"Hospital statistics extraction failed: {e}")
            return False
            
    def generate_basic_report(self) -> Dict[str, Any]:
        """기본 데이터 프로파일 리포트 생성"""
        report = {}
        
        # 전체 레코드 수
        total_query = "SELECT COUNT(*) as total FROM nedis_original.nedis2017"
        report['total_records'] = self.db.fetch_dataframe(total_query)['total'][0]
        
        # 시도별 분포
        region_query = """
        SELECT pat_do_cd, COUNT(*) as count
        FROM nedis_original.nedis2017
        GROUP BY pat_do_cd
        ORDER BY count DESC
        LIMIT 10
        """
        report['top_regions'] = self.db.fetch_dataframe(region_query).to_dict('records')
        
        # KTAS 분포
        ktas_query = """
        SELECT ktas_fstu, COUNT(*) as count, 
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM nedis_original.nedis2017
        WHERE ktas_fstu != ''
        GROUP BY ktas_fstu
        ORDER BY ktas_fstu
        """
        report['ktas_distribution'] = self.db.fetch_dataframe(ktas_query).to_dict('records')
        
        # 결측값 분석
        missing_query = """
        SELECT 
            'vst_sbp' as field, SUM(CASE WHEN vst_sbp = -1 THEN 1 ELSE 0 END) as missing_count
        UNION ALL
        SELECT 
            'vst_dbp', SUM(CASE WHEN vst_dbp = -1 THEN 1 ELSE 0 END)
        UNION ALL
        SELECT 
            'vst_per_pu', SUM(CASE WHEN vst_per_pu = -1 THEN 1 ELSE 0 END)
        FROM nedis_original.nedis2017
        """
        report['missing_values'] = self.db.fetch_dataframe(missing_query).to_dict('records')
        
        self.logger.info("Basic data profile report generated")
        return report
```

#### Phase 1 실행 스크립트
```python
# scripts/run_phase1.py
import logging
from core.database import DatabaseManager
from core.config import ConfigManager
from population.profiler import NEDISProfiler
import json
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/phase1.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== Phase 1: Data Profiling Started ===")
    
    # 초기화
    db_manager = DatabaseManager()
    config = ConfigManager()
    profiler = NEDISProfiler(db_manager, config)
    
    # 실행
    success_steps = []
    
    # Step 1: 인구학적 마진 추출
    if profiler.extract_population_margins():
        success_steps.append("population_margins")
        logger.info("✓ Population margins extraction completed")
    else:
        logger.error("✗ Population margins extraction failed")
        return False
        
    # Step 2: 병원 통계 추출
    if profiler.extract_hospital_statistics():
        success_steps.append("hospital_statistics")
        logger.info("✓ Hospital statistics extraction completed")
    else:
        logger.error("✗ Hospital statistics extraction failed")
        return False
        
    # Step 3: 기본 리포트 생성
    try:
        report = profiler.generate_basic_report()
        
        # 리포트 저장
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        with open(outputs_dir / "phase1_data_profile.json", 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        success_steps.append("basic_report")
        logger.info("✓ Basic data profile report generated")
        
    except Exception as e:
        logger.error(f"✗ Basic report generation failed: {e}")
        return False
    
    logger.info(f"=== Phase 1 Completed Successfully ===")
    logger.info(f"Completed steps: {success_steps}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

### Phase 1 테스트 및 검증

#### tests/test_phase1.py
```python
import pytest
import pandas as pd
from core.database import DatabaseManager
from core.config import ConfigManager
from population.profiler import NEDISProfiler

class TestPhase1:
    @pytest.fixture
    def setup_test_env(self):
        """테스트 환경 설정"""
        # 테스트용 인메모리 데이터베이스
        db_manager = DatabaseManager(":memory:")
        config = ConfigManager()
        
        # 테스트 데이터 생성
        test_data = pd.DataFrame({
            'pat_do_cd': ['11', '41', '28'] * 100,
            'pat_age_gr': ['20', '30', '40'] * 100,
            'pat_sex': ['M', 'F'] * 150,
            'vst_dt': ['20170315'] * 300,
            'emorg_cd': ['A001', 'B002', 'C003'] * 100,
            'hospname': ['서울대병원', '삼성서울병원', '아산병원'] * 100,
            'gubun': ['권역센터'] * 300,
            'adr': ['서울', '서울', '서울'] * 100,
            'ktas_fstu': ['1', '2', '3'] * 100
        })
        
        # 테스트 테이블 생성 및 데이터 삽입
        db_manager.conn.execute("CREATE TABLE nedis_original.nedis2017 AS SELECT * FROM test_data")
        
        profiler = NEDISProfiler(db_manager, config)
        
        return {
            'db_manager': db_manager,
            'config': config, 
            'profiler': profiler,
            'test_data': test_data
        }
    
    def test_population_margins_extraction(self, setup_test_env):
        """인구학적 마진 추출 테스트"""
        env = setup_test_env
        profiler = env['profiler']
        db = env['db_manager']
        
        # 실행
        result = profiler.extract_population_margins()
        assert result == True
        
        # 결과 검증
        margins_df = db.fetch_dataframe("SELECT * FROM nedis_meta.population_margins")
        
        # 기본 검증
        assert len(margins_df) > 0
        assert all(col in margins_df.columns for col in [
            'pat_do_cd', 'pat_age_gr', 'pat_sex', 'yearly_visits',
            'seasonal_weight_spring', 'weekend_weight'
        ])
        
        # 비즈니스 로직 검증  
        assert all(margins_df['yearly_visits'] > 0)
        assert all(margins_df['seasonal_weight_spring'] >= 0)
        assert all(margins_df['seasonal_weight_spring'] <= 1)
        
    def test_hospital_statistics_extraction(self, setup_test_env):
        """병원 통계 추출 테스트"""
        env = setup_test_env
        profiler = env['profiler']
        db = env['db_manager']
        
        result = profiler.extract_hospital_statistics()
        assert result == True
        
        # 결과 검증
        hospital_df = db.fetch_dataframe("SELECT * FROM nedis_meta.hospital_capacity")
        
        assert len(hospital_df) > 0
        assert all(col in hospital_df.columns for col in [
            'emorg_cd', 'hospname', 'daily_capacity_mean'
        ])
        
    def test_data_quality_checks(self, setup_test_env):
        """데이터 품질 체크"""
        env = setup_test_env
        profiler = env['profiler']
        
        # 마진 추출 실행
        profiler.extract_population_margins()
        
        # 품질 체크
        db = env['db_manager']
        
        # 1. 모든 조합이 양수 방문수를 가지는가?
        query = "SELECT COUNT(*) FROM nedis_meta.population_margins WHERE yearly_visits <= 0"
        zero_visits = db.fetch_dataframe(query).iloc[0, 0]
        assert zero_visits == 0
        
        # 2. 계절별 가중치 합이 1인가?
        query = """
        SELECT AVG(seasonal_weight_spring + seasonal_weight_summer + 
                   seasonal_weight_fall + seasonal_weight_winter) as avg_seasonal_sum
        FROM nedis_meta.population_margins
        """
        avg_sum = db.fetch_dataframe(query).iloc[0, 0]
        assert abs(avg_sum - 1.0) < 0.01  # 부동소수점 오차 허용
```

### Phase 1 체크포인트

Phase 1 완료 후 다음 사항을 확인하세요:

#### 데이터베이스 테이블 검증
```sql
-- 1. population_margins 테이블 확인
SELECT COUNT(*) FROM nedis_meta.population_margins;
SELECT pat_do_cd, COUNT(*) FROM nedis_meta.population_margins GROUP BY pat_do_cd;

-- 2. hospital_capacity 테이블 확인  
SELECT COUNT(*) FROM nedis_meta.hospital_capacity WHERE daily_capacity_mean IS NOT NULL;

-- 3. 데이터 품질 검증
SELECT 
    MIN(yearly_visits) as min_visits,
    MAX(yearly_visits) as max_visits,
    AVG(yearly_visits) as avg_visits
FROM nedis_meta.population_margins;
```

#### 성공 기준
- [ ] `nedis_meta.population_margins` 테이블에 1000+ 조합 생성
- [ ] `nedis_meta.hospital_capacity` 테이블에 모든 병원 통계 포함
- [ ] 계절별 가중치 합계가 1.0 (±0.01 오차 허용)
- [ ] 모든 yearly_visits > 0
- [ ] 기본 데이터 프로파일 리포트 JSON 파일 생성
- [ ] 단위 테스트 모두 통과
- [ ] 처리 시간 < 30분 (표준 개발 환경)

---

## Phase 2: 인구 및 시간 패턴 생성

Phase 2는 Dirichlet-Multinomial 모델과 NHPP(비균질 포아송 과정)를 사용하여 현실적인 시간 패턴을 가진 인구 볼륨을 생성합니다.

### 2.1 인구 볼륨 생성기 구현

#### population/generator.py
```python
import numpy as np
import pandas as pd
from scipy.stats import dirichlet, multinomial
from typing import Dict, List, Tuple
from core.database import DatabaseManager
from core.config import ConfigManager
import logging
from tqdm import tqdm

class PopulationVolumeGenerator:
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """인구 볼륨 생성기 초기화"""
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alpha = config.get('population.dirichlet_alpha', 1.0)
        
    def generate_yearly_volumes(self, target_total: int = 9_200_000) -> bool:
        """연간 볼륨 생성 및 저장"""
        self.logger.info(f"Starting yearly volume generation (target: {target_total:,})")
        
        try:
            # 1. 원본 시도별 비율 계산
            region_proportions = self._get_regional_proportions()
            
            # 2. 시도별 목표 볼륨 할당
            region_targets = self._allocate_regional_targets(region_proportions, target_total)
            
            # 3. 시도별 Dirichlet-Multinomial 생성
            all_synthetic_volumes = []
            
            for region_code, target_volume in tqdm(region_targets.items(), desc="Processing regions"):
                region_volumes = self._generate_region_volumes(region_code, target_volume)
                all_synthetic_volumes.extend(region_volumes)
                
            # 4. 결과를 DataFrame으로 변환하여 저장
            volumes_df = pd.DataFrame(all_synthetic_volumes, columns=[
                'pat_do_cd', 'pat_age_gr', 'pat_sex', 'synthetic_yearly_count'
            ])
            
            self._save_yearly_volumes(volumes_df)
            
            # 5. 결과 검증
            total_generated = volumes_df['synthetic_yearly_count'].sum()
            self.logger.info(f"Generated {total_generated:,} total records (target: {target_total:,})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Yearly volume generation failed: {e}")
            return False
            
    def _get_regional_proportions(self) -> Dict[str, float]:
        """원본 데이터에서 시도별 비율 계산"""
        query = """
        SELECT 
            pat_do_cd,
            SUM(yearly_visits) as total_visits
        FROM nedis_meta.population_margins
        GROUP BY pat_do_cd
        """
        
        region_data = self.db.fetch_dataframe(query)
        total_visits = region_data['total_visits'].sum()
        
        proportions = {}
        for _, row in region_data.iterrows():
            proportions[row['pat_do_cd']] = row['total_visits'] / total_visits
            
        self.logger.info(f"Calculated proportions for {len(proportions)} regions")
        return proportions
        
    def _allocate_regional_targets(self, proportions: Dict[str, float], 
                                 total_target: int) -> Dict[str, int]:
        """시도별 목표 볼륨 할당"""
        targets = {}
        allocated_total = 0
        
        # 비례 할당
        for region, proportion in proportions.items():
            target = int(total_target * proportion)
            targets[region] = target
            allocated_total += target
            
        # 반올림으로 인한 차이 보정
        difference = total_target - allocated_total
        if difference != 0:
            # 가장 큰 지역에 차이만큼 할당/차감
            largest_region = max(targets.keys(), key=lambda k: targets[k])
            targets[largest_region] += difference
            
        self.logger.info(f"Allocated targets: total={sum(targets.values()):,}")
        return targets
        
    def _generate_region_volumes(self, region_code: str, target_volume: int) -> List[Tuple]:
        """특정 시도의 연령×성별 볼륨 생성"""
        
        # 해당 지역의 원본 분포 조회
        query = """
        SELECT pat_age_gr, pat_sex, yearly_visits
        FROM nedis_meta.population_margins
        WHERE pat_do_cd = ?
        ORDER BY pat_age_gr, pat_sex
        """
        
        region_data = self.db.fetch_dataframe(query, [region_code])
        
        if len(region_data) == 0:
            self.logger.warning(f"No data found for region {region_code}")
            return []
            
        # Dirichlet 파라미터 설정
        observed_counts = region_data['yearly_visits'].values
        alpha_vector = observed_counts + self.alpha
        
        # Dirichlet 분포에서 확률 벡터 샘플링
        probabilities = dirichlet.rvs(alpha_vector)[0]
        
        # Multinomial 분포로 개수 생성
        synthetic_counts = multinomial.rvs(target_volume, probabilities)
        
        # 결과 구성
        results = []
        for i, (_, row) in enumerate(region_data.iterrows()):
            results.append((
                region_code,
                row['pat_age_gr'],
                row['pat_sex'],
                synthetic_counts[i]
            ))
            
        return results
        
    def _save_yearly_volumes(self, volumes_df: pd.DataFrame):
        """연간 볼륨 데이터베이스 저장"""
        # 기존 데이터 삭제
        self.db.execute_query("DELETE FROM nedis_synthetic.yearly_volumes")
        
        # 새 데이터 삽입
        self.db.conn.execute("INSERT INTO nedis_synthetic.yearly_volumes SELECT * FROM volumes_df")
        
        self.logger.info(f"Saved {len(volumes_df)} yearly volume records")
```

### 2.2 시간 패턴 생성기 구현

#### temporal/nhpp_generator.py  
```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import calendar
from core.database import DatabaseManager
from core.config import ConfigManager
import logging
from tqdm import tqdm

class NHPPTemporalGenerator:
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """NHPP 시간 패턴 생성기 초기화"""
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 공휴일 정의 (2017년 기준)
        self.holidays_2017 = [
            '20170101', '20170127', '20170128', '20170129', '20170130',  # 신정, 설날 연휴
            '20170301',  # 3.1절
            '20170503', '20170509',  # 어린이날, 부처님오신날  
            '20170506',  # 어린이날 대체공휴일
            '20170815',  # 광복절
            '20171003', '20171004', '20171005', '20171006',  # 추석 연휴
            '20171009',  # 한글날
            '20171225'   # 성탄절
        ]
        
    def generate_daily_events(self, year: int = 2017) -> bool:
        """일별 이벤트 분해 생성"""
        self.logger.info(f"Starting daily event generation for year {year}")
        
        try:
            # 1. 연간 볼륨 데이터 로드
            yearly_volumes = self._load_yearly_volumes()
            
            # 2. 365일 날짜 리스트 생성
            date_list = self._generate_date_list(year)
            
            # 3. 각 (시도, 연령, 성별) 조합별로 일별 분해
            all_daily_volumes = []
            
            for _, volume_row in tqdm(yearly_volumes.iterrows(), 
                                    total=len(yearly_volumes),
                                    desc="Processing volume combinations"):
                daily_volumes = self._decompose_to_daily(volume_row, date_list)
                all_daily_volumes.extend(daily_volumes)
                
            # 4. 결과 저장
            daily_df = pd.DataFrame(all_daily_volumes, columns=[
                'vst_dt', 'pat_do_cd', 'pat_age_gr', 'pat_sex', 
                'synthetic_daily_count', 'lambda_value'
            ])
            
            self._save_daily_volumes(daily_df)
            
            # 5. 검증
            total_daily = daily_df['synthetic_daily_count'].sum()
            total_yearly = yearly_volumes['synthetic_yearly_count'].sum()
            
            self.logger.info(f"Daily total: {total_daily:,}, Yearly total: {total_yearly:,}")
            self.logger.info(f"Difference: {abs(total_daily - total_yearly):,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Daily event generation failed: {e}")
            return False
            
    def _load_yearly_volumes(self) -> pd.DataFrame:
        """연간 볼륨 데이터 로드"""
        query = """
        SELECT yv.*, pm.seasonal_weight_spring, pm.seasonal_weight_summer,
               pm.seasonal_weight_fall, pm.seasonal_weight_winter,
               pm.weekday_weight, pm.weekend_weight
        FROM nedis_synthetic.yearly_volumes yv
        JOIN nedis_meta.population_margins pm 
        ON yv.pat_do_cd = pm.pat_do_cd 
        AND yv.pat_age_gr = pm.pat_age_gr 
        AND yv.pat_sex = pm.pat_sex
        """
        return self.db.fetch_dataframe(query)
        
    def _generate_date_list(self, year: int) -> List[str]:
        """연도별 날짜 리스트 생성 (YYYYMMDD 형식)"""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        date_list = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            date_list.append(date_str)
            current_date += timedelta(days=1)
            
        return date_list
        
    def _decompose_to_daily(self, volume_row: pd.Series, date_list: List[str]) -> List[tuple]:
        """연간 볼륨을 일별로 분해"""
        yearly_count = volume_row['synthetic_yearly_count']
        base_lambda = yearly_count / 365.0
        
        results = []
        daily_counts = []
        lambda_values = []
        
        # 1. 각 날짜별 강도 계산
        for date_str in date_list:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            
            # 계절 가중치
            month = date_obj.month
            if month in [3, 4, 5]:  # 봄
                seasonal_weight = volume_row['seasonal_weight_spring']
            elif month in [6, 7, 8]:  # 여름  
                seasonal_weight = volume_row['seasonal_weight_summer']
            elif month in [9, 10, 11]:  # 가을
                seasonal_weight = volume_row['seasonal_weight_fall']
            else:  # 겨울
                seasonal_weight = volume_row['seasonal_weight_winter']
                
            # 요일 가중치
            weekday = date_obj.weekday()  # 0=Monday, 6=Sunday
            if weekday in [5, 6]:  # 토요일, 일요일
                weekday_weight = volume_row['weekend_weight']
            else:
                weekday_weight = volume_row['weekday_weight']
                
            # 공휴일 가중치
            holiday_weight = 1.2 if date_str in self.holidays_2017 else 1.0
            
            # 최종 강도 계산
            lambda_t = base_lambda * seasonal_weight * weekday_weight * holiday_weight
            lambda_values.append(lambda_t)
            
            # 포아송 샘플링
            daily_count = np.random.poisson(lambda_t)
            daily_counts.append(daily_count)
            
        # 2. Rescaling (연간 총합 맞추기)
        total_generated = sum(daily_counts)
        if total_generated > 0:
            scaling_factor = yearly_count / total_generated
            daily_counts = [max(0, int(count * scaling_factor + 0.5)) for count in daily_counts]
            
        # 3. 결과 구성
        for i, date_str in enumerate(date_list):
            results.append((
                date_str,
                volume_row['pat_do_cd'],
                volume_row['pat_age_gr'], 
                volume_row['pat_sex'],
                daily_counts[i],
                lambda_values[i]
            ))
            
        return results
        
    def _save_daily_volumes(self, daily_df: pd.DataFrame):
        """일별 볼륨 저장"""
        # 기존 데이터 삭제
        self.db.execute_query("DELETE FROM nedis_synthetic.daily_volumes")
        
        # 배치 단위로 삽입 (메모리 효율성)
        batch_size = 50000
        total_rows = len(daily_df)
        
        for i in range(0, total_rows, batch_size):
            batch = daily_df[i:i + batch_size]
            self.db.conn.execute("INSERT INTO nedis_synthetic.daily_volumes SELECT * FROM batch")
            
            if i % (batch_size * 10) == 0:
                self.logger.info(f"Inserted {i:,} / {total_rows:,} daily volume records")
                
        self.logger.info(f"Completed saving {total_rows:,} daily volume records")
```

### 2.3 Phase 2 실행 및 검증

#### scripts/run_phase2.py
```python
import logging
from core.database import DatabaseManager
from core.config import ConfigManager
from population.generator import PopulationVolumeGenerator
from temporal.nhpp_generator import NHPPTemporalGenerator
import time

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/phase2.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== Phase 2: Population & Temporal Pattern Generation Started ===")
    
    # 초기화
    db_manager = DatabaseManager()
    config = ConfigManager()
    
    # Step 1: 인구 볼륨 생성
    logger.info("Step 1: Generating yearly population volumes...")
    start_time = time.time()
    
    volume_generator = PopulationVolumeGenerator(db_manager, config)
    target_records = config.get('population.target_total_records', 9_200_000)
    
    if not volume_generator.generate_yearly_volumes(target_records):
        logger.error("✗ Yearly volume generation failed")
        return False
        
    step1_time = time.time() - start_time
    logger.info(f"✓ Yearly volumes generated in {step1_time:.2f} seconds")
    
    # Step 2: 일별 시간 패턴 생성
    logger.info("Step 2: Generating daily temporal patterns...")
    start_time = time.time()
    
    temporal_generator = NHPPTemporalGenerator(db_manager, config)
    
    if not temporal_generator.generate_daily_events(2017):
        logger.error("✗ Daily event generation failed")
        return False
        
    step2_time = time.time() - start_time
    logger.info(f"✓ Daily patterns generated in {step2_time:.2f} seconds")
    
    # 검증
    logger.info("Step 3: Validating generated data...")
    validation_success = validate_phase2_results(db_manager)
    
    if validation_success:
        logger.info("=== Phase 2 Completed Successfully ===")
        return True
    else:
        logger.error("=== Phase 2 Validation Failed ===")
        return False

def validate_phase2_results(db: DatabaseManager) -> bool:
    """Phase 2 결과 검증"""
    logger = logging.getLogger(__name__)
    
    try:
        # 1. 연간 총합 일치 확인
        yearly_total_query = "SELECT SUM(synthetic_yearly_count) FROM nedis_synthetic.yearly_volumes"
        yearly_total = db.fetch_dataframe(yearly_total_query).iloc[0, 0]
        
        daily_total_query = "SELECT SUM(synthetic_daily_count) FROM nedis_synthetic.daily_volumes"
        daily_total = db.fetch_dataframe(daily_total_query).iloc[0, 0]
        
        difference_pct = abs(yearly_total - daily_total) / yearly_total * 100
        logger.info(f"Total validation: Yearly={yearly_total:,}, Daily={daily_total:,}, Diff={difference_pct:.2f}%")
        
        if difference_pct > 1.0:  # 1% 이상 차이나면 실패
            logger.error("Total count validation failed")
            return False
            
        # 2. 일별 분포 검증 (극값 체크)
        daily_stats_query = """
        SELECT 
            MIN(synthetic_daily_count) as min_daily,
            MAX(synthetic_daily_count) as max_daily,
            AVG(synthetic_daily_count) as avg_daily,
            COUNT(*) as total_records
        FROM nedis_synthetic.daily_volumes
        """
        stats = db.fetch_dataframe(daily_stats_query).iloc[0]
        
        logger.info(f"Daily stats: min={stats['min_daily']}, max={stats['max_daily']}, avg={stats['avg_daily']:.1f}")
        
        # 3. 음수 체크
        negative_count_query = "SELECT COUNT(*) FROM nedis_synthetic.daily_volumes WHERE synthetic_daily_count < 0"
        negative_count = db.fetch_dataframe(negative_count_query).iloc[0, 0]
        
        if negative_count > 0:
            logger.error(f"Found {negative_count} negative daily counts")
            return False
            
        # 4. 날짜 범위 체크
        date_range_query = """
        SELECT 
            MIN(vst_dt) as min_date,
            MAX(vst_dt) as max_date,
            COUNT(DISTINCT vst_dt) as unique_dates
        FROM nedis_synthetic.daily_volumes
        """
        date_stats = db.fetch_dataframe(date_range_query).iloc[0]
        
        if date_stats['unique_dates'] != 365:
            logger.error(f"Expected 365 unique dates, got {date_stats['unique_dates']}")
            return False
            
        logger.info("✓ All validation checks passed")
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

### Phase 2 성능 최적화 팁

#### 메모리 효율적인 처리
```python
# 청크 단위 처리로 메모리 사용량 제한
def process_volumes_in_chunks(self, chunk_size: int = 1000):
    """볼륨 데이터를 청크 단위로 처리"""
    offset = 0
    
    while True:
        query = f"""
        SELECT * FROM nedis_synthetic.yearly_volumes
        LIMIT {chunk_size} OFFSET {offset}
        """
        
        chunk_data = self.db.fetch_dataframe(query)
        if len(chunk_data) == 0:
            break
            
        # 청크 처리
        for _, row in chunk_data.iterrows():
            yield row
            
        offset += chunk_size
```

#### 병렬 처리 (선택사항)
```python
from multiprocessing import Pool, cpu_count

def parallel_daily_generation(self, yearly_volumes: pd.DataFrame):
    """병렬로 일별 생성 처리"""
    n_workers = min(cpu_count(), 8)  # 최대 8개 프로세스
    
    # 작업 분할
    chunk_size = len(yearly_volumes) // n_workers
    chunks = [yearly_volumes[i:i+chunk_size] for i in range(0, len(yearly_volumes), chunk_size)]
    
    with Pool(n_workers) as pool:
        results = pool.map(self._process_chunk, chunks)
        
    # 결과 병합
    all_results = []
    for chunk_result in results:
        all_results.extend(chunk_result)
        
    return all_results
```

### Phase 2 체크포인트

Phase 2 완료 후 확인해야 할 사항들:

#### 데이터 검증 쿼리
```sql
-- 1. 기본 개수 확인
SELECT COUNT(*) FROM nedis_synthetic.yearly_volumes;
SELECT COUNT(*) FROM nedis_synthetic.daily_volumes;

-- 2. 총합 일치 확인  
SELECT SUM(synthetic_yearly_count) as yearly_total FROM nedis_synthetic.yearly_volumes;
SELECT SUM(synthetic_daily_count) as daily_total FROM nedis_synthetic.daily_volumes;

-- 3. 날짜별 분포 확인
SELECT 
    vst_dt,
    SUM(synthetic_daily_count) as daily_total
FROM nedis_synthetic.daily_volumes
GROUP BY vst_dt
ORDER BY daily_total DESC
LIMIT 10;

-- 4. 요일별 패턴 확인
SELECT 
    DAYOFWEEK(CAST(vst_dt AS DATE)) as weekday,
    AVG(synthetic_daily_count) as avg_count
FROM nedis_synthetic.daily_volumes
GROUP BY DAYOFWEEK(CAST(vst_dt AS DATE))
ORDER BY weekday;
```

#### 성공 기준
- [ ] `yearly_volumes` 테이블 레코드 수 > 1000
- [ ] `daily_volumes` 테이블 레코드 수 = yearly_volumes × 365
- [ ] 연간 총합과 일별 총합 차이 < 1%
- [ ] 모든 daily_count ≥ 0
- [ ] 365개 고유 날짜 생성 (2017-01-01 ~ 2017-12-31)
- [ ] 요일별 패턴이 원본 데이터와 유사한 트렌드
- [ ] 처리 시간 < 1시간 (10만 레코드 기준)

이 구현 가이드를 통해 Phase 1-2의 견고한 기반을 구축하고, 이후 Phase들을 위한 안정적인 데이터 파이프라인을 확보할 수 있습니다.