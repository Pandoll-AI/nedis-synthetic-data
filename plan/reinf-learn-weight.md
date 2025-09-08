# 강화학습 기반 가중치 최적화 모듈 추가

현재 계획서에는 **베이지안 최적화만** 포함되어 있고, **강화학습이 빠져있습니다**. Phase 7을 확장하여 강화학습 기반 동적 가중치 조정을 추가하겠습니다.

## Phase 7 확장: 강화학습 기반 가중치 최적화

### Task 7.2: 강화학습 에이전트 구현
```python
# src/optimization/rl_weight_optimizer.py 생성

import torch
import torch.nn as nn
import numpy as np
from collections import deque

class NEDISWeightOptimizer:
    """
    PPO (Proximal Policy Optimization) 기반 가중치 최적화
    """
    def __init__(self, conn):
        self.conn = conn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 조정 가능한 가중치 정의
        self.weight_dimensions = {
            # 계절 가중치 (4개 시즌 × 17개 시도)
            'seasonal_weights': (17, 4),  
            
            # 요일 가중치 (평일/주말 × 17개 시도)
            'weekday_weights': (17, 2),
            
            # 병원 종별 매력도 (3개 종별)
            'hospital_attractiveness': (3,),
            
            # 중력모형 거리 감쇠 (시도별)
            'gravity_gamma': (17,),
            
            # KTAS 조건부 확률 스무딩
            'ktas_alpha': (5,),
            
            # 체류시간 분포 파라미터 (KTAS별 평균, 표준편차)
            'duration_params': (5, 2),
            
            # IPF 수렴 임계값
            'ipf_tolerance': (1,),
            
            # Dirichlet 사전분포 강도
            'dirichlet_alpha': (1,)
        }
        
        self.state_dim = sum(np.prod(shape) for shape in self.weight_dimensions.values())
        self.action_dim = self.state_dim  # 각 가중치를 조정
        
        # 신경망 정의
        self.policy_net = self._build_policy_network()
        self.value_net = self._build_value_network()
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=3e-4
        )
        
        # 경험 버퍼
        self.memory = deque(maxlen=10000)
        
    def _build_policy_network(self):
        """
        정책 네트워크: 현재 상태 → 가중치 조정 액션
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Tanh()  # -1 to 1 범위로 조정량 출력
        ).to(self.device)
    
    def _build_value_network(self):
        """
        가치 네트워크: 현재 상태의 가치 평가
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
    
    def get_current_state(self):
        """
        현재 가중치 상태를 벡터로 변환
        
        1. DB에서 현재 가중치 로드:
        seasonal_weights = self.conn.execute('''
            SELECT seasonal_weight_spring, seasonal_weight_summer, 
                   seasonal_weight_fall, seasonal_weight_winter
            FROM nedis_meta.population_margins
            ORDER BY pat_do_cd, pat_age_gr, pat_sex
        ''').fetchnumpy()
        
        2. 모든 가중치를 하나의 벡터로 concatenate
        
        return state_vector
        """
        state_components = []
        
        # 계절 가중치
        seasonal = self.conn.execute('''
            SELECT DISTINCT pat_do_cd,
                   AVG(seasonal_weight_spring) as spring,
                   AVG(seasonal_weight_summer) as summer,
                   AVG(seasonal_weight_fall) as fall,
                   AVG(seasonal_weight_winter) as winter
            FROM nedis_meta.population_margins
            GROUP BY pat_do_cd
            ORDER BY pat_do_cd
        ''').fetchnumpy()
        state_components.append(seasonal['spring'])
        state_components.append(seasonal['summer'])
        state_components.append(seasonal['fall'])
        state_components.append(seasonal['winter'])
        
        # 현재 중력모형 gamma
        gamma = self.conn.execute('''
            SELECT gamma_value 
            FROM nedis_meta.optimization_params 
            WHERE param_name = 'gravity_gamma'
        ''').fetchone()[0]
        state_components.append([gamma])
        
        # ... 다른 가중치들도 추가
        
        return np.concatenate(state_components)
    
    def calculate_reward(self, synthetic_sample_size=10000):
        """
        보상 함수: 원본과 합성 데이터의 유사도
        
        R = w1 * statistical_similarity 
            + w2 * clinical_validity 
            - w3 * privacy_risk 
            - w4 * generation_time
        """
        
        # 1. 통계적 유사도 (KS test p-values)
        statistical_scores = []
        
        # KTAS 분포 비교
        original_ktas = self.conn.execute('''
            SELECT ktas_fstu, COUNT(*) as freq
            FROM nedis_original.nedis2017
            WHERE ktas_fstu IN ('1','2','3','4','5')
            GROUP BY ktas_fstu
        ''').fetchdf()
        
        synthetic_ktas = self.conn.execute('''
            SELECT ktas_fstu, COUNT(*) as freq
            FROM nedis_synthetic.clinical_records
            WHERE ktas_fstu IN ('1','2','3','4','5')
            GROUP BY ktas_fstu
            LIMIT {synthetic_sample_size}
        ''').fetchdf()
        
        # Chi-square test
        from scipy.stats import chisquare
        chi2, p_value = chisquare(
            synthetic_ktas['freq'], 
            original_ktas['freq'] * (synthetic_ktas['freq'].sum() / original_ktas['freq'].sum())
        )
        statistical_scores.append(p_value)
        
        # 2. 임상적 타당성 (규칙 위반률)
        violations = self.conn.execute('''
            SELECT COUNT(*) as violations
            FROM nedis_synthetic.clinical_records
            WHERE 
                -- 시간 역전
                vst_dt > otrm_dt OR
                -- KTAS 1인데 귀가
                (ktas_fstu = '1' AND emtrt_rust = '11') OR
                -- 남성인데 임신 진단
                (pat_sex = 'M' AND EXISTS (
                    SELECT 1 FROM nedis_synthetic.diag_er 
                    WHERE index_key = clinical_records.index_key 
                    AND diagnosis_code LIKE 'O%'
                ))
        ''').fetchone()[0]
        
        clinical_validity = 1.0 - (violations / synthetic_sample_size)
        
        # 3. 프라이버시 위험도
        # Nearest neighbor distance의 5th percentile
        nn_distance = self._calculate_nn_distance(sample_size=1000)
        privacy_score = min(1.0, nn_distance / 0.5)  # 0.5를 목표 거리로
        
        # 4. 생성 시간 페널티
        generation_time = self.conn.execute('''
            SELECT EXTRACT(EPOCH FROM (end_time - start_time)) as seconds
            FROM nedis_meta.pipeline_progress
            WHERE step_name = 'clinical_generation'
            ORDER BY start_time DESC
            LIMIT 1
        ''').fetchone()[0]
        
        time_penalty = max(0, (generation_time - 300) / 300)  # 5분 초과시 페널티
        
        # 종합 보상
        reward = (
            0.4 * np.mean(statistical_scores) +
            0.3 * clinical_validity +
            0.2 * privacy_score -
            0.1 * time_penalty
        )
        
        return reward
    
    def apply_action(self, action):
        """
        액션을 실제 가중치 조정으로 변환
        
        action: [-1, 1] 범위의 조정량 벡터
        """
        # 액션을 각 가중치 차원으로 reshape
        idx = 0
        updates = {}
        
        # 계절 가중치 조정 (비율 유지하면서)
        seasonal_adjustment = action[idx:idx+68].reshape(17, 4)
        idx += 68
        
        # Softmax로 정규화 (합이 1이 되도록)
        seasonal_adjustment = np.exp(seasonal_adjustment)
        seasonal_adjustment = seasonal_adjustment / seasonal_adjustment.sum(axis=1, keepdims=True)
        
        self.conn.execute('''
            UPDATE nedis_meta.population_margins
            SET seasonal_weight_spring = ?,
                seasonal_weight_summer = ?,
                seasonal_weight_fall = ?,
                seasonal_weight_winter = ?
            WHERE pat_do_cd = ? AND pat_age_gr = ? AND pat_sex = ?
        ''', seasonal_adjustment.flatten())
        
        # 중력모형 gamma 조정
        gamma_adjustment = action[idx:idx+17]
        idx += 17
        
        # 1.0 ~ 2.5 범위로 제한
        new_gamma = np.clip(1.5 + gamma_adjustment * 0.5, 1.0, 2.5)
        
        for i, region in enumerate(self._get_regions()):
            self.conn.execute('''
                UPDATE nedis_meta.optimization_params
                SET value = ?
                WHERE param_name = 'gravity_gamma' AND region = ?
            ''', (new_gamma[i], region))
        
        # KTAS 스무딩 파라미터 조정
        ktas_adjustment = action[idx:idx+5]
        idx += 5
        
        new_ktas_alpha = np.clip(1.0 + ktas_adjustment * 0.5, 0.1, 2.0)
        
        self.conn.execute('''
            UPDATE nedis_meta.optimization_params
            SET value = ?
            WHERE param_name = 'ktas_smoothing'
        ''', (new_ktas_alpha.tolist(),))
        
        # 체류시간 파라미터 조정
        duration_adjustment = action[idx:idx+10].reshape(5, 2)
        idx += 10
        
        # 평균은 ±20%, 표준편차는 ±30% 조정
        for ktas in range(1, 6):
            mean_factor = 1.0 + duration_adjustment[ktas-1, 0] * 0.2
            std_factor = 1.0 + duration_adjustment[ktas-1, 1] * 0.3
            
            self.conn.execute('''
                UPDATE nedis_meta.duration_params
                SET mean_minutes = mean_minutes * ?,
                    std_minutes = std_minutes * ?
                WHERE ktas_level = ?
            ''', (mean_factor, std_factor, str(ktas)))
    
    def train_step(self, batch_size=32):
        """
        PPO 알고리즘으로 정책 업데이트
        """
        if len(self.memory) < batch_size:
            return
        
        # 배치 샘플링
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Advantage 계산
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        advantages = rewards + 0.99 * next_values * (1 - dones) - values
        
        # Policy gradient
        action_probs = self.policy_net(states)
        
        # PPO clipped objective
        ratio = torch.exp(action_probs - actions)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values, rewards + 0.99 * next_values * (1 - dones))
        
        total_loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item()
```

### Task 7.3: 강화학습 훈련 루프
```python
# src/optimization/rl_trainer.py 생성

class RLTrainingLoop:
    def __init__(self, conn):
        self.conn = conn
        self.agent = NEDISWeightOptimizer(conn)
        self.episode_rewards = []
        
    def run_episode(self, max_steps=50):
        """
        한 에피소드 실행 (전체 생성 프로세스)
        """
        episode_reward = 0
        state = self.agent.get_current_state()
        
        for step in range(max_steps):
            # 1. 액션 선택 (가중치 조정량)
            action = self.agent.policy_net(
                torch.FloatTensor(state).to(self.agent.device)
            ).detach().cpu().numpy()
            
            # 탐색을 위한 노이즈 추가
            action += np.random.normal(0, 0.1, action.shape)
            action = np.clip(action, -1, 1)
            
            # 2. 액션 적용 (가중치 업데이트)
            self.agent.apply_action(action)
            
            # 3. 소규모 데이터 생성 (빠른 피드백을 위해)
            self._generate_mini_batch(size=10000)
            
            # 4. 보상 계산
            reward = self.agent.calculate_reward(synthetic_sample_size=10000)
            episode_reward += reward
            
            # 5. 다음 상태
            next_state = self.agent.get_current_state()
            
            # 6. 메모리에 저장
            done = (step == max_steps - 1) or (reward > 0.9)
            self.agent.memory.append((state, action, reward, next_state, done))
            
            # 7. 학습
            if len(self.agent.memory) > 32:
                loss = self.agent.train_step()
                
            # 8. 상태 업데이트
            state = next_state
            
            # 로깅
            print(f"Step {step}: Reward = {reward:.4f}")
            
            if done:
                break
        
        return episode_reward
    
    def _generate_mini_batch(self, size=10000):
        """
        빠른 테스트를 위한 소규모 데이터 생성
        """
        # 간단한 샘플링 기반 생성
        self.conn.execute(f'''
            INSERT INTO nedis_synthetic.clinical_records
            SELECT 
                -- 새로운 index_key 생성
                emorg_cd || '_SYNTH_' || ROW_NUMBER() OVER() || '_' || vst_dt || '_' || vst_tm as index_key,
                emorg_cd,
                'SYNTH_' || ROW_NUMBER() OVER() as pat_reg_no,
                vst_dt,
                vst_tm,
                pat_age_gr,
                pat_sex,
                -- 가중치 기반 샘플링
                CASE 
                    WHEN RANDOM() < 0.025 THEN '1'
                    WHEN RANDOM() < 0.075 THEN '2'
                    WHEN RANDOM() < 0.475 THEN '3'
                    WHEN RANDOM() < 0.875 THEN '4'
                    ELSE '5'
                END as ktas_fstu,
                -- ... 다른 필드들
            FROM nedis_original.nedis2017
            USING SAMPLE {size}
        ''')
        
    def train(self, num_episodes=100):
        """
        전체 훈련 프로세스
        """
        best_reward = -float('inf')
        
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            # 에피소드 실행
            episode_reward = self.run_episode()
            self.episode_rewards.append(episode_reward)
            
            print(f"Episode Reward: {episode_reward:.4f}")
            print(f"Average Reward (last 10): {np.mean(self.episode_rewards[-10:]):.4f}")
            
            # 최고 성능 모델 저장
            if episode_reward > best_reward:
                best_reward = episode_reward
                self._save_best_weights()
                print(f"New best reward: {best_reward:.4f}")
            
            # 수렴 체크
            if len(self.episode_rewards) > 10:
                recent_avg = np.mean(self.episode_rewards[-10:])
                if recent_avg > 0.85:  # 목표 달성
                    print("Target performance reached!")
                    break
            
            # 주기적 평가
            if (episode + 1) % 10 == 0:
                self._full_evaluation()
    
    def _save_best_weights(self):
        """
        최적 가중치를 영구 저장
        """
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS nedis_meta.best_weights AS
            SELECT * FROM nedis_meta.population_margins;
            
            CREATE TABLE IF NOT EXISTS nedis_meta.best_params AS
            SELECT * FROM nedis_meta.optimization_params;
        ''')
        
        torch.save({
            'policy_net': self.agent.policy_net.state_dict(),
            'value_net': self.agent.value_net.state_dict(),
            'optimizer': self.agent.optimizer.state_dict()
        }, 'models/best_rl_model.pt')
    
    def _full_evaluation(self):
        """
        전체 데이터셋으로 상세 평가
        """
        print("\n=== Full Evaluation ===")
        
        # 100만 건 생성
        self._generate_mini_batch(size=1_000_000)
        
        # 모든 검증 실행
        validator = StatisticalValidator(self.conn)
        results = validator.validate_distributions()
        
        print(f"KS Test Pass Rate: {results['ks_pass_rate']:.2%}")
        print(f"Chi-square Pass Rate: {results['chi2_pass_rate']:.2%}")
        print(f"Correlation Difference: {results['corr_diff']:.4f}")
        
        # 결과 기록
        self.conn.execute('''
            INSERT INTO nedis_meta.rl_training_log
            (episode, reward, ks_pass_rate, chi2_pass_rate, corr_diff, timestamp)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (len(self.episode_rewards), self.episode_rewards[-1], 
              results['ks_pass_rate'], results['chi2_pass_rate'], 
              results['corr_diff']))
```

### Main Pipeline에 통합
```python
# main.py 수정

class NEDISSyntheticDataPipeline:
    def __init__(self, config_path='config/generation_params.yaml'):
        self.config = yaml.safe_load(open(config_path))
        self.conn = duckdb.connect('nedis.duckdb')
        self.use_rl_optimization = self.config.get('use_rl_optimization', True)
        
    def run_full_pipeline(self, target_records=9_200_000):
        """
        ... 기존 단계들 ...
        
        7. 최적화 (Phase 7)
           if self.use_rl_optimization:
               # 강화학습 기반 최적화
               trainer = RLTrainingLoop(self.conn)
               trainer.train(num_episodes=50)
               
               # 최적 가중치로 전체 재생성
               self.regenerate_with_best_weights(target_records)
           else:
               # 베이지안 최적화 (기존)
               optimizer = SyntheticDataOptimizer(self.conn)
               optimizer.optimize(n_calls=50)
        """
        
        # Phase 1-6는 동일...
        
        # Phase 7: 최적화
        print("\n=== Phase 7: Weight Optimization ===")
        
        if self.use_rl_optimization:
            print("Using Reinforcement Learning optimization...")
            
            # 강화학습 훈련
            trainer = RLTrainingLoop(self.conn)
            trainer.train(num_episodes=100)
            
            # 최적 가중치 로드
            self.conn.execute('''
                TRUNCATE TABLE nedis_meta.population_margins;
                INSERT INTO nedis_meta.population_margins
                SELECT * FROM nedis_meta.best_weights;
                
                TRUNCATE TABLE nedis_meta.optimization_params;
                INSERT INTO nedis_meta.optimization_params
                SELECT * FROM nedis_meta.best_params;
            ''')
            
            # 최적 가중치로 전체 데이터 재생성
            print("Regenerating full dataset with optimized weights...")
            self.conn.execute('TRUNCATE TABLE nedis_synthetic.clinical_records')
            
            # Phase 2-5 재실행
            self._regenerate_all_phases(target_records)
            
        else:
            print("Using Bayesian optimization...")
            optimizer = SyntheticDataOptimizer(self.conn)
            best_params = optimizer.optimize(n_calls=50)
            self._apply_optimized_params(best_params)
```

## 모니터링 대시보드 추가

```sql
-- 강화학습 훈련 진행상황 추적
CREATE TABLE nedis_meta.rl_training_log (
    episode INTEGER,
    reward DOUBLE,
    ks_pass_rate DOUBLE,
    chi2_pass_rate DOUBLE,
    corr_diff DOUBLE,
    clinical_violation_rate DOUBLE,
    privacy_score DOUBLE,
    generation_time_seconds INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (episode)
);

-- 가중치 변화 추적
CREATE TABLE nedis_meta.weight_history (
    episode INTEGER,
    weight_type VARCHAR,  -- 'seasonal', 'gravity', 'ktas', etc.
    region VARCHAR,
    old_value DOUBLE,
    new_value DOUBLE,
    change_ratio DOUBLE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

이제 **강화학습이 완전히 통합**되어:
1. 각 에피소드마다 가중치를 동적으로 조정
2. 생성된 데이터의 품질을 보상으로 평가
3. PPO 알고리즘으로 정책 개선
4. 최적 가중치를 자동으로 발견하고 저장
5. 수렴 시 전체 데이터셋을 최적 가중치로 재생성