"""
LLM Metadata Logger

LLM API 호출의 세부 메타데이터를 수집하고 문서화하는 도구
"""

import os
import json
from datetime import datetime
from pathlib import Path
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMMetadataLogger:
    """
    LLM API 호출 메타데이터 로거

    LLM API 호출에 대한 다음 정보를 수집:
    - 모델명/버전
    - API 파라미터 (temperature, max_tokens, top_p)
    - 호출 타임스탬프
    - 평가 설계 파라미터 (datasets, metrics, aggregation method)
    - Deterministic sampling 여부 (temperature=0)
    """

    def __init__(self, output_dir=None):
        """
        메타데이터 로거 초기화

        Args:
            output_dir: 메타데이터 저장 디렉토리 (기본값: ../data/llm_metadata)
        """
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now().isoformat()
        
        # 출력 디렉토리 설정
        if output_dir is None:
            base_dir = Path(__file__).parent.parent
            output_dir = base_dir / 'data' / 'llm_metadata'
        else:
            output_dir = Path(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.output_path = output_dir / f"metadata_{self.session_id}.json"
        
        # 데이터 저장소 초기화
        self.api_calls = []
        self.evaluation_parameters = {}
        
        logger.info(f"LLM Metadata Logger initialized. Session ID: {self.session_id}")
        logger.info(f"Metadata will be saved to: {self.output_path}")

    def set_evaluation_parameters(self, datasets=None, num_topics_per_dataset=None, 
                                 metrics_evaluated=None, aggregation_method="mean", 
                                 num_iterations=1):
        """
        평가 설계 파라미터 설정

        Args:
            datasets: 평가 대상 데이터셋 리스트
            num_topics_per_dataset: 데이터셋별 토픽 수
            metrics_evaluated: 평가된 메트릭 리스트
            aggregation_method: 집계 방법 (예: "mean", "median")
            num_iterations: 항목당 평가 반복 횟수
        """
        self.evaluation_parameters = {
            "datasets": datasets or [],
            "num_topics_per_dataset": num_topics_per_dataset or {},
            "metrics_evaluated": metrics_evaluated or [],
            "aggregation_method": aggregation_method,
            "num_iterations": num_iterations
        }
        logger.info(f"Evaluation parameters set: {self.evaluation_parameters}")

    def log_api_call(self, evaluator_name, model_name, metric, prompt, response,
                    api_parameters, dataset_name=None, topic_indices=None):
        """
        단일 LLM API 호출 로깅

        Args:
            evaluator_name: 평가 클래스명 (예: "AnthropicEval")
            model_name: 사용된 LLM 모델명 (예: "claude-sonnet-4-5-20250929")
            metric: 평가 메트릭 (예: "coherence", "distinctiveness")
            prompt: API에 전송된 프롬프트 텍스트
            response: API 응답 텍스트
            api_parameters: API 파라미터 딕셔너리 {"temperature": 0, "max_tokens": 150}
            dataset_name: 평가 대상 데이터셋 이름 (예: "distinct")
            topic_indices: 평가된 토픽 인덱스 리스트 (배치 호출의 경우)
        """
        timestamp = datetime.now().isoformat()
        
        # temperature=0 체크하여 deterministic 여부 판단
        is_deterministic = api_parameters.get('temperature', 1.0) == 0
        
        call_data = {
            "timestamp": timestamp,
            "evaluator": evaluator_name,
            "model": model_name,
            "metric": metric,
            "dataset": dataset_name,
            "topic_indices": topic_indices,
            "api_parameters": api_parameters,
            "is_deterministic": is_deterministic,
            # 보안상 민감한 데이터는 필요에 따라 저장 여부 결정
            "prompt_length": len(prompt) if prompt else 0,
            "response_length": len(response) if response else 0,
            # 필요시 프롬프트와 응답 전체 저장도 가능
            # "prompt": prompt,
            # "response": response,
        }
        
        self.api_calls.append(call_data)
        logger.debug(f"Logged API call: {evaluator_name}, {model_name}, {metric}, {dataset_name}")

    def save_current_state(self):
        """현재 수집된 메타데이터를 JSON 파일로 저장"""
        metadata = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "last_updated": datetime.now().isoformat(),
            "total_api_calls": len(self.api_calls),
            "evaluation_parameters": self.evaluation_parameters,
            "api_calls": self.api_calls
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved to {self.output_path}")
        return str(self.output_path)

    def finalize_session(self):
        """세션 종료 및 최종 메타데이터 저장"""
        metadata_path = self.save_current_state()
        logger.info(f"LLM evaluation session finalized. {len(self.api_calls)} API calls logged.")
        return metadata_path


def create_metadata_report(metadata_path):
    """
    메타데이터 JSON 파일로부터 마크다운 리포트 생성

    Args:
        metadata_path: 메타데이터 JSON 파일 경로

    Returns:
        str: 마크다운 형식의 리포트
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return f"Error loading metadata file: {e}"
    
    # 모델 및 파라미터 통계 추출
    models = {}
    deterministic_count = 0
    temperature_values = set()
    max_tokens_values = set()
    
    for call in metadata['api_calls']:
        model = call['model']
        if model not in models:
            models[model] = 0
        models[model] += 1
        
        if call.get('is_deterministic', False):
            deterministic_count += 1
            
        params = call.get('api_parameters', {})
        temperature_values.add(params.get('temperature'))
        max_tokens_values.add(params.get('max_tokens'))
    
    # 마크다운 리포트 생성
    report = [
        "# LLM Evaluation Metadata Report\n",
        f"**Session ID**: {metadata['session_id']}",
        f"**Date**: {metadata['start_time']}",
        f"**Total API Calls**: {metadata['total_api_calls']}",
        
        "\n## API Configuration\n",
    ]
    
    # 모델 정보
    for model, count in models.items():
        report.append(f"- **Model**: {model} ({count} calls)")
    
    # API 파라미터
    report.append(f"- **Temperature**: {', '.join(str(t) for t in temperature_values if t is not None)}")
    report.append(f"- **Max Tokens**: {', '.join(str(t) for t in max_tokens_values if t is not None)}")
    report.append(f"- **Deterministic**: {'Yes' if deterministic_count == metadata['total_api_calls'] else 'No'} ({deterministic_count}/{metadata['total_api_calls']} calls)")
    
    # 평가 설계
    report.extend([
        "\n## Evaluation Design\n",
        f"- **Datasets**: {', '.join(metadata['evaluation_parameters'].get('datasets', []))}",
        f"- **Metrics**: {', '.join(metadata['evaluation_parameters'].get('metrics_evaluated', []))}",
        f"- **Aggregation Method**: {metadata['evaluation_parameters'].get('aggregation_method', 'unknown')}",
        f"- **Iterations per Item**: {metadata['evaluation_parameters'].get('num_iterations', 1)}"
    ])
    
    # 호출 시간 분포
    report.extend([
        "\n## API Call Timeline\n",
        f"- **First Call**: {metadata['api_calls'][0]['timestamp'] if metadata['api_calls'] else 'N/A'}",
        f"- **Last Call**: {metadata['api_calls'][-1]['timestamp'] if metadata['api_calls'] else 'N/A'}"
    ])
    
    return "\n".join(report)