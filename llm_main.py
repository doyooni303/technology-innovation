#!/usr/bin/env python3
"""
LLM 배치 처리 스크립트 - 단일 파일 완전판
텍스트 파일에서 쿼리를 읽어와서 LLM에 직접 입력하고 결과를 CSV로 저장
"""

import os
import csv
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings("ignore")

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 토크나이저 병렬 처리 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 사용 설정 (1번 GPU)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # CUDA 동기화 모드 (디버깅 용도)
# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMBatchProcessor:
    """LLM 배치 처리 클래스"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        use_quantization: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.pipeline = None

    def initialize_model(self) -> bool:
        """LLM 모델 초기화"""
        try:
            logger.info(f"🤖 Initializing LLM: {self.model_name}")

            # 양자화 설정
            quantization_config = None
            if self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            # 파이프라인 생성
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                return_full_text=False,
            )

            logger.info("✅ LLM model initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            return False

    def create_prompt(self, query: str) -> str:
        """프롬프트 생성"""
        prompt = """Please answer the following question in Korean based on the provided context. 
Provide a comprehensive and technical answer that is easy to understand.

Question: {query}

Please provide your answer in Korean:"""

    def generate_response(self, query: str) -> Dict[str, Any]:
        """단일 쿼리에 대한 응답 생성"""
        if self.pipeline is None:
            return {"success": False, "error": "Model not initialized"}

        start_time = time.time()

        try:
            prompt = self.create_prompt(query)
            result = self.pipeline(prompt)

            if result and len(result) > 0:
                response = result[0]["generated_text"].strip()
                processing_time = time.time() - start_time

                return {
                    "success": True,
                    "response": response,
                    "processing_time": processing_time,
                    "prompt": prompt,
                    "error": "",
                }
            else:
                return {"success": False, "error": "No response generated"}

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "response": "",
                "prompt": "",
            }

    def load_queries(self, query_file: str) -> List[str]:
        """텍스트 파일에서 쿼리 로드"""
        query_path = Path(query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_file}")

        queries = []
        with open(query_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                query = line.strip()
                if query and not query.startswith("#"):
                    queries.append(query)

        logger.info(f"✅ Loaded {len(queries)} queries from {query_file}")
        return queries

    def save_csv_results(self, results: List[Dict], output_path: Path):
        """결과를 CSV로 저장 (한글 지원 - 다중 인코딩)"""
        fieldnames = [
            "query_index",
            "query",
            "response",
            "prompt",
            "processing_time",
            "success",
            "error_msg",
            "timestamp",
        ]

        # 1. UTF-8 BOM으로 저장 (Excel 호환)
        with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        # 2. UTF-8 버전도 저장 (프로그래밍 용도)
        utf8_path = output_path.with_name(output_path.stem + "_utf8.csv")
        with open(utf8_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        # 3. CP949 저장 (한국어 Windows 호환)
        try:
            cp949_path = output_path.with_name(output_path.stem + "_cp949.csv")
            with open(cp949_path, "w", newline="", encoding="cp949") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    # CP949로 인코딩할 수 없는 문자 처리
                    safe_row = {}
                    for key, value in result.items():
                        if isinstance(value, str):
                            try:
                                value.encode("cp949")
                                safe_row[key] = value
                            except UnicodeEncodeError:
                                # 인코딩 불가능한 문자는 대체
                                safe_row[key] = value.encode(
                                    "cp949", errors="replace"
                                ).decode("cp949")
                        else:
                            safe_row[key] = value
                    writer.writerow(safe_row)

            logger.info(
                f"✅ Results saved to {output_path}, {utf8_path}, and {cp949_path}"
            )

        except Exception as e:
            logger.warning(f"⚠️ CP949 encoding failed: {e}")
            logger.info(f"✅ Results saved to {output_path} and {utf8_path}")

    def process_batch(self, query_file: str, output_file: str):
        """배치 처리 실행"""
        if self.pipeline is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        # 쿼리 로드
        queries = self.load_queries(query_file)

        # 결과 저장 경로
        results_dir = Path("./results/LLM")
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / output_file

        logger.info(f"🚀 Starting batch processing of {len(queries)} queries...")

        all_results = []
        success_count = 0
        start_time = time.time()

        for i, query in enumerate(queries):
            logger.info(f"🔄 Processing query {i+1}/{len(queries)}: {query[:50]}...")

            # LLM으로 응답 생성
            result = self.generate_response(query)
            timestamp = datetime.now().isoformat()

            # CSV 행 데이터 준비
            csv_row = {
                "query_index": i + 1,
                "query": query,
                "response": result.get("response", ""),
                "prompt": result.get("prompt", "")[:500] + "...",  # 길이 제한
                "processing_time": round(result.get("processing_time", 0), 2),
                "success": result["success"],
                "error_msg": result.get("error", ""),
                "timestamp": timestamp,
            }

            all_results.append(csv_row)

            if result["success"]:
                success_count += 1
                logger.info(
                    f"✅ Query {i+1} completed in {csv_row['processing_time']}s"
                )
            else:
                logger.error(f"❌ Query {i+1} failed: {result['error']}")

            # 진행률 표시
            progress = (i + 1) / len(queries) * 100
            logger.info(f"📊 Progress: {i+1}/{len(queries)} ({progress:.1f}%)")

            # 시스템 부하 방지를 위한 대기
            if i < len(queries) - 1:
                time.sleep(1.0)

        # 결과 저장
        self.save_csv_results(all_results, output_path)

        # 통계 출력
        total_time = time.time() - start_time
        success_rate = success_count / len(queries) * 100 if queries else 0

        logger.info(f"🎉 Batch processing completed!")
        logger.info(f"   📊 Total time: {total_time:.2f}s")
        logger.info(
            f"   ✅ Success rate: {success_count}/{len(queries)} ({success_rate:.1f}%)"
        )
        logger.info(f"   📄 Results saved to: {output_path}")


def check_prerequisites() -> bool:
    """사전 요구사항 확인"""
    logger.info("🔍 Checking prerequisites...")

    try:
        # GPU 확인
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"   GPU: ✅ {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.info("   GPU: ❌ Using CPU (will be slower)")

        # 라이브러리 확인
        import transformers

        logger.info(f"   Transformers: ✅ v{transformers.__version__}")

        return True

    except Exception as e:
        logger.error(f"❌ Prerequisites check failed: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LLM Batch Processing")
    parser.add_argument(
        "query_file",
        default="./queries.txt",
        help="Input text file with queries (one per line)",
    )
    parser.add_argument(
        "-o", "--output", default="llm_batch_results.csv", help="Output CSV file name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM 모델 이름",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for generation"
    )
    parser.add_argument(
        "--no-quantization", action="store_true", help="Disable model quantization"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("🚀 LLM Batch Processing Started")
    logger.info(f"   Input file: {args.query_file}")
    logger.info(f"   Output file: ./results/LLM/{args.output}")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Max tokens: {args.max_tokens}")
    logger.info(f"   Temperature: {args.temperature}")
    logger.info(f"   Quantization: {'disabled' if args.no_quantization else 'enabled'}")

    try:
        # 사전 요구사항 확인
        if not check_prerequisites():
            logger.error("❌ Prerequisites check failed. Exiting.")
            return 1

        # 배치 프로세서 초기화
        processor = LLMBatchProcessor(
            model_name=args.model,
            use_quantization=not args.no_quantization,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # 모델 초기화
        if not processor.initialize_model():
            logger.error("❌ Model initialization failed. Exiting.")
            return 1

        # 배치 처리 실행
        processor.process_batch(args.query_file, args.output)

        logger.info("🎉 Batch processing completed successfully!")
        logger.info("💡 File usage guide:")
        logger.info("   - Main file (.csv): UTF-8 BOM, opens directly in Excel")
        logger.info("   - _utf8.csv: Pure UTF-8 for programming")

        return 0

    except KeyboardInterrupt:
        logger.warning("⚠️ Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


# ============================================================================
# 사용 예시
# ============================================================================
"""
📝 쿼리 파일 형식 (queries.txt):
배터리 SOC 예측에 사용된 머신러닝 기법은 무엇인가요?
논문에서 제안하는 새로운 방법론의 핵심은 무엇인가요?
실험 결과에서 가장 중요한 발견은 무엇인가요?
# 이것은 주석입니다 (무시됨)
어떤 데이터셋을 사용했나요?

🚀 실행 방법:
1. 기본 실행:
   python simple_llm_batch.py queries.txt

2. 출력 파일명 지정:
   python simple_llm_batch.py queries.txt -o my_results.csv

3. 다른 모델 사용:
   python simple_llm_batch.py queries.txt --model "microsoft/DialoGPT-large"

4. 생성 파라미터 조정:
   python simple_llm_batch.py queries.txt --max-tokens 1024 --temperature 0.3

5. 상세 로그:
   python simple_llm_batch.py queries.txt -v

📁 결과 파일:
- ./results/LLM/llm_batch_results.csv (UTF-8 BOM - Excel용)
- ./results/LLM/llm_batch_results_utf8.csv (UTF-8 - 프로그래밍용)

🔧 주요 특징:
✅ 단일 파일로 완전 실행 가능
✅ 텍스트 파일에서 쿼리 로드 (한 줄씩)
✅ 한글 CSV 저장 (Excel 호환)
✅ 4-bit 양자화로 메모리 절약
✅ 실시간 진행률 표시
✅ 에러 처리 및 복구
✅ GPU/CPU 자동 감지
✅ 다양한 모델 지원
"""
