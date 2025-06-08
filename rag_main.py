"""
PDF 기반 RAG 시스템 구현
LangChain + Hugging Face Llama 모델 사용
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings("ignore")

# LangChain imports
from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 토크나이저 병렬 처리 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 사용 설정 (1번 GPU)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # CUDA 동기화 모드 (디버깅 용도)
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFRAGSystem:
    """PDF 기반 RAG 시스템"""

    def __init__(
        self,
        pdf_dir: str = "./data/pdfs/",
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",  # 또는 로컬 경로
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str = "./data/rag_vector_store/",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        use_quantization: bool = True,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.use_quantization = use_quantization

        # 컴포넌트 초기화
        self.documents = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None

        # 디렉토리 생성
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

    def load_pdfs(self) -> List[Any]:
        """PDF 파일들 로드"""
        logger.info(f"📚 Loading PDFs from {self.pdf_dir}")

        # DirectoryLoader로 모든 PDF 파일 로드
        loader = DirectoryLoader(
            str(self.pdf_dir),
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True,
        )

        documents = loader.load()
        logger.info(f"✅ Loaded {len(documents)} PDF pages")

        return documents

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """문서 청킹"""
        logger.info("🔪 Splitting documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(documents)
        logger.info(f"✅ Created {len(chunks)} chunks")

        return chunks

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """임베딩 모델 초기화"""
        logger.info(f"🧮 Initializing embedding model: {self.embedding_model}")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        return embeddings

    def create_vector_store(
        self, chunks: List[Any], embeddings: HuggingFaceEmbeddings
    ) -> FAISS:
        """벡터 스토어 생성 또는 로드"""
        vector_store_file = self.vector_store_path / "faiss_index"

        if vector_store_file.exists():
            logger.info("📂 Loading existing vector store")
            vectorstore = FAISS.load_local(
                str(vector_store_file), embeddings, allow_dangerous_deserialization=True
            )
        else:
            logger.info("🏗️ Creating new vector store")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(str(vector_store_file))
            logger.info(f"💾 Vector store saved to {vector_store_file}")

        return vectorstore

    def setup_llm(self) -> HuggingFacePipeline:
        """Llama 모델 설정"""
        logger.info(f"🤖 Setting up LLM: {self.model_name}")

        # 양자화 설정 (메모리 절약)
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
            self.model_name,
            trust_remote_code=True,
            padding_side="left",  # 생성을 위해 left padding
            clean_up_tokenization_spaces=True,
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
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # LangChain LLM 래퍼
        llm = HuggingFacePipeline(pipeline=pipe)

        return llm

    def create_qa_chain(
        self, llm: HuggingFacePipeline, vectorstore: FAISS
    ) -> RetrievalQA:
        """QA 체인 생성"""
        logger.info("🔗 Creating QA chain")

        prompt_template = """Please answer the following question in Korean based on the provided context. 
Provide a comprehensive and technical answer that is easy to understand.

Context: {context}

Question: {question}

Please provide your answer in Korean:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # 리트리버 생성
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )

        # QA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )

        return qa_chain

    def initialize(self):
        """RAG 시스템 전체 초기화"""
        logger.info("🚀 Initializing PDF RAG System")

        # 1. PDF 로드
        self.documents = self.load_pdfs()

        # 2. 문서 청킹
        chunks = self.split_documents(self.documents)

        # 3. 임베딩 생성
        embeddings = self.create_embeddings()

        # 4. 벡터 스토어 생성
        self.vectorstore = self.create_vector_store(chunks, embeddings)

        # 5. LLM 설정
        self.llm = self.setup_llm()

        # 6. QA 체인 생성
        self.qa_chain = self.create_qa_chain(self.llm, self.vectorstore)

        logger.info("✅ PDF RAG System initialized successfully!")

    def query(self, question: str) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        if self.qa_chain is None:
            raise ValueError("RAG system not initialized. Call initialize() first.")

        logger.info(f"❓ Processing question: {question}")

        # try:
        result = self.qa_chain({"query": question})

        response = {
            "question": question,
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                }
                for doc in result["source_documents"]
            ],
        }

        return response

    # except Exception as e:
    #     logger.error(f"❌ Error processing question: {e}")
    #     return {
    #         "question": question,
    #         "answer": f"오류가 발생했습니다: {str(e)}",
    #         "source_documents": [],
    #     }

    def add_memory(self):
        """대화 메모리 추가 (선택사항)"""
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        return memory


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PDF 기반 RAG 시스템 실행")
    parser.add_argument(
        "--pdf_dir", type=str, default="./data/pdfs/", help="PDF 파일 디렉토리"
    )
    parser.add_argument(
        "--queries", type=str, default="./queries.txt", help="query text 파일"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM 모델 이름",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="임베딩 모델 이름",
    )
    args = parser.parse_args()

    """메인 실행 함수"""
    # RAG 시스템 초기화
    rag_system = PDFRAGSystem(
        pdf_dir=args.pdf_dir,
        model_name=args.model_name,  # 실제 모델 경로로 변경
        embedding_model=args.embedding_model,
        use_quantization=True,  # GPU 메모리 부족시 True
    )

    # 시스템 초기화 (최초 실행시만)
    rag_system.initialize()

    # 테스트 질문들
    with open(args.queries, "r", encoding="utf-8") as f:
        test_questions = [line.strip() for line in f if line.strip()]

    # 질문 처리
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"질문: {question}")
        print(f"{'='*60}")

        result = rag_system.query(question)

        print(f"답변: {result['answer']}")
        print(f"\n참고 문서:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"{i}. {doc['source']} (페이지: {doc['page']})")
            print(f"   내용: {doc['content']}")

        result = rag_system.query(question)
        print(result)

        import pdb

        pdb.set_trace()
        print(f"\n답변: {result['answer']}")


if __name__ == "__main__":
    main()
