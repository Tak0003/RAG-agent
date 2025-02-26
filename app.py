import streamlit as st
import os
import json
from typing import TypedDict, List
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import AzureCognitiveSearchRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from pydantic import Field
from langchain_core.documents import Document
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, END, START
from langsmith import Client
from azure.storage.blob import BlobServiceClient
import config

# Set up environment variables
# LangChain environment variables will be set from config.py which loads from .env
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Azure Blob Storage Setup
blob_service_client = BlobServiceClient.from_connection_string(config.AZURE_BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(config.AZURE_BLOB_CONTAINER)

# Supported file types
SUPPORTED_TYPES = [
    "pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls",
    "txt", "csv", "json", "xml", "html", "htm"
]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


class GraphState(TypedDict):
    question: str
    generation: str
    search_results: str  # Renamed from web_search to search_results
    documents: List[Document]
    original_question: str  # Added to store the original question when transforming queries


llm = AzureChatOpenAI(
    openai_api_version=config.AZURE_OPENAI_API_VERSION,
    azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_key=config.AZURE_OPENAI_API_KEY,
    tags=["corrective-rag"]
)

retriever = AzureCognitiveSearchRetriever(
    service_name="yamanaka-agent-test",
    api_key=config.AZURE_SEARCH_KEY,
    index_name=config.AZURE_SEARCH_INDEX,
    content_key="content",
    top_k=3,
)


def upload_file_to_blob(uploaded_file, file_type):
    """Upload a file to Azure Blob Storage."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"{timestamp}_{uploaded_file.name}"

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(uploaded_file.getvalue())

        return {
            "status": "success",
            "message": f"ファイル {uploaded_file.name} がアップロードされました。",
            "blob_name": blob_name
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"アップロードエラー: {str(e)}",
            "blob_name": None
        }


def is_valid_file(file):
    """Check if the file type is supported."""
    if file is None:
        return False
    file_type = file.name.split('.')[-1].lower()
    return file_type in SUPPORTED_TYPES


def retrieve(state: GraphState) -> dict:
    with st.spinner('ドキュメント検索中...'):
        question = state["question"]

        # Get documents from the retriever
        documents = retriever.get_relevant_documents(question)

        # Extract filenames and add useful metadata
        filenames = []
        for doc in documents:
            if 'source' in doc.metadata:
                filename = doc.metadata['source']
                filenames.append(filename)
                # Set source_type to 'blob' for all retrieved documents from Azure Cognitive Search
                # This is critical for proper categorization in the generate function
                doc.metadata['source_type'] = 'blob'

                # Check if filename contains any terms from the query
                # This will help with evaluation later
                query_terms = question.lower().split()
                if any(term.lower() in filename.lower() for term in query_terms):
                    doc.metadata['filename_match'] = True
                else:
                    doc.metadata['filename_match'] = False

            else:
                # Ensure all documents have source_type even if they don't have source
                doc.metadata['source_type'] = 'blob'
                doc.metadata['source'] = 'Unknown'
                doc.metadata['filename_match'] = False

            # Add search score as part of metadata if available
            if '@search.score' in doc.metadata:
                doc.metadata['search_score'] = doc.metadata['@search.score']

        if debug_mode:
            st.write(f"検索されたドキュメント数: {len(documents)}")
            if documents:
                st.write("### 検索されたドキュメント:")
                for i, doc in enumerate(documents, 1):
                    st.write(f"\n**Document {i}:**")
                    source_name = doc.metadata.get('source', 'Unknown')
                    st.write(f"Source: {source_name}")
                    st.write(
                        f"Source Type: {doc.metadata.get('source_type', 'Not specified')}")  # Add this line for debugging

                    if doc.metadata.get('filename_match', False):
                        st.write("⭐ **ファイル名が検索クエリと一致**")

                    if '@search.score' in doc.metadata:
                        st.write(f"Score: {doc.metadata['@search.score']}")

                    st.write(f"Content (前200文字): {doc.page_content[:200]}...")
                    st.write(f"Metadata: {doc.metadata}")
            else:
                st.write("⚠️ ドキュメントが見つかりませんでした")

            st.write("### 検索されたファイル名:")
            st.write(filenames)

        return {"documents": documents, "question": question}


def grade_documents(state: GraphState) -> dict:
    with st.spinner('関連性を評価中...'):
        question = state["question"]
        documents = state["documents"]

        # First, prioritize documents with filename matches
        filename_matches = []
        other_docs = []

        for doc in documents:
            # Ensure all documents have source_type set
            if 'source_type' not in doc.metadata:
                doc.metadata['source_type'] = 'blob'  # Default to blob for retrieved documents

            if doc.metadata.get('filename_match', False):
                filename_matches.append(doc)
            else:
                other_docs.append(doc)

        # More sophisticated grading prompt - with escaped curly braces for the JSON example
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは、検索された文書とユーザーの質問との関連性を厳密に評価する採点者です。

評価基準:
1. 文書が質問に対して明確に関連する情報を含んでいる場合は 'yes' としてください
2. 文書が質問のキーワードを含んでいても、実際に質問に答えていない場合は 'no' としてください
3. 部分的な関連性がある場合でも、質問の主要な部分に対応していれば 'yes' としてください
4. 文書のファイル名が質問のキーワードと一致する場合は特に注意深く評価してください

以下のフォーマットで回答してください:
{{
    "binary_score": "yes/no",
    "reason": "評価理由の詳細な説明",
    "confidence": "high/medium/low"
}}"""),
            ("human", """
文書: {document}

質問: {question}

この文書は質問に関連していますか？""")
        ])

        # Create lists to store results
        filtered_docs = []
        low_confidence_docs = []  # Docs where relevance is uncertain
        need_search = "No"  # Changed variable name from web_search to need_search

        if debug_mode:
            st.write(f"\n### ドキュメント評価:")
            st.write(f"評価対象: {len(documents)}件")
            st.write(f"ファイル名一致: {len(filename_matches)}件")

        # First process filename matches - they get a higher chance of being included
        for i, doc in enumerate(filename_matches, 1):
            if debug_mode:
                st.write(f"\n**ファイル名一致ドキュメント {i} の評価:**")
                st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
                st.write(f"Source Type: {doc.metadata.get('source_type', 'Not specified')}")
                st.write(f"Content: {doc.page_content[:200]}...")

            response = grade_prompt | llm
            result = response.invoke({
                "question": question,
                "document": doc.page_content
            })

            if debug_mode:
                st.write(f"評価結果: {result.content}")

            try:
                evaluation = json.loads(result.content.replace("'", '"'))
                is_relevant = evaluation.get("binary_score", "").lower() == "yes"
                confidence = evaluation.get("confidence", "medium").lower()

                if debug_mode:
                    st.write(f"関連性: {'あり' if is_relevant else 'なし'}")
                    st.write(f"理由: {evaluation.get('reason', 'Not provided')}")
                    st.write(f"信頼度: {confidence}")

                # For filename matches, we're more lenient
                if is_relevant or confidence != "low":
                    # Make sure source_type is preserved
                    filtered_docs.append(doc)
                    if debug_mode:
                        st.write("✅ ドキュメントを採用 (ファイル名一致)")
                else:
                    if debug_mode:
                        st.write("❌ ドキュメントを不採用 (ファイル名一致だが内容が無関係)")
            except:
                # Fallback if JSON parsing fails
                content_lower = result.content.lower()
                is_relevant = "yes" in content_lower

                # For filename matches, default to including them
                filtered_docs.append(doc)
                if debug_mode:
                    st.write(f"関連性: {'あり' if is_relevant else 'なし'} (フォーマット解析失敗)")
                    st.write("✅ ドキュメントを採用 (ファイル名一致)")

        # Then process the other documents with normal criteria
        for i, doc in enumerate(other_docs, 1):
            if debug_mode:
                st.write(f"\n**通常ドキュメント {i} の評価:**")
                st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
                st.write(f"Source Type: {doc.metadata.get('source_type', 'Not specified')}")
                st.write(f"Content: {doc.page_content[:200]}...")

            response = grade_prompt | llm
            result = response.invoke({
                "question": question,
                "document": doc.page_content
            })

            if debug_mode:
                st.write(f"評価結果: {result.content}")

            try:
                evaluation = json.loads(result.content.replace("'", '"'))
                is_relevant = evaluation.get("binary_score", "").lower() == "yes"
                confidence = evaluation.get("confidence", "medium").lower()

                if debug_mode:
                    st.write(f"関連性: {'あり' if is_relevant else 'なし'}")
                    st.write(f"理由: {evaluation.get('reason', 'Not provided')}")
                    st.write(f"信頼度: {confidence}")

                if is_relevant:
                    filtered_docs.append(doc)
                    if debug_mode:
                        st.write("✅ ドキュメントを採用")
                else:
                    if confidence == "low":
                        # Store docs where the model is uncertain
                        low_confidence_docs.append(doc)
                        if debug_mode:
                            st.write("⚠️ 低信頼度: Web検索に備えて保存")
                    else:
                        if debug_mode:
                            st.write("❌ ドキュメントを不採用")
            except:
                # Fallback if JSON parsing fails
                is_relevant = "yes" in result.content.lower()
                if is_relevant:
                    filtered_docs.append(doc)
                    if debug_mode:
                        st.write(f"関連性: あり (フォーマット解析失敗)")
                        st.write("✅ ドキュメントを採用")
                else:
                    if debug_mode:
                        st.write(f"関連性: なし (フォーマット解析失敗)")
                        st.write("❌ ドキュメントを不採用")

        # If no clearly relevant docs but some low confidence ones, include them
        if not filtered_docs and low_confidence_docs:
            if debug_mode:
                st.write("明確な関連ドキュメントがないため、低信頼度のドキュメントを採用します")
            filtered_docs = low_confidence_docs

        # If we still have no relevant documents, set need_search to "Yes"
        if not filtered_docs:
            need_search = "Yes"
            if debug_mode:
                st.write("関連ドキュメントがないため、Web検索を実行します")
        else:
            if debug_mode:
                st.write(f"{len(filtered_docs)}件の関連ドキュメントが見つかりました")
                # Debug document types
                blob_docs = [doc for doc in filtered_docs if doc.metadata.get('source_type', '') == 'blob']
                web_docs = [doc for doc in filtered_docs if doc.metadata.get('source_type', '') == 'web']
                st.write(f"内訳: ブロブ文書 {len(blob_docs)}件, Web文書 {len(web_docs)}件")

        return {
            "documents": filtered_docs,
            "question": question,
            "search_results": need_search  # Updated to use search_results instead of web_search
        }


def transform_query(state: GraphState) -> dict:
    """
    Transform the query to produce a better search question.
    This can help when the original query doesn't yield good search results.
    """
    with st.spinner('検索クエリを最適化中...'):
        question = state["question"]
        documents = state["documents"]

        # CRAG-inspired query transformation
        transform_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは検索に最適化された質問を生成するエキスパートです。
            与えられた質問を分析して、検索エンジンで最も良い結果を得るための質問に書き換えてください。

            以下の方法で質問を改善してください:
            1. キーワードを明確にする
            2. 曖昧な表現を具体的にする
            3. 検索エンジンが理解しやすい形式にする
            4. 必要に応じて、検索に役立つ追加コンテキストを含める

            元の質問の意味を変えないように注意してください。"""),
            ("human", """
            元の質問:
            {question}

            検索用に最適化された質問を生成してください:""")
        ])

        # Generate optimized query
        response = transform_prompt | llm | StrOutputParser()
        optimized_query = response.invoke({"question": question})

        if debug_mode:
            st.write("### クエリ最適化")
            st.write(f"元のクエリ: {question}")
            st.write(f"最適化されたクエリ: {optimized_query}")

        return {
            "documents": documents,
            "question": optimized_query,
            "original_question": question  # Store the original question
        }


def web_search_fn(state: GraphState) -> dict:  # Renamed function from web_search to web_search_fn
    with st.spinner('Web検索実行中...'):
        question = state["question"]
        documents = state["documents"]
        original_question = state.get("original_question", question)  # Get original if available

        if debug_mode:
            st.write("### Web検索実行")
            st.write(f"検索クエリ: {question}")

        try:
            search = DuckDuckGoSearchAPIWrapper(
                region='jp-jp',
                time='m',
                safesearch='moderate',
                backend='html'
            )
            results = search.results(
                query=question,
                max_results=5,
                source="text"
            )

            if debug_mode:
                st.write(f"検索結果取得: {len(results)}件")

            if not results:
                web_doc = Document(
                    page_content="検索結果が見つかりませんでした。",
                    metadata={
                        "source": "web_search",
                        "source_type": "web",
                        "query": question
                    }
                )
                documents.append(web_doc)
            else:
                # Process and add each result as a separate document for better evaluation
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'No Title')
                    snippet = result.get('snippet', 'No Content')
                    link = result.get('link', 'No Link')

                    content = f"タイトル: {title}\n内容: {snippet}\nリンク: {link}"

                    web_doc = Document(
                        page_content=content,
                        metadata={
                            "source": f"web_result_{i}",
                            "source_type": "web",
                            "title": title,
                            "link": link,
                            "query": question
                        }
                    )
                    documents.append(web_doc)

                    if debug_mode and i <= 2:  # Show first two results in debug mode
                        st.write(f"Web結果 {i}:")
                        st.write(f"タイトル: {title}")
                        st.write(f"内容: {snippet[:100]}...")
                        st.write(f"リンク: {link}")

        except Exception as e:
            if debug_mode:
                st.error(f"Web検索エラー: {str(e)}")
                import traceback
                st.write("Traceback:", traceback.format_exc())
            web_doc = Document(
                page_content=f"Web検索で情報を取得できませんでした。エラー: {str(e)}",
                metadata={
                    "source": "web_search_error",
                    "source_type": "error",
                    "error": str(e)
                }
            )
            documents.append(web_doc)

        # Return to original question for generation
        return {
            "documents": documents,
            "question": original_question  # Use original question for answer generation
        }


def generate(state: GraphState) -> dict:
    with st.spinner('回答を生成中...'):
        question = state["question"]
        documents = state["documents"]

        # Debug documents before categorization
        if debug_mode:
            st.write("### 回答生成前のドキュメント状態")
            st.write(f"ドキュメント総数: {len(documents)}")
            for i, doc in enumerate(documents):
                st.write(f"Doc {i + 1} source_type: {doc.metadata.get('source_type', 'Not set')}")
                st.write(f"Doc {i + 1} metadata: {doc.metadata}")

        # Check if we have blob documents or only web results
        # Make sure to handle documents that might not have source_type set
        blob_docs = [doc for doc in documents if doc.metadata.get('source_type', '') == 'blob']
        web_docs = [doc for doc in documents if doc.metadata.get('source_type', '') == 'web']

        # If we have documents but no source_type is set, categorize them as blob by default
        if len(blob_docs) == 0 and len(web_docs) == 0 and len(documents) > 0:
            blob_docs = documents
            # Update the metadata for these docs
            for doc in blob_docs:
                doc.metadata['source_type'] = 'blob'

        has_blob_docs = len(blob_docs) > 0
        has_web_docs = len(web_docs) > 0

        if debug_mode:
            st.write("### 回答生成")
            st.write(f"Blobドキュメント数: {len(blob_docs)}")
            st.write(f"Webドキュメント数: {len(web_docs)}")
            st.write(f"全ドキュメント数: {len(documents)}")

        # Different prompts depending on document sources
        if has_blob_docs:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたは提供された文書に基づいて質問に回答する専門家です。
                以下のガイドラインに従ってください：

                1. 提供された文書の内容だけを使用して回答を作成してください
                2. 文書にない情報は含めないでください
                3. 文書に記載された情報が不十分な場合は、その旨を正直に伝えてください
                4. 回答は簡潔かつ明確に、日本語で記載してください
                5. 引用やソース情報は含めないでください

                提供される文書の内容を信頼し、それに基づいて回答を構築してください。"""),
                ("human", """
                文書:
                {context}

                質問:
                {question}

                質問に対する回答を提供してください。""")
            ])
        else:
            # When using only web search results, be more cautious
            prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたは提供された検索結果に基づいて質問に回答する専門家です。
                以下のガイドラインに従ってください：

                1. 提供された検索結果の内容だけを使用して回答を作成してください
                2. 検索結果にない情報は含めないでください
                3. 検索結果に記載された情報が不十分な場合は、その旨を正直に伝えてください
                4. 回答は簡潔かつ明確に、日本語で記載してください
                5. 引用やソース情報は含めないでください

                提供される検索結果に基づいて回答を構築してください。"""),
                ("human", """
                検索結果:
                {context}

                質問:
                {question}

                質問に対する回答を提供してください。""")
            ])

        # Determine which documents to use for context
        if has_blob_docs:
            # Prioritize blob documents when available
            context_docs = blob_docs
        else:
            # Use web documents when no blob documents are available
            context_docs = web_docs

        # If no context documents are available, provide a message
        if not context_docs:
            if debug_mode:
                st.write("⚠️ コンテキスト文書が見つかりませんでした")
            return {
                "documents": documents,
                "question": question,
                "generation": "申し訳ありませんが、質問に関連する情報が見つかりませんでした。"
            }

        # Generate context for the prompt
        context = "\n\n".join([doc.page_content for doc in context_docs])

        if debug_mode:
            st.write("### コンテキスト内容")
            st.write(context[:500] + "..." if len(context) > 500 else context)

        # Generate the answer
        generation_chain = prompt | llm | StrOutputParser()
        generation = generation_chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "documents": documents,
            "question": question,
            "generation": generation
        }


def decide_to_generate(state: GraphState) -> str:
    """
    Determines whether to generate an answer, transform the query, or perform web search.
    """
    need_search = state["search_results"]  # Updated from web_search to search_results
    documents = state["documents"]

    # Check if we have any documents at all
    has_documents = len(documents) > 0

    if debug_mode:
        st.write(f"### 次のステップを決定")
        st.write(f"Web検索が必要か: {'必要' if need_search == 'Yes' else '不要'}")
        st.write(f"文書が存在するか: {'はい' if has_documents else 'いいえ'}")

    if need_search == "Yes":
        # Need web search because no relevant documents found
        if debug_mode:
            st.write("決定: クエリを最適化してWeb検索を実行")
        return "transform_query"
    elif has_documents:
        # We have relevant documents, so generate answer
        if debug_mode:
            st.write("決定: 回答を生成")
        return "generate"
    else:
        # No documents at all, try web search as fallback
        if debug_mode:
            st.write("決定: ドキュメントがないためWeb検索を実行")
        return "transform_query"


# Initialize the workflow
workflow = StateGraph(GraphState)

# Add nodes with metadata
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search_fn)  # Renamed from web_search to web_search_node

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")  # Updated to web_search_node
workflow.add_edge("web_search_node", "generate")  # Updated to web_search_node
workflow.add_edge("generate", END)

app_chain = workflow.compile()

# Streamlit UI
st.title("Corrective RAG Demo")

# Sidebar
st.sidebar.title("設定")
debug_mode = st.sidebar.checkbox("デバッグモード", value=False)
show_langsmith = st.sidebar.checkbox("LangSmith トレース表示", value=False)

# Initialize session state for uploaded files
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = set()

# File Upload Section
st.sidebar.markdown("---")
st.sidebar.header("ドキュメントアップロード")
st.sidebar.markdown(f"サポートされているファイル形式: {', '.join(SUPPORTED_TYPES)}")

uploaded_files = st.sidebar.file_uploader(
    "ファイルを選択してください",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True
)

if uploaded_files:
    # Get new files that haven't been uploaded yet
    new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_file_names]

    if new_files:  # Only process new files
        upload_status = st.empty()
        for uploaded_file in new_files:
            if is_valid_file(uploaded_file):
                upload_status.info(f"{uploaded_file.name} をアップロード中...")
                result = upload_file_to_blob(uploaded_file, uploaded_file.type)
                if result["status"] == "success":
                    st.sidebar.success(result["message"])
                    # Add to uploaded files set
                    st.session_state.uploaded_file_names.add(uploaded_file.name)
                else:
                    st.sidebar.error(result["message"])
            else:
                st.sidebar.error(f"未対応のファイル形式です: {uploaded_file.name}")
        upload_status.empty()

question = st.text_input("質問を入力してください:", key="question_input")

if st.button("質問する"):
    if question:
        try:
            with st.spinner('処理中...'):
                response = app_chain.invoke({
                    "question": question,
                    "documents": [],
                    "generation": "",
                    "search_results": "No",  # Updated from web_search to search_results
                    "original_question": question
                })

            st.write("### 回答:")
            st.write(response["generation"])

            if debug_mode:
                st.write("### デバッグ情報:")
                blob_docs = [doc for doc in response["documents"] if doc.metadata.get('source_type', '') == 'blob']
                web_docs = [doc for doc in response["documents"] if doc.metadata.get('source_type', '') == 'web']

                st.json({
                    "質問": question,
                    "ドキュメント数": len(response["documents"]),
                    "Blob文書数": len(blob_docs),
                    "Web検索結果数": len(web_docs),
                    "Web検索実行": response.get("search_results") == "Yes"  # Updated from web_search to search_results
                })

            if show_langsmith:
                thread_id = str(hash(question))
                st.write("### LangSmith トレース情報")
                st.markdown(f"""
                    トレースを確認するには以下のURLにアクセスしてください:
                    https://smith.langchain.com/public/threads/{thread_id}/
                """)

            st.session_state.chat_history.append({
                "question": question,
                "answer": response["generation"]
            })

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            if debug_mode:
                import traceback
                st.write("詳細エラー情報:")
                st.code(traceback.format_exc())

if st.session_state.chat_history:
    st.write("### チャット履歴")
    for chat in reversed(st.session_state.chat_history):
        with st.expander(f"Q: {chat['question']}", expanded=False):
            st.write("**回答:**")
            st.write(chat['answer'])

st.markdown("""
    <style>
    .stTextInput input { font-size: 16px; }
    .stButton button {
        width: 100%;
        margin: 10px 0 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)