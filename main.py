import logging
import os
import httpx
import json
from mcp.server.fastmcp import FastMCP
from mcp.types import (
    TextContent,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ragflow_mcp_server")

# Initialize FastMCP server
mcp = FastMCP("ragflow_mcp_server")

config = {
    "url": os.getenv("RAGFLOW_URL", "http://172.20.31.71:8000"),
    "api_key": os.getenv("RAGFLOW_API_KEY", "ragflow-cwMWZiMjk2MDllYTExZjA4MmQ3MDI0Mm"),
    "dataset_id": os.getenv("RAGFLOW_DATASET_ID", "dcd4b87e085e11f08c2b0242ac120005"),
    "similarity_threshold": float(os.getenv("RAGFLOW_SIMILARITY_THRESHOLD", 0.3)),
    "vector_similarity_weight": float(os.getenv("RAGFLOW_VECTOR_SIMILARITY_WEIGHT", 0.5)),
    "limit": int(os.getenv("RAGFLOW_LIMIT", 5)),
}

@mcp.tool()
async def retrieve_knowledge(query: str) -> str:
    """Retrieve knowledge related to the query from Ragflow

    Args:
        query (str): The query to retrieve knowledge for
    """
    api_url = f"{config['url']}/api/v1/retrieval"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    payload = {
        "question": query,
        "dataset_ids": [config["dataset_id"]],
        # "document_ids": [], # Optional: Add document IDs if needed
        # Add other parameters based on RAGflow API documentation if needed
        "similarity_threshold": config["similarity_threshold"],
        "vector_similarity_weight": config["vector_similarity_weight"],
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            
            formatted_knowledge_parts = []
            if "data" in result and "chunks" in result["data"]:
                for chunk in result["data"]["chunks"]:
                    if "content" in chunk:
                        try:
                            content_str = chunk["content"]
                            content_obj = json.loads(content_str)
                            title = content_obj.get("title", "未知标题")
                            description = content_obj.get("description", "无描述")
                            sql = content_obj.get("sql", "无 SQL")
                            
                            part = (
                                f"===== 文件名: {title} =====\n"
                                f"内容描述:\n{description}\n"
                                f"数据查询SQL:\n{sql}\n"
                                f"===== 文件名: {title} ====="
                            )
                            formatted_knowledge_parts.append(part)
                            if len(formatted_knowledge_parts) >= config["limit"]:
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON content in chunk: {chunk['content']}")
                        except KeyError as e:
                            logger.warning(f"Missing key {e} in decoded content object: {content_obj}")
                    else:
                        logger.warning(f"Chunk missing 'content' field: {chunk}")
            
            if not formatted_knowledge_parts:
                logger.warning(f"No processable knowledge found for query: {query}. Response: {result}")
                return [TextContent(type="text", text="未能找到或处理相关知识。")]

            knowledge = "\n\n".join(formatted_knowledge_parts) # 使用两个换行符分隔不同的条目
            return [TextContent(type="text", text=knowledge)]
        except httpx.RequestError as exc:
            logger.error(f"An error occurred while requesting {exc.request.url!r}: {exc}")
            return [TextContent(type="text", text=f"请求 RAG 服务时出错: {exc}")]
        except httpx.HTTPStatusError as exc:
            logger.error(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}: {exc.response.text}")
            return [TextContent(type="text", text=f"RAG 服务返回错误: {exc.response.status_code}")]

def main():
    logger.info("ragflow_mcp_server running with stdio transport")
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
