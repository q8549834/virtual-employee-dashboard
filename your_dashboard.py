import streamlit as st
from crewai import Crew, Agent
import ollama
import crawl4ai
import os

# -------------------------------
# 禁用 CrewAI telemetry 避免 SIG 错误
# -------------------------------
os.environ["CREWAI_TELEMETRY_DISABLED"] = "1"

# -------------------------------
# 禁用 CrewAI fallback 调用 OpenAI
# -------------------------------
os.environ["CREWAI_DISABLE_OPENAI_FALLBACK"] = "1"

# -------------------------------
# Ollama 大模型调用函数（本地 Gemma3）
# -------------------------------
def call_model(prompt):
    try:
        response = ollama.chat(
            model="gemma3:12b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content
    except Exception as e:
        return f"[Ollama] 模型调用失败: {e}"
