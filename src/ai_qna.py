import os
from typing import Optional

import pandas as pd

# Support both old (0.28.x) and new (>=1.x) OpenAI SDKs without crashing on import
try:  # new SDK style
    from openai import OpenAI  # type: ignore
    _HAS_CLIENT = True
except ImportError:  # old SDK style
    import openai  # type: ignore

    OpenAI = None  # type: ignore
    _HAS_CLIENT = False


class DataframeAssistant:
    def __init__(self, df: pd.DataFrame, openai_api_key: Optional[str] = None, verbose: bool = False):
        key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in environment or Streamlit secrets.")
        self.df = df
        self.verbose = verbose

        if _HAS_CLIENT and OpenAI is not None:
            # New client-based API
            self.client = OpenAI(api_key=key)
            self.openai_legacy = None
        else:
            # Legacy global API
            openai.api_key = key  # type: ignore[name-defined]
            self.client = None
            self.openai_legacy = openai  # type: ignore[name-defined]

    def answer(self, question: str) -> str:
        # Use plain text preview to avoid extra dependencies like `tabulate`
        preview = self.df.head(10).to_string(index=False)
        schema = ", ".join([f"{c} ({str(self.df[c].dtype)})" for c in self.df.columns])
        prompt = (
            "You are a data analyst. The user will ask questions about a pandas DataFrame.\n"
            "You are given the column schema and a small sample of the data.\n"
            "Answer using clear bullet points and include any aggregations they ask for.\n"
            "If you are unsure because the sample is too small, say so explicitly.\n\n"
            f"Columns: {schema}\n\nSample (first 10 rows):\n{preview}\n\n"
            f"User question: {question}"
        )

        if self.client is not None:
            resp = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content.strip()

        # Legacy style (0.28.x)
        resp = self.openai_legacy.ChatCompletion.create(  # type: ignore[union-attr]
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return resp.choices[0].message["content"].strip()
