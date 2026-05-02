"""LLM call wrapper with reliable JSON parsing."""
import json
from typing import Type, TypeVar
import litellm
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(Exception):
    """Raised when LLM call or parsing fails."""
    pass


def call_json(
    prompt: str,
    schema: Type[T],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> T:
    """Call LLM and parse response as a pydantic model.

    Uses JSON mode where supported. Raises LLMError with clear messages
    on failure (which is what ragas often fails to do).
    """
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise LLMError(f"LLM call failed for model '{model}': {e}") from e

    content = response.choices[0].message.content
    if not content:
        raise LLMError(f"LLM returned empty content. Model: {model}")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMError(
            f"Could not parse JSON from LLM. First 500 chars: {content[:500]}"
        ) from e

    try:
        return schema.model_validate(data)
    except Exception as e:
        raise LLMError(
            f"LLM output didn't match schema {schema.__name__}. Got: {data}"
        ) from e
