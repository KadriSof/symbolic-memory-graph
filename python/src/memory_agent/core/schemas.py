# python/src/memory_agent/core/schemas.py
# Pydantic Validation Schemas: contracts between the agent and LLM

from pydantic import BaseModel, Field, field_validator

from typing import Any, Literal, Union, Type


class NodeSchema(BaseModel):
    """Schema for graph node."""

    id: str = Field(..., description="Node ID")
    label: str = Field(..., description="Node label")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Node metadata")


class EdgeSchema(BaseModel):
    """Schema for graph edge."""

    id: str = Field(..., description="Edge ID")
    label: str = Field(..., description="Edge label")
    type: int = Field(..., description="Edge type (0=asymmetric, 1=symmetric)")
    connectiosn: Union[list[str], dict[str, Any]] = Field(
        ..., description="Edge connections"
    )
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Edge weight")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Edge metadata")


class EntitySchema(BaseModel):
    """Schema for entity extraction."""

    id: str = Field(..., description="Unique identifier for the entity")
    label: str = Field(..., description="Human-readable label")
    type: str | None = Field(
        None, description="Entity type (person, place, concept, etc.)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence score"
    )


class RelationSchema(BaseModel):
    """Schema for relation extraction."""

    source: str = Field(..., description="Source entity ID")
    target: str = Field(..., description="Target entity ID")
    label: str = Field(..., description="Relation label")
    direction: Literal["symmetric", "asymmetric"] = Field(
        default="asymmetric", description="Directionality"
    )
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Relation weight")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence score"
    )


class ComprehensionSchema(BaseModel):
    """Schema for comprehension output."""

    reconstructed_query: str = Field(..., description="Rephrased user query")
    user_intent: Literal["ask", "task", "clarify", "correct"] = Field(
        ..., description="User intent"
    )
    modus_operandi: Literal["REACT", "COGITO"] = Field(
        ..., description="Agent operation mode"
    )
    active_context: str = Field(..., description="Active context classification")
    entities: list[EntitySchema] = Field(
        default_factory=list, description="Extracted entities"
    )
    relations: list[RelationSchema] = Field(
        default_factory=list, description="Extracted relations"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Overall confidence"
    )

    @field_validator("modus_operandi")
    @classmethod
    def validate_mode(cls, mode: str) -> str:
        if mode not in ["REACT", "COGITO"]:
            raise ValueError(
                f"[ComprehensionSchema:validate_mode] modus_operandi must be REACT or COGITO, got '{mode}'"
            )
        return mode


class ConsolidationSchema(BaseModel):
    """Schema for consolidation output."""

    knowledge_summary: str = Field(
        default="", description="Summary of current knowledge"
    )
    new_nodes: list[NodeSchema] = Field(
        default_factory=list, description="New nodes to add"
    )
    new_edges: list[EdgeSchema] = Field(
        default_factory=list, description="New edges to add"
    )
    modified_nodes: list[NodeSchema] = Field(
        default_factory=list, description="Modified nodes"
    )
    modified_edges: list[EdgeSchema] = Field(
        default_factory=list, description="Modified edges"
    )
    gaps: list[str] = Field(default_factory=list, description="Missing information")
    conflicts: list[dict[str, Any]] = Field(
        default_factory=list, description="Knowledge conflicts"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Consolidation confidence"
    )


def get_json_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """
    Get JSON schema for a Pydantic mode.
    (this is used by LLM providers that support structured output via schema)
    """
    return model.model_json_schema()


def get_llm_format_instructions(model: Type[BaseModel]) -> str:
    """
    Generate Human-readable format instructions for LLM prompts.
    (this creates instructions that tell the LLM exactly what JSON structure to return)
    """
    schema = get_json_schema(model)
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    lines = ["Return a JSON object with the following structure:"]
    lines.append("{")

    for prop_name, prop_info in properties.items():
        required_marker = " (required)" if prop_name in required else " (optional)"
        prop_type = prop_info.get("type", "any")
        description = prop_info.get("description", "")

        # Hanle nested objects:
        if prop_type == "array":
            items = prop_info.get("items", {})
            item_type = items.get("types", "any")
            lines.append(
                f' "{prop_name}": [{{ ... }}]  # {description} (array of {item_type}){required_marker}'
            )
        elif prop_type == "object":
            lines.append(f' "{prop_name}": {{ ... }}  # {description}{required_marker}')
        else:
            lines.append(
                f' "{prop_name}": <{prop_type}>  # {description}{required_marker}'
            )

    lines.append("}")
    lines.append("")
    lines.append("Rules:")
    lines.append("1. Return ONLY valid JSON - no explanations, markdown, or extra text")
    lines.append("2. Include all required fields")
    lines.append("3. Use proper types (strings in quotes, booleans as true/false)")
    lines.append("4. No trailing commas")
    lines.append("5. If you cannot provide a value, use null or omit optional fields")

    return "\n".join(lines)
