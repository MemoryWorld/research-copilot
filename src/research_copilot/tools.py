"""Read-only, schema-checked tools. Never evaluate code or fetch user URLs."""
import ast
import math
import operator

from pydantic import BaseModel, ConfigDict, Field


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1200)


class CalculateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=120)


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def calculate(expression):
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 40:
        raise ValueError("Expression is too complex")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            result = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in {ast.UAdd, ast.USub}:
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            result = binary[type(node.op)](visit(node.left), visit(node.right))
        else:
            raise ValueError("Only numeric literals, parentheses and + - * / are allowed")
        if not math.isfinite(result) or abs(result) > 1e12:
            raise ValueError("Arithmetic value exceeds the allowed range")
        return result

    return visit(tree)


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": name, "description": description,
                                      "parameters": schema.model_json_schema()}}
    for name, description, schema in [
        ("search_knowledge", "Read-only search of user-uploaded knowledge, never URL fetching", SearchArgs),
        ("list_documents", "List uploaded document titles and IDs", NoArgs),
        ("calculator", "Evaluate bounded numeric arithmetic; no Python execution", CalculateArgs),
    ]
]


class ToolRegistry:
    def __init__(self, store, retriever):
        self.store, self.retriever = store, retriever

    def execute(self, name, arguments):
        if name == "search_knowledge":
            args = SearchArgs.model_validate(arguments)
            return self.retriever.search(args.query)
        if name == "list_documents":
            NoArgs.model_validate(arguments)
            return self.store.documents()
        if name == "calculator":
            args = CalculateArgs.model_validate(arguments)
            return {"expression": args.expression, "value": calculate(args.expression)}
        raise ValueError("Tool is not on the read-only allowlist")
