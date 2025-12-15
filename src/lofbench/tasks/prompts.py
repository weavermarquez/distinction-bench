"""Prompt templates for LoF evaluation tasks."""

SINGLE_SYSTEM_PROMPT = """You are an expert in evaluating Laws of Form expressions.
Your task is to analyze an expression that represents a structure of distinctions
and reduce it to its simplest form using two fundamental axioms."""

SINGLE_USER_TEMPLATE = """Here is the expression you need to evaluate:

<expression>
{input}
</expression>

## The form and axioms

#### Axiom 1. The law of calling
Multiple adjacent boundaries with nothing else inside them condense into one.
Example: ()() = ()

#### Axiom 2. The law of crossing
Two nested boundaries with nothing else between them annihilate to void.
Example: (()) = void

## Instructions

1. Identify the structure
2. Look for opportunities to apply axioms (I1 or I2)
3. Apply reductions iteratively until no more reductions are possible
4. State whether the final result is marked (reduces to ()) or unmarked (reduces to void)

After completing your reduction, provide your final answer in this exact format:

<answer>X</answer>

where X is either:
- unmarked (if the expression reduces to void)
- marked (if structure remains)"""


COMPOSITE_SYSTEM_PROMPT = """You are an expert in evaluating Laws of Form expressions.
Your task is to analyze multiple expressions and determine how many reduce to marked."""

COMPOSITE_USER_TEMPLATE = """Here are the expressions you need to evaluate:

<expressions>
{input}
</expressions>

#### Axiom 1. The law of calling
Multiple adjacent boundaries condense into one: ()() = ()

#### Axiom 2. The law of crossing
Two nested boundaries annihilate to void: (()) = void

## Instructions

1. For each expression, identify the structure
2. Apply axioms (I1 or I2) iteratively
3. Determine if each reduces to marked () or unmarked (void)
4. Count the total number of marked expressions

After working through ALL expressions, provide your final answers in this exact JSON format:

```json
{{"E1": "marked", "E2": "unmarked", ..., "total_marked": N}}
```

Where each E# is either "marked" or "unmarked", and total_marked is the count of
marked expressions."""
