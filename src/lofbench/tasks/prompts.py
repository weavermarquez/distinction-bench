"""Prompt templates for LoF evaluation tasks."""

SINGLE_SYSTEM_PROMPT = """You are an expert in evaluating Laws of Form expressions.
Your task is to analyze an expression that represents a structure of distinctions
and reduce it to its simplest form using two fundamental axioms.

## The form and axioms

### Axiom 1. The law of calling
Multiple adjacent boundaries with nothing else inside them condense into one.
Example: ()() = ()

### Axiom 2. The law of crossing
Two nested boundaries with nothing else between them annihilate to nothing.
Example: (()) = nothing

## Instructions
1. Identify the structure 
2. Look for opportunities to apply axioms (I1 or I2)
3. Apply reductions iteratively until no more reductions are possible
4. State whether the final result is marked or unmarked

If structure remains, answer marked. Answer unmarked if the expression reduces to nothing."""

SINGLE_USER_TEMPLATE = """
Here is the expression you need to evaluate:
--INPUT--
{expression}

Your response:
"""

COMPOSITE_SYSTEM_PROMPT = """You are an expert in evaluating Laws of Form expressions.
Your task is to analyze multiple expressions and determine how many reduce to marked.

## The form and axioms

### Axiom 1. The law of calling
Multiple adjacent boundaries condense into one: ()() = ()

### Axiom 2. The law of crossing
Two nested boundaries annihilate to nothing: (()) = nothing

## Instructions

1. For each expression, identify the structure
3. Apply axioms (I1 or I2) iteratively until no more reductions are possible
4. Determine if each reduces to marked or unmarked
5. Count the total number of marked expressions

After working through ALL expressions, provide your final answers in this exact JSON format:

```json
{{"E1": "marked", "E2": "unmarked", ..., "total_marked": N}}
```
Where each E# is either "marked" or "unmarked", and total_marked is the count of marked expressions.
"""

COMPOSITE_USER_TEMPLATE = """Here are the expressions you need to evaluate:
--INPUT--
{expressions}

Your response:
"""
