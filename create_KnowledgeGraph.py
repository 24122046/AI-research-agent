import json
import os
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import networkx as nx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import (
    ResponseSchema,
    StructuredOutputParser
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests
from xml.etree import ElementTree as ET
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from decouple import config

@dataclass
class PaperEntity:
    """Represents a entity in the knowledge graph"""
    Name : str
    ### Research context
    ResearchProblem : str
    Motivation: str
    ResearchGap : str
    Task : str

    ### Proposed solution
    Contribution : list[str]
    Method : str

    ### Evaluation
    Experiment : list[str]
    Metric : list[str]
    Dataset : list[str]
    Result : list[str]

    ### Interpretation
    Conclusion: list[str]
    Evidence: list[str]

    ### Critical information
    Limitation: list[str]
    FutureWork: list[str]

    ###Metadata
    Metadata :  dict
RelationType = [
    "cites",
    "uses_method_from",
    "uses_dataset_from",
    "extends",
    "improves",
    "compares_with",
    "addresses_limitation_of",
    "supports",
    "contradicts"
]
@dataclass
class PaperRelation:
    """Represents a relationship between Papers"""
    source: str
    target: str
    RelationType: str
    Description: str
    Evidence: str


class KnowledgeGraph:
    """Dynamic Knowledge Graph"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relations = []

    def add_entity(self, entity: PaperEntity):
        """Add a entity to the graph"""
        self.entities[entity.Name] = entity
        self.graph.add_node(
            entity.Name,
            entity.ResearchProblem,
            entity.Motivation,
            entity.Motivation,
            entity.ResearchGap,
            entity.Task,
            entity.Contribution,
            entity.Method,
            entity.Experiment,
            entity.Metric,
            entity.Dataset,
        )