# TMN_DataGen/TMN_DataGen/__init__.py
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simple format to just show messages for tree visualization
    force=True  # Ensure our config takes precedence
)

logger = logging.getLogger(__name__)

from .tree.node import Node
from .tree.dependency_tree import DependencyTree
from .parsers.diaparser_impl import DiaParserTreeParser
from .parsers.spacy_impl import SpacyTreeParser
from .dataset_generator import DatasetGenerator
from .parsers.multi_parser import MultiParser
from .utils.feature_utils import FeatureExtractor
from .utils.viz_utils import print_tree_text, visualize_tree_graphviz, format_tree_pair

__all__ = [
    'Node', 'DependencyTree',
    'DiaParserTreeParser', 'SpacyTreeParser', 'MultiParser',
    'DatasetGenerator', 'FeatureExtractor',
    'print_tree_text', 'visualize_tree_graphviz', 'format_tree_pair'
]
