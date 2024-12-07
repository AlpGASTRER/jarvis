"""
Enhanced Code Helper Module

This module provides advanced code analysis and assistance capabilities using
Google's Gemini AI model. It supports multiple programming languages and offers
various code analysis features including complexity calculation, best practices,
security analysis, and improvement suggestions.

Key Features:
- Multi-language support (Python, JavaScript, TypeScript, Java, etc.)
- Code complexity analysis
- Language detection
- Best practices recommendations
- Security analysis
- Code improvement suggestions
- Context-aware code help
- Documentation reference finding

Dependencies:
- google.generativeai: For AI-powered code analysis
- ast: For Python code parsing and analysis
- dotenv: For environment variable management
- re: For pattern matching in code
"""

from __future__ import annotations
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Union, Tuple, TypeVar, Callable
import ast
import re
from dataclasses import dataclass
from pathlib import Path
import json
from abc import ABC, abstractmethod

# Custom exceptions for better error handling
class CodeHelperError(Exception):
    """Base exception for code helper errors."""
    pass

class LanguageNotSupportedError(CodeHelperError):
    """Raised when an unsupported language is encountered."""
    pass

class ConfigurationError(CodeHelperError):
    """Raised when there's an issue with configuration."""
    pass

class AnalysisError(CodeHelperError):
    """Raised when code analysis fails."""
    pass

@dataclass
class CodeAnalysis:
    """
    Data class for storing code analysis results.
    
    Attributes:
        language: Detected programming language
        complexity: Calculated code complexity score
        suggestions: List of improvement suggestions
        code_blocks: List of extracted code blocks
        references: List of relevant documentation links
        metrics: Additional language-specific metrics
    """
    language: str
    complexity: float
    suggestions: List[str]
    code_blocks: List[str]
    references: List[str]
    metrics: Dict[str, Any] = None

# Type aliases for better code readability
ModelConfig = Dict[str, Union[float, int]]
LanguageConfig = Dict[str, ModelConfig]
T = TypeVar('T')

class LanguageAnalyzer(ABC):
    """Abstract base class for language-specific analyzers."""
    
    @abstractmethod
    def analyze(self, code: str) -> CodeAnalysis:
        """Analyze code and return results."""
        pass
    
    @abstractmethod
    def calculate_complexity(self, code: str) -> float:
        """Calculate language-specific complexity."""
        pass

class PythonAnalyzer(LanguageAnalyzer):
    """Python-specific code analyzer."""
    
    def analyze(self, code: str) -> CodeAnalysis:
        try:
            tree = ast.parse(code)
            complexity = self.calculate_complexity(code)
            suggestions = self._generate_suggestions(tree)
            metrics = self._calculate_metrics(tree)
            
            return CodeAnalysis(
                language='python',
                complexity=complexity,
                suggestions=suggestions,
                code_blocks=self._extract_code_blocks(code),
                references=self._find_references(code),
                metrics=metrics
            )
        except SyntaxError as e:
            raise AnalysisError(f"Python syntax error: {str(e)}")
        except Exception as e:
            raise AnalysisError(f"Python analysis failed: {str(e)}")
    
    def calculate_complexity(self, code: str) -> float:
        try:
            tree = ast.parse(code)
            return self._calculate_cyclomatic_complexity(tree)
        except Exception as e:
            raise AnalysisError(f"Complexity calculation failed: {str(e)}")
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> float:
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _calculate_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        return {
            'num_functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            'num_classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            'num_imports': len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
            'lines_of_code': len(ast.unparse(tree).splitlines())
        }

    def _generate_suggestions(self, tree: ast.AST) -> List[str]:
        """Generate code improvement suggestions based on AST analysis."""
        suggestions = []
        
        # Check for function and class documentation
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not ast.get_docstring(node):
                suggestions.append(f"Add docstring to {node.__class__.__name__.lower()} '{node.name}'")
        
        # Check for type hints in function arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self':
                        suggestions.append(f"Add type hint for argument '{arg.arg}' in function '{node.name}'")
        
        # Check for overly complex functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_function_complexity(node)
                if complexity > 10:
                    suggestions.append(f"Consider breaking down function '{node.name}' (complexity: {complexity})")
        
        return suggestions

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate complexity score for a single function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                complexity += 1
        
        return complexity

    def _extract_code_blocks(self, code: str) -> List[str]:
        """Extract meaningful code blocks."""
        blocks = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    blocks.append(ast.unparse(node))
        except Exception:
            pass
        return blocks

    def _find_references(self, code: str) -> List[str]:
        """Find relevant documentation references."""
        references = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        references.append(f"https://docs.python.org/3/library/{name.name}.html")
                elif isinstance(node, ast.ImportFrom):
                    references.append(f"https://docs.python.org/3/library/{node.module}.html")
        except Exception:
            pass
        return references

class EnhancedCodeHelper:
    """
    A class providing enhanced code analysis and assistance using Gemini AI.
    
    This class combines traditional static code analysis with AI-powered
    insights to provide comprehensive code assistance. It supports multiple
    programming languages and provides various analysis features.
    
    Attributes:
        language_configs: Configuration settings for each supported language
        models: Dictionary of language-specific Gemini models
        default_model: Default Gemini model for general queries
        analyzers: Dictionary of language-specific analyzers
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the code helper with language-specific configurations.
        
        Args:
            config_path: Optional path to custom configuration file
            
        Raises:
            ConfigurationError: If configuration loading fails
            ValueError: If Google API key is not found
        """
        self._load_environment()
        self._load_configurations(config_path)
        self._initialize_models()
        self._initialize_analyzers()
    
    def _load_environment(self) -> None:
        """Load environment variables."""
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ConfigurationError("Google API key not found in environment variables")
        genai.configure(api_key=api_key)
    
    def _load_configurations(self, config_path: Optional[str]) -> None:
        """Load language configurations from file or defaults."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    self.language_configs = json.load(f)
            except Exception as e:
                raise ConfigurationError(f"Failed to load configuration: {str(e)}")
        else:
            # Default configurations remain unchanged
            self.language_configs = {
                'python': {
                    'temperature': 0.3,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                },
                # ... other language configs ...
            }
    
    def _initialize_models(self) -> None:
        """Initialize AI models for each supported language."""
        try:
            self.models = {
                lang: genai.GenerativeModel(
                    'gemini-pro',
                    generation_config=genai.types.GenerationConfig(**config)
                )
                for lang, config in self.language_configs.items()
            }
            
            self.default_model = genai.GenerativeModel(
                'gemini-pro',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                )
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize models: {str(e)}")
    
    def _initialize_analyzers(self) -> None:
        """Initialize language-specific analyzers."""
        self.analyzers = {
            'python': PythonAnalyzer(),
            # Add more language analyzers here
        }

    def detect_language(self, code: str) -> str:
        """
        Detect programming language from code snippet.
        
        Uses a combination of language indicators and pattern matching to
        determine the programming language of the provided code snippet.
        
        Args:
            code: Code snippet to analyze
            
        Returns:
            str: Detected programming language
        """
        # Common language indicators
        indicators = {
            'python': ['.py', 'def ', 'import ', 'class ', 'print(', 'async def', 'await', '->'],
            'javascript': ['.js', 'function', 'const ', 'let ', 'var ', '=>', 'async/await', 'console.log'],
            'typescript': ['.ts', 'interface ', 'type ', 'enum ', 'implements', 'export', 'private', 'public'],
            'java': ['.java', 'public class', 'private ', 'protected ', 'System.out', 'void ', 'extends'],
            'cpp': ['.cpp', '#include', 'std::', 'cout <<', 'namespace', 'template<', 'vector<'],
            'csharp': ['.cs', 'using System', 'namespace', 'public class', 'private ', 'protected', 'async Task'],
            'go': ['.go', 'package ', 'func ', 'import (', 'fmt.', 'struct {', 'interface {'],
            'rust': ['.rs', 'fn ', 'let mut', 'impl ', 'trait ', 'pub ', 'use std'],
            'ruby': ['.rb', 'def ', 'class ', 'require ', 'module ', 'attr_', 'puts '],
            'php': ['.php', '<?php', 'function ', 'class ', 'public function', 'namespace', '$'],
            'swift': ['.swift', 'func ', 'var ', 'let ', 'class ', 'struct ', 'protocol '],
            'kotlin': ['.kt', 'fun ', 'val ', 'var ', 'class ', 'data class', 'suspend ']
        }
        
        code = code.lower()
        for lang, patterns in indicators.items():
            if any(pattern.lower() in code for pattern in patterns):
                return lang
        
        return 'unknown'
        
    def analyze_code(self, code: str) -> CodeAnalysis:
        """
        Analyze code for complexity and potential improvements.
        
        Args:
            code: Code to analyze
            
        Returns:
            CodeAnalysis: Analysis results
            
        Raises:
            LanguageNotSupportedError: If language is not supported
            AnalysisError: If analysis fails
        """
        language = self.detect_language(code)
        if language == 'unknown':
            raise LanguageNotSupportedError("Could not detect programming language")
            
        try:
            # Use language-specific analyzer if available
            analyzer = self.analyzers.get(language)
            if analyzer:
                return analyzer.analyze(code)
            
            # Fallback to generic analysis
            return self._generic_analysis(code, language)
        except Exception as e:
            raise AnalysisError(f"Analysis failed: {str(e)}")
    
    def _generic_analysis(self, code: str, language: str) -> CodeAnalysis:
        """Perform generic code analysis for unsupported languages."""
        try:
            metrics = self._calculate_generic_metrics(code)
            complexity = metrics.get('cyclomatic_complexity', 1.0)
            suggestions = self.get_suggestions(code, language)
            code_blocks = self._extract_code_blocks(code)
            references = self._find_references(code)
            
            return CodeAnalysis(
                language=language,
                complexity=complexity,
                suggestions=suggestions,
                code_blocks=code_blocks,
                references=references,
                metrics=metrics
            )
        except Exception as e:
            raise AnalysisError(f"Generic analysis failed: {str(e)}")
    
    def _calculate_generic_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate generic code metrics."""
        lines = code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        metrics = {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'average_line_length': sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0,
            'cyclomatic_complexity': 1.0  # Base complexity
        }
        
        # Basic complexity factors
        complexity_indicators = [
            'if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'catch ', 'switch',
            'case ', 'break', 'continue', '?', '&&', '||', '==', '!=', '>=', '<='
        ]
        
        # Add to complexity for each control flow indicator
        for indicator in complexity_indicators:
            metrics['cyclomatic_complexity'] += code.lower().count(indicator) * 0.1
            
        return metrics

    def analyze_syntax(self, code: str, language: str = None) -> Dict:
        """
        Analyze code syntax for any supported language.
        
        Uses the Gemini model to analyze the code syntax and provide feedback
        on correctness and best practices.
        
        Args:
            code: Code to analyze
            language: Programming language (auto-detected if not specified)
            
        Returns:
            Dict: Analysis results, including success flag and error message
        """
        if not language:
            language = self.detect_language(code)

        model = self.models.get(language, self.default_model)
        prompt = f"""Analyze the following {language} code for syntax correctness and best practices:

{code}

Provide analysis in the following format:
1. Syntax correctness
2. Best practices
3. Potential improvements
4. Security considerations
"""
        try:
            response = model.generate_content(prompt)
            return {
                'success': True,
                'analysis': response.text,
                'language': language
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': language
            }

    def get_code_help(self, query: str, code_context: Optional[str] = None) -> str:
        """
        Get enhanced code help with context awareness.
        
        This method combines code analysis with AI-powered assistance to provide
        contextually relevant help. It considers the programming language,
        code context, and any available analysis results.
        
        Args:
            query: User's question or request
            code_context: Optional code snippet for context
            
        Returns:
            str: Detailed response addressing the query
            
        Note:
            The response is tailored based on:
            - Detected programming language
            - Code complexity analysis
            - Identified best practices
            - Relevant documentation references
        """
        try:
            # Detect language if code context is provided
            language = self.detect_language(code_context) if code_context else 'unknown'
            
            # Select appropriate model based on language
            model = self.models.get(language, self.default_model)
            
            # Perform code analysis if context is provided
            analysis = None
            if code_context:
                analysis = self.analyze_code(code_context)
            
            # Generate context-aware prompt
            prompt = self._prepare_prompt(query, language, code_context, analysis)
            
            # Get AI-powered response
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
            
    def _prepare_prompt(self, query: str, language: str, code_context: Optional[str], analysis: Optional[CodeAnalysis]) -> str:
        """
        Prepare a context-aware prompt for the AI model.
        
        Combines various pieces of information to create a comprehensive prompt
        that helps the AI model provide more relevant and accurate responses.
        
        Args:
            query: User's question
            language: Detected programming language
            code_context: Code snippet for context
            analysis: Code analysis results if available
            
        Returns:
            str: Formatted prompt combining all available information
        """
        prompt = [
            "As an expert programming assistant, help with this question.",
            f"\nQuestion: {query}"
        ]
        
        if language != 'unknown':
            prompt.append(f"\nLanguage: {language}")
            
        if code_context:
            prompt.append(f"\nCode Context:\n{code_context}")
            
        if analysis:
            prompt.append("\nCode Analysis:")
            prompt.append(f"- Complexity Score: {analysis.complexity:.2f}")
            if analysis.suggestions:
                prompt.append("- Suggestions:")
                for suggestion in analysis.suggestions:
                    prompt.append(f"  • {suggestion}")
            if analysis.references:
                prompt.append("- Relevant Documentation:")
                for ref in analysis.references:
                    prompt.append(f"  • {ref}")
                    
        return "\n".join(prompt)

    def get_suggestions(self, code: str, language: str = None) -> List[str]:
        """
        Get code improvement suggestions.
        
        Uses AI to analyze code and provide specific suggestions for improvement
        focusing on organization, performance, error handling, and best practices.
        
        Args:
            code: Code to analyze
            language: Programming language (auto-detected if not specified)
            
        Returns:
            List[str]: List of improvement suggestions
        """
        if not language:
            language = self.detect_language(code)

        model = self.models.get(language, self.default_model)
        prompt = f"""Analyze the following {language} code and provide specific suggestions for improvement:

{code}

Focus on:
1. Code organization
2. Performance optimization
3. Error handling
4. Code readability
5. Best practices
"""
        try:
            response = model.generate_content(prompt)
            suggestions = [s.strip() for s in response.text.split('\n') if s.strip()]
            return suggestions
        except Exception as e:
            print(f"Error generating suggestions: {str(e)}")
            return []

    def get_best_practices(self, code: str, language: str = None) -> List[str]:
        """
        Get language-specific best practices.
        
        Analyzes code against established best practices for the specific
        programming language, considering conventions, patterns, and standards.
        
        Args:
            code: Code to analyze
            language: Programming language (auto-detected if not specified)
            
        Returns:
            List[str]: List of best practice recommendations
        """
        if not language:
            language = self.detect_language(code)

        model = self.models.get(language, self.default_model)
        prompt = f"""Review the following {language} code and list specific best practices that should be followed:

{code}

Focus on:
1. Language-specific conventions
2. Design patterns
3. Documentation standards
4. Testing considerations
5. Maintainability
"""
        try:
            response = model.generate_content(prompt)
            practices = [p.strip() for p in response.text.split('\n') if p.strip()]
            return practices
        except Exception as e:
            print(f"Error getting best practices: {str(e)}")
            return []

    def analyze_security(self, code: str, language: str = None) -> List[str]:
        """
        Analyze code for security issues.
        
        Performs security analysis focusing on common vulnerabilities,
        input validation, authentication, and secure coding practices.
        
        Args:
            code: Code to analyze
            language: Programming language (auto-detected if not specified)
            
        Returns:
            List[str]: List of identified security issues
        """
        if not language:
            language = self.detect_language(code)

        model = self.models.get(language, self.default_model)
        prompt = f"""Analyze the following {language} code for potential security issues:

{code}

Focus on:
1. Input validation
2. Authentication/Authorization
3. Data sanitization
4. Common vulnerabilities
5. Secure coding practices
"""
        try:
            response = model.generate_content(prompt)
            issues = [i.strip() for i in response.text.split('\n') if i.strip()]
            return issues
        except Exception as e:
            print(f"Error analyzing security: {str(e)}")
            return []

    def calculate_complexity(self, code: str, language: str = None) -> float:
        """
        Calculate code complexity score.
        
        Computes a complexity score based on various metrics including control
        flow, nesting, and language-specific features. Uses specialized
        calculations for Python and Rust, with a generic approach for others.
        
        Args:
            code: Code to analyze
            language: Programming language (auto-detected if not specified)
            
        Returns:
            float: Calculated complexity score
            
        Note:
            Scores are relative and should be used for comparison rather
            than absolute measurement.
        """
        if not language:
            language = self.detect_language(code)

        try:
            if language == 'python':
                return self._calculate_python_complexity(code)
            elif language == 'rust':
                return self._calculate_rust_complexity(code)
            else:
                # Generic complexity calculation for other languages
                complexity = 0.0
                
                # Control flow complexity (if, else, loops, etc.)
                complexity += len(re.findall(r'\b(if|else|for|while|switch|case|try|catch)\b', code)) * 0.5
                
                # Function/method complexity
                complexity += len(re.findall(r'\b(function|def|fn|func|method)\b', code)) * 0.3
                
                # Class/type complexity
                complexity += len(re.findall(r'\b(class|struct|interface|trait|impl)\b', code)) * 0.4
                
                # Nesting complexity (nested blocks)
                complexity += len(re.findall(r'[{]\s*[^}]*[{]', code)) * 0.6
                
                return round(complexity, 2)
                
        except Exception as e:
            print(f"Error calculating complexity: {str(e)}")
            return 0.0

    def _calculate_python_complexity(self, code: str) -> float:
        """
        Calculate Python-specific complexity.
        
        Analyzes the provided Python code to calculate a complexity score based
        on various metrics, including control flow, nesting, and function complexity.
        
        Args:
            code: Python code to analyze
            
        Returns:
            float: Calculated complexity score
        """
        try:
            tree = ast.parse(code)
            complexity = self._calculate_complexity(tree)
            return round(complexity, 2)
        except:
            return 0.0

    def _calculate_rust_complexity(self, code: str) -> float:
        """
        Calculate Rust-specific complexity.
        
        Analyzes the provided Rust code to calculate a complexity score based
        on various metrics, including control flow, pattern matching, and type complexity.
        
        Args:
            code: Rust code to analyze
            
        Returns:
            float: Calculated complexity score
        """
        try:
            complexity = 0.0
            
            # Control flow complexity
            complexity += len(re.findall(r'\b(if|else|match|for|while|loop)\b', code)) * 0.5
            
            # Function complexity
            complexity += len(re.findall(r'\bfn\s+\w+', code)) * 0.3
            
            # Type complexity
            complexity += len(re.findall(r'\b(struct|enum|trait|impl)\b', code)) * 0.4
            
            # Generic/lifetime complexity
            complexity += len(re.findall(r'<[^>]+>', code)) * 0.3
            
            # Pattern matching complexity
            complexity += len(re.findall(r'\bmatch\b[^{]*{([^}]*})*', code)) * 0.6
            
            return round(complexity, 2)
        except:
            return 0.0

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """
        Calculate code complexity using AST.
        
        Analyzes the provided AST to calculate a complexity score based on
        various metrics, including control flow, nesting, and function complexity.
        
        Args:
            tree: AST to analyze
            
        Returns:
            float: Calculated complexity score
        """
        complexity = 0
        
        for node in ast.walk(tree):
            # Count control flow statements
            if isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef)):
                complexity += 1
            # Add complexity for nested functions and comprehensions
            elif isinstance(node, (ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp)):
                complexity += 0.5
            # Add complexity for exception handling
            elif isinstance(node, ast.Try):
                complexity += 0.3
                
        return complexity
        
    def _generate_suggestions(self, tree: ast.AST) -> List[str]:
        """
        Generate code improvement suggestions.
        
        Analyzes the provided AST to generate suggestions for improving the code,
        including refactoring, simplification, and best practices.
        
        Args:
            tree: AST to analyze
            
        Returns:
            List[str]: List of improvement suggestions
        """
        suggestions = []
        
        for node in ast.walk(tree):
            # Check for long functions
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 20:
                    suggestions.append(f"Consider breaking down function '{node.name}' into smaller functions")
                if len(node.args.args) > 5:
                    suggestions.append(f"Function '{node.name}' has many parameters. Consider using a class or dataclass")
                    
            # Check for nested loops
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        suggestions.append("Consider refactoring nested loops to improve readability and performance")
                        break
                        
            # Check for multiple return statements
            if isinstance(node, ast.FunctionDef):
                return_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Return))
                if return_count > 3:
                    suggestions.append(f"Function '{node.name}' has multiple return points. Consider consolidating them")
                    
        return suggestions
        
    def _extract_code_blocks(self, code: str) -> List[str]:
        """
        Extract meaningful code blocks.
        
        Splits the provided code into individual blocks, such as functions,
        classes, and loops.
        
        Args:
            code: Code to extract blocks from
            
        Returns:
            List[str]: List of extracted code blocks
        """
        blocks = []
        current_block = []
        
        for line in code.split('\n'):
            if line.strip():
                current_block.append(line)
            elif current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
                
        if current_block:
            blocks.append('\n'.join(current_block))
            
        return blocks
        
    def _find_references(self, code: str) -> List[str]:
        """
        Find relevant documentation references.
        
        Extracts import statements and module names from the provided code to
        find relevant documentation references.
        
        Args:
            code: Code to find references for
            
        Returns:
            List[str]: List of relevant documentation references
        """
        references = []
        
        # Extract imports
        imports = re.findall(r'import (\w+)|from (\w+)', code)
        for imp in imports:
            module = imp[0] or imp[1]
            references.append(f"https://docs.python.org/3/library/{module}.html")
            
        return references
