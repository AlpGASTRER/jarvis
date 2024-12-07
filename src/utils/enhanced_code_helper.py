import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
import ast
import re
from dataclasses import dataclass

@dataclass
class CodeAnalysis:
    language: str
    complexity: float
    suggestions: List[str]
    code_blocks: List[str]
    references: List[str]

class EnhancedCodeHelper:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Language-specific configurations
        self.language_configs = {
            'python': {
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2048,
            },
            'javascript': {
                'temperature': 0.4,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 2048,
            },
            'java': {
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
        }
        
        # Initialize models for different languages
        self.models = {}
        for lang, config in self.language_configs.items():
            generation_config = genai.types.GenerationConfig(**config)
            self.models[lang] = genai.GenerativeModel('gemini-pro', generation_config=generation_config)
        
        # Default model for general queries
        default_config = genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
        self.default_model = genai.GenerativeModel('gemini-pro', generation_config=default_config)
        
    def detect_language(self, code: str) -> str:
        """Detect programming language from code snippet"""
        # Common language indicators
        indicators = {
            'python': ['.py', 'def ', 'import ', 'class ', 'print(', 'async def', 'await'],
            'javascript': ['.js', 'function', 'const ', 'let ', 'var ', '=>', 'async/await'],
            'java': ['.java', 'public class', 'private ', 'protected ', 'System.out'],
            'cpp': ['.cpp', '#include', 'std::', 'cout <<', 'namespace'],
            'typescript': ['.ts', 'interface ', 'type ', 'enum ', 'implements'],
        }
        
        code = code.lower()
        for lang, patterns in indicators.items():
            if any(pattern.lower() in code for pattern in patterns):
                return lang
        
        return 'unknown'
        
    def analyze_code(self, code: str) -> Optional[CodeAnalysis]:
        """Analyze code for complexity and potential improvements"""
        language = self.detect_language(code)
        
        try:
            if language == 'python':
                return self._analyze_python(code)
            # Add more language analyzers as needed
            
            return CodeAnalysis(
                language=language,
                complexity=0.0,
                suggestions=[],
                code_blocks=[],
                references=[]
            )
            
        except Exception as e:
            print(f"Code analysis error: {str(e)}")
            return None
            
    def _analyze_python(self, code: str) -> CodeAnalysis:
        """Analyze Python code"""
        try:
            tree = ast.parse(code)
            
            # Calculate complexity
            complexity = self._calculate_complexity(tree)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(tree)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(code)
            
            # Find relevant documentation
            references = self._find_references(code)
            
            return CodeAnalysis(
                language='python',
                complexity=complexity,
                suggestions=suggestions,
                code_blocks=code_blocks,
                references=references
            )
            
        except Exception as e:
            print(f"Python analysis error: {str(e)}")
            return None
            
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate code complexity using AST"""
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
        """Generate code improvement suggestions"""
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
        """Extract meaningful code blocks"""
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
        """Find relevant documentation references"""
        references = []
        
        # Extract imports
        imports = re.findall(r'import (\w+)|from (\w+)', code)
        for imp in imports:
            module = imp[0] or imp[1]
            references.append(f"https://docs.python.org/3/library/{module}.html")
            
        return references
        
    def get_code_help(self, query: str, code_context: Optional[str] = None) -> str:
        """Get enhanced code help with context awareness"""
        try:
            # Detect language if code context is provided
            language = self.detect_language(code_context) if code_context else 'unknown'
            
            # Select appropriate model
            model = self.models.get(language, self.default_model)
            
            # Analyze code if context is provided
            analysis = None
            if code_context:
                analysis = self.analyze_code(code_context)
            
            # Prepare prompt
            prompt = self._prepare_prompt(query, language, code_context, analysis)
            
            # Generate response
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
            
    def _prepare_prompt(self, query: str, language: str, code_context: Optional[str], analysis: Optional[CodeAnalysis]) -> str:
        """Prepare context-aware prompt"""
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
