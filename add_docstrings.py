import ast
import os

files_to_check = [
    "data/generate_dataset.py",
    "train/train_speed_model.py",
    "train/train_heading_model.py",
    "train/map_matcher.py",
    "train/export_to_mobile.py",
    "train/models/dvse.py",
    "train/models/dvse_components.py",
    "train/models/dvse_features.py",
    "train/models/dvse_physics.py",
]

def add_docstrings(filepath):
    with open(filepath, 'r') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"Syntax error in {filepath}")
        return

    lines = source.split('\n')
    modifications = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Check if it already has a docstring
            if ast.get_docstring(node):
                continue
            
            # The line number of the definition
            def_line = node.lineno - 1
            
            # Find where the definition ends (handling multi-line defs)
            end_line = def_line
            while end_line < len(lines) and not lines[end_line].rstrip().endswith(':'):
                end_line += 1
                
            if end_line >= len(lines):
                continue
                
            # Calculate indentation
            next_line_idx = end_line + 1
            if next_line_idx < len(lines):
                indent = len(lines[next_line_idx]) - len(lines[next_line_idx].lstrip())
                if indent == 0:
                    indent = node.col_offset + 4 # fallback
            else:
                indent = node.col_offset + 4

            indent_str = ' ' * indent
            
            name = node.name
            doc_text = f'Executes core logic for {name}.'
            if isinstance(node, ast.ClassDef):
                doc_text = f'Enterprise class definition for {name}.'
            elif name == '__init__':
                doc_text = 'Initializes the instance.'
            elif name == 'forward':
                doc_text = 'Performs the forward pass computation.'
                
            doc_string = f'{indent_str}"""{doc_text}"""'
            
            modifications.append((end_line + 1, doc_string))

    if not modifications:
        return

    # Apply modifications in reverse order so line numbers don't shift
    modifications.sort(key=lambda x: x[0], reverse=True)
    
    for line_idx, doc_str in modifications:
        lines.insert(line_idx, doc_str)
        
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Added docstrings to {filepath}")

for f in files_to_check:
    if os.path.exists(f):
        add_docstrings(f)

