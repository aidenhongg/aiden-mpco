from pathlib import Path
from collections import namedtuple
import ast
from ast import FunctionDef, AsyncFunctionDef, ClassDef

NODE_TYPES = (FunctionDef, AsyncFunctionDef, ClassDef)
Snippet = namedtuple(
    "Snippet",
    ["rel_path", "base_indent", "code", "start_line", "end_line", "scope"],
)

def _node_to_obj(node, root_dir : Path):
    abs_path = node.filename

    with open(abs_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    relative_path = Path(abs_path).relative_to(root_dir)
    
    tree = ast.parse(''.join(lines), abs_path)
    node_dump = ast.dump(node, include_attributes=False)

    for node_tmp in ast.walk(tree):
        if node_dump == ast.dump(node_tmp, include_attributes=False):
            node = node_tmp
            break

    if hasattr(node, 'decorator_list') and node.decorator_list:
        start_line = node.decorator_list[0].lineno
    else:
        start_line = node.lineno

    end_line = node.end_lineno
    
    start_idx, end_idx = start_line - 1, end_line - 1
    base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    
    # get dedented snippet
    snippet = '\n'.join([line[base_indent:] for line in lines[start_idx : end_line]])
    
    # get enclosing scopes
    enclosing_scopes = _get_enclosing_scopes(tree, node)

    function = Snippet(
        rel_path=relative_path,
        base_indent=base_indent,
        code=snippet,
        start_line=start_idx,
        end_line=end_idx,
        scope=enclosing_scopes,
    )
        
    return function

def _get_enclosing_scopes(tree, target_node):
    parent_map = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    
    scopes = []
    current = target_node
    while current in parent_map:
        parent = parent_map[current]
        if isinstance(parent, NODE_TYPES):
            scope_type = 'class' if isinstance(parent, ClassDef) else 'function'
            scopes.append({'type': scope_type, 'name': parent.name})
        current = parent
    
    return list(reversed(scopes))  # outermost first
